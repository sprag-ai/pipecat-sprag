import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from helpers import run_until
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    TTSTextFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.text.base_text_aggregator import AggregationType

from pipecat_sprag.turns import (
    SpragAgentTextObserver,
    SpragTurnAnalyzer,
    SpragTurnDetector,
    SpragTurnParams,
)

# One /v1/turns session recorded from the sprag-turns model: every server event it sent.
SESSION = json.loads((Path(__file__).parent / "data" / "turns_session.json").read_text())
FRAME_16K = b"\x01\x00" * 320  # 20 ms at 16 kHz


class _TurnsServer:
    """Replays the recorded session's turn events once ``replay_after`` bytes of audio arrive."""

    def __init__(self, replay_after: int = 6400, refuse: str | None = None) -> None:
        self.replay_after = replay_after
        self.refuse = refuse
        self.handshakes: list[web.Request] = []
        self.events: list[dict[str, Any]] = []
        self.audio_bytes = 0

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.handshakes.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        server_events = list(SESSION["server"])
        await ws.send_json(server_events.pop(0))
        updated = server_events.pop(0)
        replayed = False
        async for msg in ws:
            if msg.type is WSMsgType.BINARY:
                self.audio_bytes += len(msg.data)
                if not replayed and self.audio_bytes >= self.replay_after:
                    replayed = True
                    for event in server_events:
                        await ws.send_json(event)
                continue
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            self.events.append(event)
            if event["type"] == "session.update":
                if self.refuse:
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": {"code": "invalid_value", "message": self.refuse},
                        }
                    )
                else:
                    await ws.send_json(updated)
            elif event["type"] == "session.close":
                await ws.close()
        return ws

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]


@pytest.fixture
async def turns() -> AsyncIterator[Callable[..., Awaitable[tuple[_TurnsServer, str]]]]:
    servers: list[TestServer] = []

    async def start(**kwargs: Any) -> tuple[_TurnsServer, str]:
        server = _TurnsServer(**kwargs)
        app = web.Application()
        app.router.add_get("/v1/turns", server.handle)
        test_server = TestServer(app)
        await test_server.start_server()
        servers.append(test_server)
        return server, str(test_server.make_url("/v1/turns")).replace("http://", "ws://")

    yield start
    for test_server in servers:
        await test_server.close()


def _audio(count: int, sample_rate: int = 16000) -> list[Frame]:
    chunk = b"\x01\x00" * (sample_rate // 50)
    return [
        InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=1)
        for _ in range(count)
    ]


def _completed(down: list[Frame], up: list[Frame]) -> bool:
    return any(isinstance(f, UserTurnInferenceCompletedFrame) for f in down)


async def test_the_detector_pushes_the_turn_complete_when_the_model_ends_it(turns) -> None:
    server, url = await turns()
    detector = SpragTurnDetector(
        api_key="sk-test", url=url, params=SpragTurnParams(eot_timeout_ms=1500)
    )
    ended: list[dict[str, Any]] = []

    @detector.event_handler("on_turn_ended")
    async def _on_ended(processor: SpragTurnDetector, event: dict[str, Any]) -> None:
        ended.append(event)

    down, _ = await run_until(detector, _audio(20), _completed)
    assert sum(isinstance(f, UserTurnInferenceCompletedFrame) for f in down) == 1
    assert sum(isinstance(f, InputAudioRawFrame) for f in down) == 20
    assert ended == [SESSION["server"][-1]]
    handshake = server.handshakes[0]
    assert handshake.query["model"] == "sprag-turns"
    assert handshake.headers["Authorization"] == "Bearer sk-test"
    assert server.of_type("session.update")[0]["session"] == {
        "eot_threshold": 0.6,
        "min_silence_ms": 200,
        "eot_timeout_ms": 1500,
    }


async def test_the_detector_streams_16_khz_audio_whatever_the_input_rate(turns) -> None:
    server, url = await turns(replay_after=10**9)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await run_until(
        detector, _audio(50, sample_rate=24000), lambda down, up: server.audio_bytes >= 28000
    )
    # 50 frames of 20 ms are 32,000 bytes at 16 kHz; the resampler holds back a short tail
    assert 28000 <= server.audio_bytes <= 32000


async def test_the_detector_sends_the_agent_text_as_context(turns) -> None:
    server, url = await turns(replay_after=10**9)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await detector.set_agent_text("What can I help you with today?")
    await run_until(detector, _audio(2), lambda down, up: server.audio_bytes > 0)
    assert server.of_type("context.update") == [
        {"type": "context.update", "agent_text": "What can I help you with today?"}
    ]


async def test_a_refused_configuration_is_reported_as_an_error(turns) -> None:
    _, url = await turns(refuse="eot_threshold must be at most 1")
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    _, up = await run_until(
        detector, _audio(1), lambda down, up: any(isinstance(f, ErrorFrame) for f in up)
    )
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert "eot_threshold must be at most 1" in errors[0].error


async def test_the_analyzer_reports_the_turn_complete_once_the_model_ends_it(turns) -> None:
    server, url = await turns()
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    states = []
    try:
        for _ in range(200):
            states.append(analyzer.append_audio(FRAME_16K, is_speech=False))
            if EndOfTurnState.COMPLETE in states:
                break
            await asyncio.sleep(0.01)
        assert states.count(EndOfTurnState.COMPLETE) == 1
        assert analyzer.append_audio(FRAME_16K, is_speech=False) is EndOfTurnState.INCOMPLETE
        state, metrics = await analyzer.analyze_end_of_turn()
        assert state is EndOfTurnState.INCOMPLETE
        assert metrics is not None and metrics.probability == SESSION["server"][-1]["p_eot"]
        assert not analyzer.speech_triggered
    finally:
        await analyzer.cleanup()
    assert server.of_type("session.update")


async def test_the_observer_sets_the_spoken_reply_as_context() -> None:
    received: list[str] = []

    class _Target:
        async def set_agent_text(self, text: str) -> None:
            received.append(text)

    observer = SpragAgentTextObserver(_Target())  # type: ignore[arg-type]

    async def push(frame: Frame, first_push: bool = True) -> None:
        await observer.on_push_frame(
            FramePushed(
                source=None,  # type: ignore[arg-type]
                destination=None,  # type: ignore[arg-type]
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
                first_push=first_push,
            )
        )

    await push(BotStartedSpeakingFrame())
    await push(TTSTextFrame(text="What can I", aggregated_by=AggregationType.WORD))
    await push(
        TTSTextFrame(text="What can I", aggregated_by=AggregationType.WORD), first_push=False
    )
    await push(TTSTextFrame(text="help you with today?", aggregated_by=AggregationType.WORD))
    await push(BotStoppedSpeakingFrame())
    assert received == ["What can I help you with today?"]
