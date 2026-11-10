import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
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
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TTSTextFrame,
)
from pipecat.observers.base_observer import FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.text.base_text_aggregator import AggregationType

from pipecat_sprag.turns import (
    SpragAgentTextObserver,
    SpragTurnAnalyzer,
    SpragTurnDetector,
    SpragTurnParams,
)
from pipecat_sprag.turns import session as turns_session

# One /v1/turns session recorded from the sprag-turns model: every server event it sent.
SESSION = json.loads((Path(__file__).parent / "data" / "turns_session.json").read_text())
FRAME_16K = b"\x01\x00" * 320  # 20 ms at 16 kHz


class _TurnsServer:
    """Replays the recorded session's turn events once ``replay_after`` bytes of audio arrive."""

    def __init__(
        self,
        replay_after: int = 6400,
        refuse: str | None = None,
        stall: bool = False,
        drop_after: int | None = None,
        before_replay: list[str] | None = None,
    ) -> None:
        self.replay_after = replay_after
        self.refuse = refuse
        self.stall = stall
        self.drop_after = drop_after
        self.before_replay = before_replay or []
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
        dropped = len(self.handshakes) > 1 or self.drop_after is None
        received = 0
        replayed = False
        async for msg in ws:
            if msg.type is WSMsgType.BINARY:
                self.audio_bytes += len(msg.data)
                received += len(msg.data)
                if not dropped and self.drop_after is not None and received >= self.drop_after:
                    await ws.close()
                    break
                if not replayed and self.audio_bytes >= self.replay_after:
                    replayed = True
                    for raw in self.before_replay:
                        await ws.send_str(raw)
                    for event in server_events:
                        await ws.send_json(event)
                continue
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            self.events.append(event)
            if event["type"] == "session.update" and not self.stall:
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


@pytest.fixture
def fast_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(turns_session, "RECONNECT_INTERVAL_SECS", 0.05)


def _audio(count: int, sample_rate: int = 16000, channels: int = 1) -> list[Frame]:
    chunk = b"\x01\x00" * (sample_rate // 50) * channels
    return [
        InputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=channels)
        for _ in range(count)
    ]


def _count(frames: list[Frame], frame_type: type[Frame]) -> int:
    return sum(isinstance(f, frame_type) for f in frames)


def _proposed_stop(down: list[Frame], up: list[Frame]) -> bool:
    return _count(down, ProposedUserStoppedSpeakingFrame) > 0


async def _wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=timeout)


async def test_the_detector_proposes_the_turn_boundaries_the_model_reports(turns) -> None:
    server, url = await turns()
    detector = SpragTurnDetector(
        api_key="sk-test", url=url, params=SpragTurnParams(eot_timeout_ms=1500)
    )
    ended: list[dict[str, Any]] = []

    @detector.event_handler("on_turn_ended")
    async def _on_ended(processor: SpragTurnDetector, event: dict[str, Any]) -> None:
        ended.append(event)

    down, up = await run_until(detector, _audio(40), _proposed_stop, between=0.01)
    assert _count(down, ProposedUserStartedSpeakingFrame) == 1
    assert _count(down, ProposedUserStoppedSpeakingFrame) == 1
    assert _count(up, ProposedUserStoppedSpeakingFrame) == 1
    assert _count(down, InputAudioRawFrame) == 40
    assert ended == [SESSION["server"][-1]]
    handshake = server.handshakes[0]
    assert handshake.query["model"] == "sprag-turns"
    assert handshake.headers["Authorization"] == "Bearer sk-test"
    assert server.of_type("session.update")[0]["session"] == {
        "eot_threshold": 0.6,
        "min_silence_ms": 200,
        "eot_timeout_ms": 1500,
    }
    assert server.of_type("session.close")


async def test_the_detector_passes_audio_on_while_the_server_stalls(turns) -> None:
    _, url = await turns(stall=True)
    detector = SpragTurnDetector(api_key="sk-test", url=url, connect_timeout=30)
    down, _ = await run_until(
        detector,
        _audio(50),
        lambda down, up: _count(down, InputAudioRawFrame) == 50,
        between=0.02,
        timeout=5,
    )
    assert _count(down, InputAudioRawFrame) == 50


async def test_the_detector_streams_16_khz_audio_whatever_the_input_rate(turns) -> None:
    server, url = await turns(replay_after=10**9)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await run_until(
        detector,
        _audio(50, sample_rate=24000),
        lambda down, up: server.audio_bytes >= 28000,
        between=0.02,
    )
    # 50 frames of 20 ms are 32,000 bytes at 16 kHz; the resampler holds back a short tail
    assert 28000 <= server.audio_bytes <= 32000


async def test_the_detector_downmixes_stereo_audio(turns) -> None:
    server, url = await turns(replay_after=10**9)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await run_until(
        detector,
        [lambda down, up: detector._session.connected, *_audio(20, channels=2)],
        lambda down, up: server.audio_bytes >= 12800,
        between=0.02,
    )
    assert server.audio_bytes == 12800


async def test_the_detector_sends_the_agent_text_as_context(turns) -> None:
    server, url = await turns(replay_after=10**9)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await detector.set_agent_text("What can I help you with today?")
    await run_until(detector, _audio(5), lambda down, up: server.audio_bytes > 0, between=0.02)
    assert server.of_type("context.update") == [
        {"type": "context.update", "agent_text": "What can I help you with today?"}
    ]


async def test_the_detector_reconnects_and_resends_the_context(turns) -> None:
    server, url = await turns(replay_after=10**9, drop_after=3200)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await detector.set_agent_text("Hi")
    await run_until(
        detector,
        _audio(200),
        lambda down, up: len(server.of_type("context.update")) >= 2,
        between=0.01,
    )
    assert len(server.handshakes) == 2
    assert server.of_type("context.update") == [{"type": "context.update", "agent_text": "Hi"}] * 2


async def test_a_malformed_server_message_does_not_end_the_session(turns) -> None:
    _, url = await turns(before_replay=["not json", "[1, 2]"])
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    down, _ = await run_until(detector, _audio(40), _proposed_stop, between=0.01)
    assert _count(down, ProposedUserStoppedSpeakingFrame) == 1


async def test_failed_connections_are_retried_with_a_growing_delay(turns, fast_retry) -> None:
    server, url = await turns(refuse="eot_threshold must be at most 1")
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    _, up = await run_until(
        detector, _audio(1), lambda down, up: _count(up, ErrorFrame) >= 4, timeout=5
    )
    # retries wait 0.05, 0.1 and 0.2 s, so four attempts happen
    assert len(server.handshakes) >= 4


async def test_a_refused_configuration_is_reported_as_an_error(turns) -> None:
    _, url = await turns(refuse="eot_threshold must be at most 1")
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    _, up = await run_until(detector, _audio(1), lambda down, up: _count(up, ErrorFrame) > 0)
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert "eot_threshold must be at most 1" in errors[0].error


async def test_a_server_error_event_is_reported_as_an_error(turns) -> None:
    error = json.dumps({"type": "error", "error": {"code": "rate_limited"}})
    _, url = await turns(before_replay=[error])
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    _, up = await run_until(
        detector, _audio(40), lambda down, up: _count(up, ErrorFrame) > 0, between=0.01
    )
    errors = [f.error for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1 and "Sprag turns error: rate_limited" in errors[0]


async def _feed_until_complete(analyzer: SpragTurnAnalyzer) -> list[EndOfTurnState]:
    states = []
    for _ in range(300):
        states.append(analyzer.append_audio(FRAME_16K, is_speech=False))
        if EndOfTurnState.COMPLETE in states:
            break
        await asyncio.sleep(0.01)
    return states


async def test_the_analyzer_keeps_the_turn_complete_until_it_is_cleared(turns) -> None:
    server, url = await turns()
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    try:
        states = await _feed_until_complete(analyzer)
        assert states.count(EndOfTurnState.COMPLETE) == 1
        assert analyzer.append_audio(FRAME_16K, is_speech=False) is EndOfTurnState.INCOMPLETE
        state, metrics = await analyzer.analyze_end_of_turn()
        assert state is EndOfTurnState.COMPLETE
        assert metrics is not None and metrics.probability == SESSION["server"][-1]["p_eot"]
        state, metrics = await analyzer.analyze_end_of_turn()
        assert state is EndOfTurnState.COMPLETE and metrics is None
        assert not analyzer.speech_triggered
        analyzer.clear()
        state, _ = await analyzer.analyze_end_of_turn()
        assert state is EndOfTurnState.INCOMPLETE
    finally:
        await analyzer.cleanup()
    assert server.of_type("session.update")
    assert server.of_type("session.close")


async def test_the_analyzer_sends_the_agent_text_once_connected(turns) -> None:
    server, url = await turns(replay_after=10**9)
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    try:
        analyzer.append_audio(FRAME_16K, is_speech=True)
        await _wait_for(lambda: bool(server.of_type("session.update")))
        await _wait_for(lambda: analyzer._session.connected)
        await analyzer.set_agent_text("Hi")
        await _wait_for(lambda: bool(server.of_type("context.update")))
        assert server.of_type("context.update") == [{"type": "context.update", "agent_text": "Hi"}]
    finally:
        await analyzer.cleanup()


async def test_the_analyzer_opens_a_new_session_after_cleanup(turns) -> None:
    server, url = await turns()
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    try:
        await _feed_until_complete(analyzer)
        await analyzer.cleanup()
        states = await _feed_until_complete(analyzer)
        assert EndOfTurnState.COMPLETE in states
        assert len(server.handshakes) == 2
    finally:
        await analyzer.cleanup()


class _Target:
    def __init__(self) -> None:
        self.received: list[str] = []

    async def set_agent_text(self, text: str) -> None:
        self.received.append(text)


async def test_the_observer_sets_the_played_reply_as_context() -> None:
    target = _Target()
    observer = SpragAgentTextObserver(target)  # type: ignore[arg-type]
    transport = BaseOutputTransport(TransportParams())

    async def push(frame: Frame, source: object = transport) -> None:
        await observer.on_push_frame(
            FramePushed(
                source=source,  # type: ignore[arg-type]
                destination=None,  # type: ignore[arg-type]
                frame=frame,
                direction=FrameDirection.DOWNSTREAM,
                timestamp=0,
            )
        )

    await push(BotStartedSpeakingFrame())
    await push(TTSTextFrame(text="What can I", aggregated_by=AggregationType.WORD))
    # the same words on their way from TTS to the transport, not yet played
    await push(TTSTextFrame(text="unplayed words", aggregated_by=AggregationType.WORD), None)
    await push(TTSTextFrame(text="help you with today?", aggregated_by=AggregationType.WORD))
    await push(BotStoppedSpeakingFrame())
    await push(BotStartedSpeakingFrame())
    await push(BotStoppedSpeakingFrame())
    assert target.received == ["What can I help you with today?", ""]


def test_downmixing_averages_the_channels() -> None:
    from pipecat_sprag.turns.detector import _mono

    stereo = np.array([[100, 300], [-50, -150]], dtype=np.int16).tobytes()
    frame = InputAudioRawFrame(audio=stereo, sample_rate=16000, num_channels=2)
    assert np.frombuffer(_mono(frame), dtype=np.int16).tolist() == [200, -100]
