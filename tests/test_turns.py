import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from helpers import run_until
from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import EndOfTurnState
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.observers.base_observer import FramePushed
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.utils.errors import ErrorCategory
from pipecat.utils.text.base_text_aggregator import AggregationType
from pydantic import ValidationError
from websockets.exceptions import ConnectionClosed

from pipecat_sprag.turns import (
    SessionConfigRefusedError,
    SessionRejectedError,
    SpragAgentTextObserver,
    SpragTurnAnalyzer,
    SpragTurnDetector,
    SpragTurnParams,
    TurnEnded,
    TurnsUsage,
)
from pipecat_sprag.turns import session as turns_session
from pipecat_sprag.turns.events import EVENT_HANDLERS
from pipecat_sprag.turns.session import TurnsSession

# One /v1/turns session recorded from the sprag-eot model: every server event it sent.
SESSION = json.loads((Path(__file__).parent / "data" / "turns_session.json").read_text())
FRAME_16K = b"\x01\x00" * 320  # 20 ms at 16 kHz


class _TurnsServer:
    """Replays the recorded session's turn events once ``replay_after`` bytes of audio arrive.

    It answers ``session.close`` like the real server: the ``on_close`` events still in
    flight, then ``session.closed`` with ``invocations``, unless ``answer_close`` is off.
    ``configure_gate`` holds back ``session.updated`` until it is set, and ``hang_up`` closes
    the socket on ``session.update``.
    """

    def __init__(
        self,
        replay_after: int = 6400,
        refuse: str | None = None,
        stall: bool = False,
        drop_after: int | None = None,
        before_replay: list[str] | None = None,
        replay: list[dict[str, Any]] | None = None,
        configure_delay: float = 0.0,
        configure_gate: asyncio.Event | None = None,
        hang_up: bool = False,
        answer_close: bool = True,
        on_close: list[dict[str, Any]] | None = None,
        invocations: int = 1,
        reject: tuple[int, str] | None = None,
    ) -> None:
        self.replay_after = replay_after
        self.refuse = refuse
        self.stall = stall
        self.drop_after = drop_after
        self.before_replay = before_replay or []
        self.replay = replay
        self.configure_delay = configure_delay
        self.configure_gate = configure_gate
        self.hang_up = hang_up
        self.answer_close = answer_close
        self.on_close = on_close or []
        self.invocations = invocations
        self.reject = reject
        self.handshakes: list[web.Request] = []
        self.events: list[dict[str, Any]] = []
        # audio bytes the connection had received when each of ``events`` arrived
        self.audio_before: list[int] = []
        self.audio_bytes = 0
        self.received: list[bytearray] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        self.handshakes.append(request)
        if self.reject is not None:
            status, message = self.reject
            return web.json_response({"error": {"message": message}}, status=status)
        audio = bytearray()
        self.received.append(audio)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        server_events = list(SESSION["server"])
        await ws.send_json(server_events.pop(0))
        updated = server_events.pop(0)
        if self.replay is not None:
            server_events = self.replay
        dropped = len(self.handshakes) > 1 or self.drop_after is None
        received = 0
        replayed = False
        async for msg in ws:
            if msg.type is WSMsgType.BINARY:
                self.audio_bytes += len(msg.data)
                received += len(msg.data)
                audio.extend(msg.data)
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
            self.audio_before.append(received)
            if event["type"] == "session.update" and self.hang_up:
                await ws.close()
                break
            if event["type"] == "session.update" and not self.stall:
                await asyncio.sleep(self.configure_delay)
                if self.configure_gate is not None:
                    await self.configure_gate.wait()
                if self.refuse:
                    await ws.send_json(
                        {
                            "type": "error",
                            "error": {"code": "invalid_value", "message": self.refuse},
                        }
                    )
                else:
                    await ws.send_json(updated)
            elif event["type"] == "session.close" and self.answer_close:
                for in_flight in self.on_close:
                    await ws.send_json(in_flight)
                await ws.send_json(
                    {"type": "session.closed", "usage": {"invocations": self.invocations}}
                )
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
    ended: list[TurnEnded] = []

    @detector.event_handler("on_turn_ended")
    async def _on_ended(processor: SpragTurnDetector, event: TurnEnded) -> None:
        ended.append(event)

    down, up = await run_until(
        detector,
        _audio(40),
        lambda down, up: _proposed_stop(down, up) and _count(down, InputAudioRawFrame) == 40,
        between=0.01,
    )
    assert _count(down, ProposedUserStartedSpeakingFrame) == 1
    assert _count(down, ProposedUserStoppedSpeakingFrame) == 1
    assert _count(up, ProposedUserStoppedSpeakingFrame) == 1
    assert _count(down, InputAudioRawFrame) == 40
    assert [e.model_dump() for e in ended] == [SESSION["server"][-1]]
    handshake = server.handshakes[0]
    assert handshake.query["model"] == "sprag-eot"
    assert handshake.headers["Authorization"] == "Bearer sk-test"
    assert server.of_type("session.update")[0]["session"] == {
        "eot_threshold": 0.6,
        "min_silence_ms": 200,
        "eot_timeout_ms": 1500,
    }
    assert server.of_type("session.close")


def _turn_metrics(frames: list[Frame]) -> list[TurnMetricsData]:
    return [
        d
        for f in frames
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, TurnMetricsData)
    ]


@pytest.mark.parametrize("enable_metrics", [True, False])
async def test_the_detector_reports_turn_metrics_when_enabled(turns, enable_metrics) -> None:
    _, url = await turns()
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    down, _ = await run_until(
        detector,
        [
            lambda down, up: detector._session.connected,
            VADUserStartedSpeakingFrame(),
            *_audio(5),
            VADUserStoppedSpeakingFrame(),
            *_audio(20),
        ],
        lambda down, up: _proposed_stop(down, up) and (not enable_metrics or _turn_metrics(down)),
        params=PipelineParams(enable_metrics=enable_metrics),
        between=0.01,
    )
    metrics = _turn_metrics(down)
    if not enable_metrics:
        assert metrics == []
        return
    assert len(metrics) == 1
    assert metrics[0].processor == detector.name
    assert metrics[0].model == "sprag-eot"
    assert metrics[0].is_complete
    assert metrics[0].probability == SESSION["server"][-1]["probability"]
    assert metrics[0].e2e_processing_time_ms > 0


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


async def test_the_detector_asks_the_server_to_end_the_turn(turns) -> None:
    server, url = await turns(replay_after=10**9)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    sent: list[asyncio.Task[None]] = []

    def end_turn_once_connected(down: list[Frame], up: list[Frame]) -> bool:
        if not detector._session.connected:
            return False
        if not sent:
            sent.append(asyncio.get_running_loop().create_task(detector.end_turn()))
        return True

    await run_until(
        detector,
        [end_turn_once_connected, *_audio(2)],
        lambda down, up: bool(server.of_type("turn.end")),
        between=0.01,
    )
    assert server.of_type("turn.end") == [{"type": "turn.end"}]


async def test_ending_the_turn_while_disconnected_sends_nothing() -> None:
    detector = SpragTurnDetector(api_key="sk-test", url="ws://127.0.0.1:9/v1/turns")
    await detector.end_turn()


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
    server, url = await turns(hang_up=True)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    _, up = await run_until(
        detector, _audio(1), lambda down, up: _count(up, ErrorFrame) >= 4, timeout=5
    )
    # retries wait 0.05, 0.1 and 0.2 s, so four attempts happen
    assert len(server.handshakes) >= 4
    assert detector.is_usable


def _after(seconds: float, condition: Callable[[list[Frame], list[Frame]], bool]) -> Any:
    """Return a ``run_until`` condition that holds ``seconds`` after ``condition`` first does."""
    since: list[float] = []

    def check(down: list[Frame], up: list[Frame]) -> bool:
        if not since and condition(down, up):
            since.append(time.perf_counter())
        return bool(since) and time.perf_counter() - since[0] >= seconds

    return check


async def test_a_refused_configuration_is_reported_once_and_not_retried(turns, fast_retry) -> None:
    server, url = await turns(refuse="eot_threshold must be at most 1")
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    down, up = await run_until(
        detector,
        [*_audio(1), _after(0.5, lambda down, up: _count(up, ErrorFrame) > 0), *_audio(3)],
        lambda down, up: _count(down, InputAudioRawFrame) == 4,
    )
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert "eot_threshold must be at most 1" in errors[0].error
    assert isinstance(errors[0].exception, SessionConfigRefusedError)
    assert errors[0].category is ErrorCategory.INVALID_REQUEST
    assert len(server.handshakes) == 1
    assert not detector.is_usable
    assert _count(down, InputAudioRawFrame) == 4


@pytest.mark.parametrize(
    ("status", "message", "category"),
    [
        (401, "Invalid API key", ErrorCategory.AUTHENTICATION),
        (404, "The model `nope` does not exist.", ErrorCategory.INVALID_REQUEST),
    ],
)
async def test_a_rejected_connection_is_reported_once_and_not_retried(
    turns, fast_retry, status: int, message: str, category: ErrorCategory
) -> None:
    server, url = await turns(reject=(status, message))
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    down, up = await run_until(
        detector,
        [*_audio(1), _after(0.5, lambda down, up: _count(up, ErrorFrame) > 0), *_audio(3)],
        lambda down, up: _count(down, InputAudioRawFrame) == 4,
    )
    errors = [f for f in up if isinstance(f, ErrorFrame)]
    assert len(errors) == 1
    assert message in errors[0].error and "not reconnecting" in errors[0].error
    assert isinstance(errors[0].exception, SessionRejectedError)
    assert errors[0].exception.status_code == status
    assert errors[0].category is category
    assert len(server.handshakes) == 1
    assert not detector.is_usable


async def test_a_rejection_that_may_pass_is_retried(turns, fast_retry) -> None:
    server, url = await turns(reject=(503, "Temporarily unavailable"))
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    await run_until(detector, _audio(1), lambda down, up: _count(up, ErrorFrame) >= 3, timeout=5)
    assert len(server.handshakes) >= 3
    assert detector.is_usable


async def test_the_analyzer_stops_connecting_after_a_rejected_connection(turns, fast_retry) -> None:
    server, url = await turns(reject=(401, "Invalid API key"))
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        analyzer.append_audio(FRAME_16K, is_speech=True)
        connection = analyzer._connection
        assert connection is not None
        await asyncio.wait_for(connection, timeout=5)
        await asyncio.sleep(0.3)
        assert len(server.handshakes) == 1
        rejections = [m for m in logged if "Invalid API key" in m]
        assert len(rejections) == 1 and "not reconnecting" in rejections[0]
    finally:
        logger.remove(sink)
        await analyzer.cleanup()


async def test_the_analyzer_stops_connecting_after_a_refused_configuration(
    turns, fast_retry
) -> None:
    server, url = await turns(refuse="eot_threshold must be at most 1")
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    logged: list[str] = []
    sink = logger.add(lambda message: logged.append(str(message)), level="ERROR")
    try:
        analyzer.append_audio(FRAME_16K, is_speech=True)
        connection = analyzer._connection
        assert connection is not None
        await asyncio.wait_for(connection, timeout=5)
        await asyncio.sleep(0.3)
        assert analyzer.append_audio(FRAME_16K, is_speech=True) is EndOfTurnState.INCOMPLETE
        assert analyzer._connection is connection
        assert len(server.handshakes) == 1
        refusals = [m for m in logged if "eot_threshold must be at most 1" in m]
        assert len(refusals) == 1 and "not reconnecting" in refusals[0]
    finally:
        logger.remove(sink)
        await analyzer.cleanup()


@pytest.mark.parametrize(
    "params",
    [
        {"min_silence_ms": 10001, "eot_timeout_ms": 20000},
        {"min_silence_ms": 3000},
        {"min_silence_ms": 500, "eot_timeout_ms": 500},
        {"eot_timeout_ms": 99, "min_silence_ms": 0},
        {"eot_timeout_ms": 60001},
        {"eot_threshold": 1.1},
        {"eager_eot_threshold": -0.1},
        {"eot_threshold": 0.5, "eager_eot_threshold": 0.6},
    ],
)
def test_params_the_server_would_refuse_fail_at_construction(params: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SpragTurnParams(**params)


def test_params_at_the_server_s_bounds_are_accepted() -> None:
    SpragTurnParams(min_silence_ms=10000, eot_timeout_ms=60000)
    SpragTurnParams(min_silence_ms=0, eot_timeout_ms=100)
    SpragTurnParams(eot_threshold=0.5, eager_eot_threshold=0.5)


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
        assert metrics is not None and metrics.probability == SESSION["server"][-1]["probability"]
        # the fake server replays the turn after ten 20 ms frames of silence
        assert metrics.e2e_processing_time_ms > 50
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


async def test_a_turn_end_is_reported_after_the_next_turn_started(turns) -> None:
    next_turn = {
        "type": "turn.started",
        "turn_index": 1,
        "audio_window_start": 5.4,
        "audio_window_end": 5.4,
    }
    _, url = await turns(replay=[EAGER_SESSION[0], EAGER_SESSION[4], next_turn])
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    seen = _record_events(analyzer)
    try:
        for _ in range(10):
            assert analyzer.append_audio(FRAME_16K, is_speech=False) is EndOfTurnState.INCOMPLETE
        await _wait_for(lambda: len(seen) == 3)
        states = [analyzer.append_audio(FRAME_16K, is_speech=True) for _ in range(5)]
        assert states == [EndOfTurnState.COMPLETE] + [EndOfTurnState.INCOMPLETE] * 4
        assert analyzer.speech_triggered
        state, metrics = await analyzer.analyze_end_of_turn()
        assert state is EndOfTurnState.INCOMPLETE
        assert metrics is not None and metrics.probability == EAGER_SESSION[4]["probability"]
    finally:
        await analyzer.cleanup()


def _error_log() -> tuple[list[str], int]:
    logged: list[str] = []
    return logged, logger.add(lambda message: logged.append(str(message)), level="ERROR")


MISSING_FIELD = json.dumps({"type": "turn.ended", "turn_index": 0})


async def test_the_detector_drops_an_event_missing_a_field(turns) -> None:
    server, url = await turns(before_replay=[MISSING_FIELD])
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    ended: list[TurnEnded] = []

    @detector.event_handler("on_turn_ended")
    async def _on_ended(processor: SpragTurnDetector, event: TurnEnded) -> None:
        ended.append(event)

    logged, sink = _error_log()
    try:
        down, _ = await run_until(detector, _audio(40), _proposed_stop, between=0.01)
    finally:
        logger.remove(sink)
    assert _count(down, ProposedUserStoppedSpeakingFrame) == 1
    assert [e.model_dump() for e in ended] == [SESSION["server"][-1]]
    assert len(server.handshakes) == 1
    assert sum("Error handling a Sprag turns event" in m for m in logged) == 1


async def test_the_analyzer_drops_an_event_missing_a_field(turns) -> None:
    server, url = await turns(before_replay=[MISSING_FIELD])
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    logged, sink = _error_log()
    try:
        states = await _feed_until_complete(analyzer)
        assert states.count(EndOfTurnState.COMPLETE) == 1
        assert len(server.handshakes) == 1
        assert sum("Error handling a Sprag turns event" in m for m in logged) == 1
    finally:
        logger.remove(sink)
        await analyzer.cleanup()


@pytest.mark.parametrize("answer_close", [True, False])
async def test_a_cancelled_detector_closes_at_once_and_reports_no_usage(
    turns, monkeypatch: pytest.MonkeyPatch, answer_close: bool
) -> None:
    server, url = await turns(replay_after=10**9, answer_close=answer_close)
    detector = SpragTurnDetector(api_key="sk-test", url=url)
    close = detector._session.close
    closes: list[float] = []

    async def timed_close(*, wait: bool = True) -> None:
        started = time.perf_counter()
        await close(wait=wait)
        closes.append(time.perf_counter() - started)

    monkeypatch.setattr(detector._session, "close", timed_close)
    await run_until(
        detector,
        [lambda down, up: detector._session.connected, *_audio(2), CancelFrame()],
        lambda down, up: bool(closes),
    )
    assert server.of_type("session.close")
    assert closes[0] < 0.5
    assert detector.usage is None


def _numbered(count: int, start: int = 0) -> list[bytes]:
    """Return 20 ms 16 kHz frames whose samples hold each frame's number, to check order."""
    return [(start + i).to_bytes(2, "little") * 320 for i in range(count)]


def _session(url: str, events: list[Any] | None = None) -> TurnsSession:
    async def on_event(event: Any) -> None:
        if events is not None:
            events.append(event)

    return TurnsSession(
        api_key="sk-test", url=url, model="sprag-eot", params=SpragTurnParams(), on_event=on_event
    )


async def test_audio_pushed_before_connecting_is_sent_in_order_once_configured(turns) -> None:
    server, url = await turns(replay_after=10**9)
    session = _session(url)
    frames = _numbered(100)
    try:
        for frame in frames:
            session.push_audio(frame, 16000)
        await session.connect()
        await _wait_for(lambda: len(server.received[0]) >= 64000)
        assert bytes(server.received[0]) == b"".join(frames)
    finally:
        await session.close()


async def test_audio_buffered_before_connecting_keeps_the_latest_scoring_window(turns) -> None:
    server, url = await turns(replay_after=10**9)
    session = _session(url)
    # ten seconds, of which the eight-second scoring window is kept
    frames = _numbered(500)
    try:
        for frame in frames:
            session.push_audio(frame, 16000)
        await session.connect()
        await _wait_for(lambda: len(server.received[0]) >= 256000)
        await asyncio.sleep(0.1)
        assert bytes(server.received[0]) == b"".join(frames[100:])
    finally:
        await session.close()


async def test_the_analyzer_sends_audio_that_arrives_while_it_connects(turns) -> None:
    configured = asyncio.Event()
    server, url = await turns(replay_after=10**9, configure_gate=configured)
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    frames = _numbered(20)
    try:
        analyzer.append_audio(frames[0], is_speech=True)
        await _wait_for(lambda: bool(server.of_type("session.update")))
        for frame in frames[1:]:
            analyzer.append_audio(frame, is_speech=True)
            await asyncio.sleep(0.01)
        assert server.audio_bytes == 0
        configured.set()
        await _wait_for(lambda: len(server.received[0]) >= 12800)
        assert bytes(server.received[0]) == b"".join(frames)
    finally:
        await analyzer.cleanup()


async def test_a_reconnect_replays_only_the_audio_since_the_connection_was_lost(turns) -> None:
    server, url = await turns(replay_after=10**9, drop_after=6400)
    session = _session(url)
    before = _numbered(10)
    after = _numbered(450, start=10)
    try:
        await session.connect()
        for frame in before:
            session.push_audio(frame, 16000)
        await _wait_for(lambda: not session.connected)
        for frame in after:
            session.push_audio(frame, 16000)
        await session.connect()
        await _wait_for(lambda: len(server.received[1]) >= 256000)
        await asyncio.sleep(0.1)
        assert bytes(server.received[0]) == b"".join(before)
        assert bytes(server.received[1]) == b"".join(after[50:])
    finally:
        await session.close()


async def test_a_chunk_longer_than_the_scoring_window_keeps_its_latest_window(turns) -> None:
    server, url = await turns(replay_after=10**9)
    session = _session(url)
    chunk = b"".join(_numbered(500))
    try:
        session.push_audio(_numbered(1, start=999)[0], 16000)
        session.push_audio(chunk, 16000)
        await session.connect()
        await _wait_for(lambda: len(server.received[0]) >= 256000)
        await asyncio.sleep(0.1)
        assert bytes(server.received[0]) == chunk[-256000:]
    finally:
        await session.close()


async def test_buffered_48_khz_audio_is_capped_at_the_scoring_window(turns) -> None:
    server, url = await turns(replay_after=10**9)
    session = _session(url)
    try:
        for frame in _audio(500, sample_rate=48000):
            assert isinstance(frame, InputAudioRawFrame)
            session.push_audio(frame.audio, 48000)
        await session.connect()
        await _wait_for(lambda: len(server.received[0]) >= 250000)
        await asyncio.sleep(0.1)
        # eight seconds at 16 kHz are 256,000 bytes; the resampler holds back a short tail
        assert 250000 <= len(server.received[0]) <= 256000
    finally:
        await session.close()


async def test_a_chunk_whose_flush_fails_is_sent_on_the_next_connection(
    turns, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, url = await turns(replay_after=10**9)
    session = _session(url)
    frames = _numbered(10)
    for frame in frames:
        session.push_audio(frame, 16000)
    send = session._send
    failed: list[bool] = []

    async def fail_the_third(websocket: Any, audio: bytes, sample_rate: int) -> None:
        if audio == frames[2] and not failed:
            failed.append(True)
            await websocket.close()
        await send(websocket, audio, sample_rate)

    monkeypatch.setattr(session, "_send", fail_the_third)
    try:
        with pytest.raises(ConnectionClosed):
            await session.connect()
        await session.connect()
        await _wait_for(lambda: len(server.received[1]) >= 5120)
        await asyncio.sleep(0.1)
        assert bytes(server.received[0]) == b"".join(frames[:2])
        assert bytes(server.received[1]) == b"".join(frames[2:])
    finally:
        await session.close()


async def test_the_chunk_in_flight_when_the_connection_drops_is_sent_on_the_next(
    turns, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, url = await turns(replay_after=10**9, drop_after=6400)
    session = _session(url)
    frames = _numbered(20)
    send = session._send
    held: list[bool] = []

    async def hold_the_eleventh(websocket: Any, audio: bytes, sample_rate: int) -> None:
        if audio == frames[10] and not held:
            held.append(True)
            await asyncio.Event().wait()
        await send(websocket, audio, sample_rate)

    monkeypatch.setattr(session, "_send", hold_the_eleventh)
    try:
        await session.connect()
        for frame in frames:
            session.push_audio(frame, 16000)
        await _wait_for(lambda: not session.connected)
        assert held
        await session.connect()
        await _wait_for(lambda: len(server.received[1]) >= 6400)
        await asyncio.sleep(0.1)
        assert bytes(server.received[0]) == b"".join(frames[:10])
        assert bytes(server.received[1]) == b"".join(frames[10:])
    finally:
        await session.close()


async def test_agent_text_set_while_connecting_reaches_the_server_before_live_audio(
    turns, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, url = await turns(replay_after=10**9)
    session = _session(url)
    await session.send_agent_text("Hi")
    frames = _numbered(6)
    for frame in frames[:5]:
        session.push_audio(frame, 16000)
    send = session._send

    async def set_text_during_the_flush(websocket: Any, audio: bytes, sample_rate: int) -> None:
        if session._agent_text == "Hi":
            await session.send_agent_text("What can I help you with?")
        await send(websocket, audio, sample_rate)

    monkeypatch.setattr(session, "_send", set_text_during_the_flush)
    try:
        await session.connect()
        session.push_audio(frames[5], 16000)
        await _wait_for(lambda: len(server.received[0]) >= 3840)
        updates = [
            (event, before)
            for event, before in zip(server.events, server.audio_before, strict=True)
            if event["type"] == "context.update"
        ]
        assert updates == [
            ({"type": "context.update", "agent_text": "Hi"}, 0),
            ({"type": "context.update", "agent_text": "What can I help you with?"}, 3200),
        ]
        assert bytes(server.received[0]) == b"".join(frames)
    finally:
        await session.close()


async def test_usage_reports_only_the_latest_connection(
    turns, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(turns_session, "SESSION_CLOSE_TIMEOUT_SECS", 0.2)
    server, url = await turns(replay_after=10**9, invocations=4)
    session = _session(url)
    await session.connect()
    await session.close()
    assert session.usage == TurnsUsage(invocations=4)
    server.answer_close = False
    await session.connect()
    assert session.usage is None
    await session.close()
    assert session.usage is None
    assert len(server.of_type("session.close")) == 2


# A v2 session with an eager end the caller talks through, then a model-ended turn.
EAGER_SESSION = [
    {"type": "turn.started", "turn_index": 0, "audio_window_start": 1.3, "audio_window_end": 1.3},
    {
        "type": "turn.update",
        "turn_index": 0,
        "probability": 0.21,
        "silence": 0.2,
        "audio_window_start": 1.3,
        "audio_window_end": 2.9,
    },
    {
        "type": "turn.eager_end",
        "turn_index": 0,
        "probability": 0.47,
        "silence": 0.3,
        "audio_window_start": 1.3,
        "audio_window_end": 3.0,
    },
    {"type": "turn.resumed", "turn_index": 0, "audio_window_start": 1.3, "audio_window_end": 3.2},
    {
        "type": "turn.ended",
        "turn_index": 0,
        "reason": "model",
        "probability": 0.83,
        "silence": 0.3,
        "audio_window_start": 1.3,
        "audio_window_end": 5.1,
    },
]
SESSION_CLOSED = {"type": "session.closed", "usage": {"invocations": 2}}
HANDLED = [
    "on_turn_started",
    "on_turn_update",
    "on_eager_end",
    "on_turn_resumed",
    "on_turn_ended",
    "on_session_closed",
]


def _record_events(target: SpragTurnAnalyzer | SpragTurnDetector) -> list[tuple[str, Any]]:
    seen: list[tuple[str, Any]] = []
    for name in EVENT_HANDLERS.values():

        async def handler(_: object, event: Any, name: str = name) -> None:
            seen.append((name, event))

        target.add_event_handler(name, handler)
    return seen


def _assert_every_event_handled(seen: list[tuple[str, Any]]) -> None:
    assert [name for name, _ in seen] == HANDLED
    events = [event for _, event in seen]
    assert [type(e).__name__ for e in events] == [
        "TurnStarted",
        "TurnUpdate",
        "TurnEagerEnd",
        "TurnResumed",
        "TurnEnded",
        "SessionClosed",
    ]
    assert [e.model_dump() for e in events] == [*EAGER_SESSION, SESSION_CLOSED]
    assert events[2].probability == 0.47 and events[4].reason == "model"
    assert events[5].usage == TurnsUsage(invocations=2)


async def test_every_event_reaches_the_analyzer_s_handlers(turns) -> None:
    _, url = await turns(replay=EAGER_SESSION, invocations=2)
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    seen = _record_events(analyzer)
    try:
        await _feed_until_complete(analyzer)
    finally:
        await analyzer.cleanup()
    _assert_every_event_handled(seen)
    assert analyzer.usage == TurnsUsage(invocations=2)


async def test_every_event_reaches_the_detector_s_handlers(turns) -> None:
    server, url = await turns(replay=EAGER_SESSION, invocations=2)
    detector = SpragTurnDetector(
        api_key="sk-test", url=url, params=SpragTurnParams(eager_eot_threshold=0.45)
    )
    seen = _record_events(detector)
    await run_until(
        detector,
        _audio(40),
        lambda down, up: _proposed_stop(down, up) and _count(down, InputAudioRawFrame) == 40,
        between=0.01,
    )
    _assert_every_event_handled(seen)
    assert detector.usage == TurnsUsage(invocations=2)
    assert server.of_type("session.update")[0]["session"]["eager_eot_threshold"] == 0.45


async def test_cleanup_delivers_in_flight_events_and_the_session_s_usage(turns) -> None:
    in_flight = dict(EAGER_SESSION[1], audio_window_end=7.0)
    _, url = await turns(on_close=[in_flight], invocations=3)
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    seen = _record_events(analyzer)
    try:
        await _feed_until_complete(analyzer)
    finally:
        await analyzer.cleanup()
    assert [name for name, _ in seen[-2:]] == ["on_turn_update", "on_session_closed"]
    assert seen[-2][1].audio_window_end == 7.0
    assert analyzer.usage == TurnsUsage(invocations=3)


async def test_cleanup_returns_promptly_when_the_server_never_closes(
    turns, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(turns_session, "SESSION_CLOSE_TIMEOUT_SECS", 0.3)
    server, url = await turns(replay_after=10**9, answer_close=False)
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url=url)
    analyzer.set_sample_rate(16000)
    analyzer.append_audio(FRAME_16K, is_speech=True)
    await _wait_for(lambda: analyzer._session.connected)
    started = time.perf_counter()
    await analyzer.cleanup()
    elapsed = time.perf_counter() - started
    assert 0.25 <= elapsed < 1.5
    assert server.of_type("session.close")
    assert analyzer.usage is None


async def test_closing_a_session_that_never_connected_returns_at_once() -> None:
    session = _session("ws://127.0.0.1:9/v1/turns")
    session.push_audio(FRAME_16K, 16000)
    await asyncio.wait_for(session.close(), timeout=0.1)
    assert session.usage is None


async def test_the_analyzer_refuses_audio_before_it_has_a_sample_rate() -> None:
    analyzer = SpragTurnAnalyzer(api_key="sk-test", url="ws://127.0.0.1:9/v1/turns")
    with pytest.raises(RuntimeError, match="no sample rate"):
        analyzer.append_audio(FRAME_16K, is_speech=True)
    assert analyzer.append_audio(FRAME_16K, is_speech=True) is EndOfTurnState.INCOMPLETE
    assert analyzer._connection is None


def test_the_session_refuses_audio_without_a_positive_sample_rate() -> None:
    session = _session("ws://127.0.0.1:9/v1/turns")
    with pytest.raises(ValueError, match="sample rate"):
        session.push_audio(FRAME_16K, 0)


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
