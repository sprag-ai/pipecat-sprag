import asyncio
import base64
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import WSMsgType, web
from aiohttp.test_utils import TestServer
from helpers import run_until
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSAudioRawFrame,
    TTSSpeakFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    TTSUpdateSettingsFrame,
)
from pipecat.metrics.metrics import TTFBMetricsData, TTSUsageMetricsData
from pipecat.pipeline.worker import PipelineParams

from pipecat_sprag.common import ATTRIBUTION_HEADERS
from pipecat_sprag.tts import SpragTTSService

PCM = b"\x01\x00" * 2400  # 0.1 s of 24 kHz audio


async def run_until_playing(*args: Any, **kwargs: Any) -> tuple[list[Frame], list[Frame]]:
    return await run_until(*args, play_audio=True, **kwargs)


class _SpeechServer:
    """Serves Sprag's realtime speech session, including its one-response-at-a-time rule.

    ``script`` maps a response's index, counted across connections, to a behaviour:
    ``ok`` (the default), ``slow`` (a pause mid-response), ``hang`` (one audio delta,
    then nothing), ``stall`` (no reply), ``error:<code>``, or ``failed:<code>``.
    """

    def __init__(self, *, reject_voice: bool = False, script: dict[int, str] | None = None):
        self.reject_voice = reject_voice
        self.script = script or {}
        self.handshakes: list[web.Request] = []
        self.voices: list[str] = []
        self.spoken: list[tuple[int, str]] = []
        self.overlaps = 0
        self.closed = 0

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        connection = len(self.handshakes)
        self.handshakes.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {}})
        pending: list[str] = []
        active: asyncio.Task[None] | None = None
        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event["type"] == "session.update":
                voice = event["session"]["audio"]["output"]["voice"]
                self.voices.append(voice)
                if self.reject_voice:
                    error = {"code": "unknown_voice", "message": f"unknown voice '{voice}'"}
                    await ws.send_json({"type": "error", "error": error})
                else:
                    await ws.send_json({"type": "session.updated", "session": event["session"]})
            elif event["type"] == "conversation.item.create":
                item = event["item"]
                parts = item.get("content") or []
                if item.get("type") != "message" or not all(
                    part.get("type") in ("input_text", "text") for part in parts
                ):
                    error = {"code": "unsupported_item", "message": "a speech session accepts text"}
                    await ws.send_json({"type": "error", "error": error})
                    continue
                pending.append("".join(part["text"] for part in parts))
                await ws.send_json({"type": "conversation.item.added", "item": {"id": "item_1"}})
                await ws.send_json({"type": "conversation.item.done", "item": {"id": "item_1"}})
            elif event["type"] == "response.create":
                if active is not None and not active.done():
                    self.overlaps += 1
                    error = {
                        "code": "conversation_already_has_active_response",
                        "message": "a response is already in progress",
                    }
                    await ws.send_json({"type": "error", "error": error})
                    continue
                index = len(self.spoken)
                self.spoken.append((connection, "".join(pending)))
                pending.clear()
                behaviour = self.script.get(index, "ok")
                active = asyncio.create_task(self._respond(ws, behaviour, self.spoken[-1][1]))
        if active is not None:
            active.cancel()
        self.closed += 1
        return ws

    async def _respond(self, ws: web.WebSocketResponse, behaviour: str, text: str) -> None:
        kind, _, arg = behaviour.partition(":")
        if kind == "stall":
            await asyncio.Event().wait()
        if kind == "error":
            await ws.send_json({"type": "error", "error": {"code": arg, "message": arg}})
            return
        await ws.send_json({"type": "response.created", "response": {"id": "resp_1"}})
        await ws.send_json({"type": "response.output_item.added", "item": {"id": "item_2"}})
        await ws.send_json({"type": "response.content_part.added", "part": {"type": "audio"}})
        await ws.send_json({"type": "response.output_audio_transcript.delta", "delta": text})
        if kind == "failed":
            details = {"type": "failed", "error": {"type": "server_error", "code": arg}}
            done = {
                "type": "response.done",
                "response": {"status": "failed", "status_details": details},
            }
            await ws.send_json(done)
            return
        for i, half in enumerate((PCM[:2400], PCM[2400:])):
            delta = base64.b64encode(half).decode()
            await ws.send_json({"type": "response.output_audio.delta", "delta": delta})
            if kind == "hang":
                await asyncio.Event().wait()
            if kind == "slow" and i == 0:
                await asyncio.sleep(0.3)
            else:
                await asyncio.sleep(0.01)
        for done in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            await ws.send_json({"type": done})
        await ws.send_json({"type": "response.done", "response": {"status": "completed"}})

    @property
    def texts(self) -> list[str]:
        return [text for _, text in self.spoken]


@pytest.fixture
async def speech() -> AsyncIterator[Any]:
    servers: list[TestServer] = []

    async def start(**kwargs: Any) -> tuple[_SpeechServer, str]:
        server = _SpeechServer(**kwargs)
        app = web.Application()
        app.router.add_get("/v1/realtime", server.handle)
        test_server = TestServer(app)
        await test_server.start_server()
        servers.append(test_server)
        return server, str(test_server.make_url("/v1/realtime")).replace("http://", "ws://")

    yield start
    for test_server in servers:
        await test_server.close()


def _turn(*chunks: str) -> list[Frame]:
    return [
        LLMFullResponseStartFrame(),
        *(LLMTextFrame(c) for c in chunks),
        LLMFullResponseEndFrame(),
    ]


def _stopped(down: list[Frame], up: list[Frame]) -> bool:
    return any(isinstance(f, TTSStoppedFrame) for f in down)


def _stopped_times(n: int) -> Any:
    return lambda down, up: sum(isinstance(f, TTSStoppedFrame) for f in down) >= n


def _audio(down: list[Frame]) -> float:
    frames = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    return sum(len(f.audio) for f in frames) / 2 / 24000


def _errors(up: list[Frame]) -> list[str]:
    return [f.error for f in up if isinstance(f, ErrorFrame)]


async def test_speaks_each_sentence_one_response_at_a_time(speech: Any) -> None:
    server, url = await speech(script={0: "slow"})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        _turn("It is lovely to meet you today. ", "How has your ", "week been going so far?"),
        _stopped,
    )

    assert _errors(up) == []
    assert server.overlaps == 0
    assert server.texts == [
        "It is lovely to meet you today.",
        "How has your week been going so far?",
    ]
    assert _audio(down) == pytest.approx(0.2, abs=0.01)
    kinds = [
        type(f) for f in down if isinstance(f, TTSStartedFrame | TTSAudioRawFrame | TTSStoppedFrame)
    ]
    assert kinds[0] is TTSStartedFrame and kinds[-1] is TTSStoppedFrame
    assert kinds.count(TTSStartedFrame) == 1 and kinds.count(TTSStoppedFrame) == 1

    request = server.handshakes[0]
    assert request.query["model"] == "chorus-clone"
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Sprag-Integration"] == ATTRIBUTION_HEADERS["X-Sprag-Integration"]
    assert server.voices == ["wade"]


async def test_reuses_the_connection_across_turns(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [TTSSpeakFrame("The first turn."), _stopped_times(1), TTSSpeakFrame("The second turn.")],
        _stopped_times(2),
    )
    assert _errors(up) == []
    assert server.spoken == [(0, "The first turn."), (0, "The second turn.")]


async def test_reports_ttfb_and_usage(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)
    down, _ = await run_until_playing(
        tts,
        [TTSSpeakFrame("Hello there.")],
        _stopped,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )
    metrics = [d for f in down if isinstance(f, MetricsFrame) for d in f.data]
    assert any(isinstance(d, TTFBMetricsData) and d.value > 0 for d in metrics)
    assert any(
        isinstance(d, TTSUsageMetricsData) and d.value == len("Hello there.") for d in metrics
    )


async def test_interruption_replaces_the_connection(speech: Any) -> None:
    server, url = await speech(script={0: "hang"})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [
            TTSSpeakFrame("A sentence the caller talks over."),
            lambda down, up: any(isinstance(f, TTSAudioRawFrame) for f in down),
            InterruptionFrame(),
            lambda down, up: server.closed >= 1,
            TTSSpeakFrame("The next turn."),
        ],
        lambda down, up: (
            server.texts == ["A sentence the caller talks over.", "The next turn."]
            and any(isinstance(f, TTSStoppedFrame) for f in down)
        ),
    )
    assert _errors(up) == []
    assert [connection for connection, _ in server.spoken] == [0, 1]
    assert server.overlaps == 0


async def test_voice_change_starts_a_new_session(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [
            TTSSpeakFrame("The first voice."),
            _stopped_times(1),
            TTSUpdateSettingsFrame(delta=SpragTTSService.Settings(voice="serena")),
            TTSSpeakFrame("The second voice."),
        ],
        _stopped_times(2),
    )
    assert _errors(up) == []
    assert server.voices == ["wade", "serena"]
    assert [connection for connection, _ in server.spoken] == [0, 1]


async def test_refused_voice_is_reported(speech: Any) -> None:
    server, url = await speech(reject_voice=True)
    tts = SpragTTSService(
        api_key="test-key", url=url, settings=SpragTTSService.Settings(voice="nobody")
    )
    _, up = await run_until_playing(tts, [], lambda down, up: bool(_errors(up)))
    assert any("unknown voice" in e for e in _errors(up))


@pytest.mark.parametrize("behaviour", ["error:input_too_long", "failed:generation_timeout"])
async def test_a_failed_sentence_does_not_stop_the_turn(speech: Any, behaviour: str) -> None:
    server, url = await speech(script={0: behaviour})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts, _turn("This first sentence fails. ", "The second sentence is fine."), _stopped
    )
    assert any(behaviour.partition(":")[2] in e for e in _errors(up))
    assert server.texts == ["This first sentence fails.", "The second sentence is fine."]
    assert _audio(down) == pytest.approx(0.1, abs=0.01)


async def test_a_stalled_response_recovers(speech: Any) -> None:
    server, url = await speech(script={0: "stall"})
    tts = SpragTTSService(api_key="test-key", url=url, response_timeout=0.3)
    down, up = await run_until_playing(
        tts,
        _turn("This first sentence never finishes. ", "The second sentence is fine."),
        lambda down, up: len(server.texts) == 2 and _stopped(down, up),
    )
    assert any("did not finish" in e for e in _errors(up))
    assert [connection for connection, _ in server.spoken] == [0, 1]
    assert _audio(down) == pytest.approx(0.1, abs=0.01)


async def test_model_and_voice_can_be_overridden(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(
        api_key="test-key",
        url=url,
        settings=SpragTTSService.Settings(model="chorus-voices", voice="serena"),
    )
    await run_until_playing(tts, [TTSSpeakFrame("Hello there.")], _stopped)
    assert server.handshakes[0].query["model"] == "chorus-voices"
    assert server.voices == ["serena"]


async def _serve_rejecting_first(status: int, server: _SpeechServer) -> tuple[TestServer, str]:
    attempts = 0

    async def handle(request: web.Request) -> web.StreamResponse:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return web.Response(status=status, text="refused")
        return await server.handle(request)

    app = web.Application()
    app.router.add_get("/v1/realtime", handle)
    test_server = TestServer(app)
    await test_server.start_server()
    return test_server, str(test_server.make_url("/v1/realtime")).replace("http://", "ws://")


async def test_an_unavailable_server_is_retried_on_the_next_sentence() -> None:
    server = _SpeechServer()
    test_server, url = await _serve_rejecting_first(503, server)
    tts = SpragTTSService(api_key="secret-test-key", url=url)
    try:
        down, up = await run_until_playing(
            tts,
            [lambda down, up: bool(_errors(up)), TTSSpeakFrame("Hello after the retry.")],
            _stopped,
        )
    finally:
        await test_server.close()

    assert any("503" in e for e in _errors(up))
    assert not any("secret-test-key" in e for e in _errors(up))
    assert server.texts == ["Hello after the retry."]
    assert _audio(down) == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("status", [401, 403])
async def test_a_rejected_key_stops_the_service(status: int) -> None:
    server = _SpeechServer()
    test_server, url = await _serve_rejecting_first(status, server)
    tts = SpragTTSService(api_key="secret-test-key", url=url)
    try:
        down, up = await run_until_playing(
            tts,
            [lambda down, up: bool(_errors(up)), TTSSpeakFrame("Never spoken.")],
            lambda down, up: any(isinstance(f, TTSStoppedFrame) for f in down) or not tts.is_usable,
        )
        await asyncio.sleep(0.2)
    finally:
        await test_server.close()

    assert any(str(status) in e for e in _errors(up))
    assert not any("secret-test-key" in e for e in _errors(up))
    assert not tts.is_usable
    assert server.handshakes == [] and server.texts == []


async def test_blank_input_sends_nothing(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)
    await run_until_playing(tts, [TTSSpeakFrame("   "), TTSSpeakFrame("Real words.")], _stopped)
    assert server.texts == ["Real words."]


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRAG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPRAG_API_KEY"):
        SpragTTSService()
