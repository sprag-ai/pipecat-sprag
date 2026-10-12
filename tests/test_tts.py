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
    BotStoppedSpeakingFrame,
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
from websockets.protocol import State

from pipecat_sprag.common import ATTRIBUTION_HEADERS
from pipecat_sprag.tts import SpragTTSService

PCM = b"\x01\x00" * 2400  # 0.1 s of 24 kHz audio


async def run_until_playing(*args: Any, **kwargs: Any) -> tuple[list[Frame], list[Frame]]:
    return await run_until(*args, play_audio=True, **kwargs)


class _SpeechServer:
    """Serves Sprag's streaming speech socket: responses queue and are spoken in order.

    ``script`` maps a response's index, counted across connections in the order they
    were asked for, to a behaviour: ``ok`` (the default), ``slow`` (a pause mid-response),
    ``hang`` (one audio delta, then nothing), ``drop:<code>`` (one audio delta, then the
    server closes the socket), ``refuse:<code>`` (refused with an error naming the
    request, its text left buffered), or ``failed:<code>``. ``configure`` maps a
    connection's index to how it answers ``session.update``: ``ok`` (the default),
    ``late`` (after 0.3 s), ``close``, or ``ignore``. With ``refuse_reconnects``, every
    handshake after the first is answered with a 503.
    """

    def __init__(
        self,
        *,
        reject_voice: bool = False,
        refuse_reconnects: bool = False,
        script: dict[int, str] | None = None,
        configure: dict[int, str] | None = None,
    ):
        self.reject_voice = reject_voice
        self.refuse_reconnects = refuse_reconnects
        self.script = script or {}
        self.configure = configure or {}
        self.open: list[web.WebSocketResponse] = []
        self.handshakes: list[web.Request] = []
        self.voices: list[str] = []
        self.asked = 0
        self.spoken: list[tuple[int, str]] = []
        self.metadata: list[dict[str, str]] = []
        self.most_waiting = 0
        self.unconfigured_sends = 0
        self.closed = 0

    async def handle(self, request: web.Request) -> web.StreamResponse:
        if self.refuse_reconnects and self.handshakes:
            return web.Response(status=503, text="unavailable")
        connection = len(self.handshakes)
        self.handshakes.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self.open.append(ws)
        await ws.send_json({"type": "session.created", "session": {}})
        buffered: list[str] = []
        waiting: asyncio.Queue[tuple[str, str, dict[str, str]]] = asyncio.Queue()
        configured = asyncio.Event()

        async def answer_update(session: dict[str, Any]) -> None:
            how = self.configure.get(connection, "ok")
            if how == "close":
                await ws.close()
                return
            if how == "ignore":
                return
            if how == "late":
                await asyncio.sleep(0.3)
            await ws.send_json({"type": "session.updated", "session": session})
            configured.set()

        async def speak_in_order() -> None:
            while True:
                behaviour, text, metadata = await waiting.get()
                self.spoken.append((connection, text))
                self.metadata.append(metadata)
                await self._respond(ws, behaviour, text, metadata)

        speaker = asyncio.create_task(speak_in_order())
        updating: asyncio.Task[None] | None = None
        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            if event["type"] != "session.update" and not configured.is_set():
                self.unconfigured_sends += 1
            if event["type"] == "session.update":
                voice = event["session"]["audio"]["output"]["voice"]
                self.voices.append(voice)
                if self.reject_voice:
                    error = {"code": "unknown_voice", "message": f"unknown voice '{voice}'"}
                    await ws.send_json({"type": "error", "error": error})
                else:
                    updating = asyncio.create_task(answer_update(event["session"]))
            elif event["type"] == "conversation.item.create":
                item = event["item"]
                parts = item.get("content") or []
                if item.get("type") != "message" or not all(
                    part.get("type") in ("input_text", "text") for part in parts
                ):
                    error = {"code": "unsupported_item", "message": "a speech session accepts text"}
                    await ws.send_json({"type": "error", "error": error})
                    continue
                buffered.append("".join(part["text"] for part in parts))
                await ws.send_json({"type": "conversation.item.added", "item": {"id": "item_1"}})
                await ws.send_json({"type": "conversation.item.done", "item": {"id": "item_1"}})
            elif event["type"] == "response.create":
                behaviour = self.script.get(self.asked, "ok")
                self.asked += 1
                kind, _, code = behaviour.partition(":")
                if kind == "refuse":
                    error = {"code": code, "message": code, "event_id": event.get("event_id")}
                    await ws.send_json({"type": "error", "error": error})
                    continue
                metadata = (event.get("response") or {}).get("metadata") or {}
                waiting.put_nowait((behaviour, "".join(buffered), metadata))
                buffered.clear()
                self.most_waiting = max(self.most_waiting, waiting.qsize())
        for task in (speaker, updating):
            if task is not None:
                task.cancel()
        self.open.remove(ws)
        self.closed += 1
        return ws

    async def close_idle(self, code: int) -> None:
        for ws in list(self.open):
            await ws.close(code=code)

    async def _respond(
        self, ws: web.WebSocketResponse, behaviour: str, text: str, metadata: dict[str, str]
    ) -> None:
        kind, _, arg = behaviour.partition(":")
        response = {"id": "resp_1", "metadata": metadata}
        await ws.send_json({"type": "response.created", "response": response})
        await ws.send_json({"type": "response.output_item.added", "item": {"id": "item_2"}})
        await ws.send_json({"type": "response.content_part.added", "part": {"type": "audio"}})
        await ws.send_json({"type": "response.output_audio_transcript.delta", "delta": text})
        if kind == "failed":
            details = {"type": "failed", "error": {"type": "server_error", "code": arg}}
            done = {**response, "status": "failed", "status_details": details}
            await ws.send_json({"type": "response.done", "response": done})
            return
        size = len(PCM) // 2
        for i in range(2):
            delta = base64.b64encode(PCM[i * size : (i + 1) * size]).decode()
            await ws.send_json({"type": "response.output_audio.delta", "delta": delta})
            if kind == "hang":
                await asyncio.Event().wait()
            if kind == "drop":
                await ws.close(code=int(arg))
                await asyncio.Event().wait()
            await asyncio.sleep(0.3 if kind == "slow" and i == 0 else 0.01)
        for done in (
            "response.output_audio.done",
            "response.output_audio_transcript.done",
            "response.content_part.done",
            "response.output_item.done",
        ):
            await ws.send_json({"type": done})
        completed = {**response, "status": "completed"}
        await ws.send_json({"type": "response.done", "response": completed})

    @property
    def texts(self) -> list[str]:
        return [text for _, text in self.spoken]


@pytest.fixture
async def speech() -> AsyncIterator[Any]:
    servers: list[TestServer] = []

    async def start(**kwargs: Any) -> tuple[_SpeechServer, str]:
        server = _SpeechServer(**kwargs)
        app = web.Application()
        app.router.add_get("/v1/audio/speech", server.handle)
        test_server = TestServer(app)
        await test_server.start_server()
        servers.append(test_server)
        return server, str(test_server.make_url("/v1/audio/speech")).replace("http://", "ws://")

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


async def test_sends_each_sentence_without_waiting_for_the_last(speech: Any) -> None:
    server, url = await speech(script={0: "slow"})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        _turn("It is lovely to meet you today. ", "How has your ", "week been going so far?"),
        _stopped,
    )

    assert _errors(up) == []
    assert server.most_waiting >= 1
    assert server.texts == [
        "It is lovely to meet you today.",
        "How has your week been going so far?",
    ]
    assert _audio(down) == pytest.approx(0.2, abs=0.01)
    contexts = {f.context_id for f in down if isinstance(f, TTSAudioRawFrame)}
    assert [m["context_id"] for m in server.metadata] == [*contexts] * 2
    kinds = [
        type(f) for f in down if isinstance(f, TTSStartedFrame | TTSAudioRawFrame | TTSStoppedFrame)
    ]
    assert kinds[0] is TTSStartedFrame and kinds[-1] is TTSStoppedFrame
    assert kinds.count(TTSStartedFrame) == 1 and kinds.count(TTSStoppedFrame) == 1

    request = server.handshakes[0]
    assert request.query["model"] == "chorus"
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
            *_turn("A sentence the caller talks over. ", "A sentence never reached."),
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
    assert server.spoken == [(0, "A sentence the caller talks over."), (1, "The next turn.")]


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


async def test_a_failed_sentence_does_not_stop_the_turn(speech: Any) -> None:
    server, url = await speech(script={0: "failed:generation_timeout"})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts, _turn("This first sentence fails. ", "The second sentence is fine."), _stopped
    )
    assert any("generation_timeout" in e for e in _errors(up))
    assert server.texts == ["This first sentence fails.", "The second sentence is fine."]
    assert _audio(down) == pytest.approx(0.1, abs=0.01)


@pytest.mark.parametrize("code", ["response_queue_full", "rate_limit_exceeded"])
async def test_a_refused_sentence_is_reported_and_the_turn_still_ends(
    speech: Any, code: str
) -> None:
    server, url = await speech(script={1: f"refuse:{code}"})
    tts = SpragTTSService(api_key="test-key", url=url)
    started = asyncio.get_running_loop().time()
    down, up = await run_until_playing(
        tts, _turn("The first sentence. ", "The last sentence is refused."), _stopped
    )
    # well inside Pipecat's stop-frame timeout, which would otherwise end the turn
    assert asyncio.get_running_loop().time() - started < 2.5
    assert any(code in e for e in _errors(up))
    assert server.texts == ["The first sentence."]
    assert _audio(down) == pytest.approx(0.1, abs=0.01)


async def test_a_reconnect_waits_for_the_session_to_be_configured(speech: Any) -> None:
    server, url = await speech(script={0: "drop:1011"}, configure={1: "late"})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [
            TTSSpeakFrame("The server drops this."),
            # the receive loop's reconnect is now waiting on session.updated
            lambda down, up: len(server.handshakes) == 2,
            TTSSpeakFrame("This one waits for the new session."),
        ],
        _stopped_times(2),
    )
    assert server.unconfigured_sends == 0
    assert server.texts == ["The server drops this.", "This one waits for the new session."]


@pytest.mark.parametrize("how", ["close", "ignore"])
async def test_a_session_that_is_never_configured_is_reported(speech: Any, how: str) -> None:
    server, url = await speech(configure={0: how})
    tts = SpragTTSService(api_key="test-key", url=url, connect_timeout=0.3)
    down, up = await run_until_playing(
        tts,
        [lambda down, up: bool(_errors(up)), TTSSpeakFrame("Spoken on the next session.")],
        _stopped,
    )
    assert any("Error connecting to Sprag TTS" in e for e in _errors(up))
    assert server.spoken == [(1, "Spoken on the next session.")]


async def test_a_failed_send_does_not_block_the_next_sentence(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)

    def break_next_send(down: list[Frame], up: list[Frame]) -> bool:
        websocket = tts._websocket
        if websocket is None:
            return False
        send = websocket.send

        async def fail_once(message: Any, *args: Any) -> None:
            websocket.send = send
            raise RuntimeError("socket write failed")

        websocket.send = fail_once
        return True

    down, up = await run_until_playing(
        tts,
        [
            lambda down, up: tts._websocket is not None,
            break_next_send,
            TTSSpeakFrame("Lost to the failed send."),
            lambda down, up: bool(_errors(up)),
            TTSSpeakFrame("Spoken normally."),
        ],
        lambda down, up: any("Spoken normally." in t for t in server.texts) and _stopped(down, up),
    )
    assert any("send error" in e for e in _errors(up))
    assert server.texts == ["Spoken normally."]


async def test_a_finished_receive_loop_is_restarted(speech: Any) -> None:
    server, url = await speech()
    # without reconnects, the receive loop ends when the server closes the socket
    tts = SpragTTSService(api_key="test-key", url=url, reconnect_on_error=False)
    down, up = await run_until_playing(
        tts,
        [
            TTSSpeakFrame("Before the server closes."),
            _stopped_times(1),
            lambda down, up: bool(server.open),
            lambda down, up: bool(asyncio.ensure_future(server.close_idle(code=1000))),
            lambda down, up: server.closed == 1,
            TTSSpeakFrame("After the server closes."),
        ],
        _stopped_times(2),
    )
    assert [e for e in _errors(up) if "closed by server" not in e] == []
    assert server.spoken == [(0, "Before the server closes."), (1, "After the server closes.")]
    assert _audio(down) == pytest.approx(0.2, abs=0.01)


async def test_a_url_with_a_query_keeps_it(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=f"{url}?region=test")
    await run_until_playing(tts, [TTSSpeakFrame("Hello there.")], _stopped)
    assert server.handshakes[0].query["region"] == "test"
    assert server.handshakes[0].query["model"] == "chorus"


async def test_only_a_real_settings_change_replaces_the_session(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [
            TTSSpeakFrame("First."),
            _stopped_times(1),
            TTSUpdateSettingsFrame(delta=SpragTTSService.Settings(voice="wade")),
            TTSSpeakFrame("Same session."),
            _stopped_times(2),
            TTSUpdateSettingsFrame(delta=SpragTTSService.Settings(model="chorus-voices")),
            TTSSpeakFrame("New model."),
        ],
        _stopped_times(3),
    )
    assert _errors(up) == []
    assert [connection for connection, _ in server.spoken] == [0, 0, 1]
    assert server.handshakes[1].query["model"] == "chorus-voices"


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
    app.router.add_get("/v1/audio/speech", handle)
    test_server = TestServer(app)
    await test_server.start_server()
    return test_server, str(test_server.make_url("/v1/audio/speech")).replace("http://", "ws://")


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


async def test_a_dropped_connection_loses_the_rest_of_its_turn(speech: Any) -> None:
    server, url = await speech(script={0: "drop:1011"})
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [
            *_turn("The server drops this first sentence. ", "The second sentence is lost."),
            _stopped_times(1),
            lambda down, up: len(server.handshakes) == 2,
            TTSSpeakFrame("The next turn is spoken."),
        ],
        _stopped_times(2),
    )
    assert server.spoken == [
        (0, "The server drops this first sentence."),
        (1, "The next turn is spoken."),
    ]
    # half of the first sentence, then all of the next turn
    assert _audio(down) == pytest.approx(0.15, abs=0.01)


async def test_a_turn_ends_promptly_when_its_last_sentence_is_dropped(speech: Any) -> None:
    server, url = await speech(script={0: "drop:1011"})
    tts = SpragTTSService(api_key="test-key", url=url)
    started = asyncio.get_running_loop().time()
    down, _ = await run_until_playing(tts, [TTSSpeakFrame("Only this sentence.")], _stopped)
    elapsed = asyncio.get_running_loop().time() - started
    # well inside Pipecat's stop-frame timeout, which would otherwise end the turn
    assert elapsed < 2.5
    assert _audio(down) == pytest.approx(0.05, abs=0.01)


async def test_an_idle_connection_closed_by_the_server_is_replaced(speech: Any) -> None:
    server, url = await speech()
    tts = SpragTTSService(api_key="test-key", url=url)

    closed: list[bool] = []

    def session_cap(down: list[Frame], up: list[Frame]) -> bool:
        if not closed:
            closed.append(True)
            asyncio.get_running_loop().create_task(server.close_idle(code=1000))
        return len(server.handshakes) == 2

    down, up = await run_until_playing(
        tts,
        [
            TTSSpeakFrame("Before the session cap."),
            _stopped_times(1),
            session_cap,
            TTSSpeakFrame("After the session cap."),
        ],
        _stopped_times(2),
    )
    assert _errors(up) == []
    assert server.spoken == [(0, "Before the session cap."), (1, "After the session cap.")]


async def test_refused_reconnects_do_not_hang_the_pipeline(speech: Any) -> None:
    server, url = await speech(script={0: "drop:1011"}, refuse_reconnects=True)
    tts = SpragTTSService(api_key="test-key", url=url)
    down, up = await run_until_playing(
        tts,
        [TTSSpeakFrame("The server drops this."), _stopped_times(1), TTSSpeakFrame("Unspoken.")],
        _stopped_times(2),
    )
    assert _errors(up)
    assert server.texts == ["The server drops this."]


async def test_an_interruption_during_a_reconnect_closes_the_unconfigured_socket(
    speech: Any,
) -> None:
    server, url = await speech(script={0: "drop:1011"}, configure={1: "late"})
    tts = SpragTTSService(api_key="test-key", url=url)
    await run_until_playing(
        tts,
        [
            TTSSpeakFrame("The server drops this."),
            # the receive loop's reconnect is waiting on session.updated
            lambda down, up: len(server.handshakes) == 2,
            # the dropped turn's playback has ended, so its stop cannot clear the next turn's start
            lambda down, up: any(isinstance(f, BotStoppedSpeakingFrame) for f in up),
            TTSSpeakFrame("This waits for the reconnect."),
            lambda down, up: tts._tts_started,
            InterruptionFrame(),
            TTSSpeakFrame("After the interruption."),
        ],
        lambda down, up: "After the interruption." in server.texts and _stopped(down, up),
    )
    await asyncio.sleep(0.5)
    # the pipeline has shut down, so a socket still open was abandoned mid-configuration
    assert len(server.handshakes) == 3 and server.open == []
    assert "This waits for the reconnect." not in server.texts


class _DroppingSocket:
    """An open socket that closes during a send; ``noticed`` handles the drop before it fails."""

    def __init__(self, tts: SpragTTSService, noticed: bool) -> None:
        self.tts = tts
        self.noticed = noticed
        self.state = State.OPEN

    async def send(self, message: str) -> None:
        self.state = State.CLOSED
        if self.noticed:
            await self.tts._strand()
        raise ConnectionError("socket closed mid-send")


@pytest.mark.parametrize("noticed", [True, False])
async def test_a_send_lost_to_a_drop_is_reported_and_released_once(noticed: bool) -> None:
    tts = SpragTTSService(api_key="test-key", url="ws://unused")
    tts._websocket = _DroppingSocket(tts, noticed)  # type: ignore[assignment]
    frames = [frame async for frame in tts.run_tts("Lost to the drop.", "turn")]
    assert any(isinstance(frame, ErrorFrame) for frame in frames)
    assert tts._held == [] and tts._unstarted == {}
    assert "turn" not in tts._unfinished


async def test_a_socket_closed_without_a_reconnect_strands_nothing(speech: Any) -> None:
    server, url = await speech(script={0: "hang"})
    tts = SpragTTSService(api_key="test-key", url=url, reconnect_on_error=False)
    await run_until_playing(
        tts,
        [
            *_turn("This one hangs. ", "This one waits behind it."),
            lambda down, up: len(server.texts) == 1 and bool(server.open),
            lambda down, up: bool(asyncio.ensure_future(server.close_idle(code=1000))),
            _stopped_times(1),
            TTSSpeakFrame("A later turn."),
        ],
        _stopped_times(2),
    )
    assert server.texts[-1] == "A later turn."
    assert tts._unstarted == {} and tts._held == []


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRAG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPRAG_API_KEY"):
        SpragTTSService()
