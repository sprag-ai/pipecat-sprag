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
    InputAudioRawFrame,
    InterimTranscriptionFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams

from pipecat_sprag.stt import SpragSTTService

SILENCE = b"\x00\x00" * 480  # 20 ms at 24 kHz


class _TranscriptionServer:
    """Speaks the realtime transcription events Sprag's gateway sends."""

    def __init__(self, transcript: str = "hello there") -> None:
        self.transcript = transcript
        self.handshakes: list[web.Request] = []
        self.received: list[dict[str, Any]] = []

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.handshakes.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {"type": "transcription"}})
        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            self.received.append(event)
            if event["type"] == "session.update":
                await ws.send_json({"type": "session.updated", "session": event["session"]})
            elif event["type"] == "input_audio_buffer.commit":
                item = {"item_id": "item_1", "content_index": 0}
                await ws.send_json({"type": "input_audio_buffer.committed", **item})
                for word in self.transcript.split(" "):
                    delta = {"type": "conversation.item.input_audio_transcription.delta"}
                    await ws.send_json({**delta, **item, "delta": word + " "})
                done = {"type": "conversation.item.input_audio_transcription.completed"}
                await ws.send_json({**done, **item, "transcript": self.transcript})
        return ws

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.received if e["type"] == event_type]


@pytest.fixture
async def transcription() -> AsyncIterator[tuple[_TranscriptionServer, str]]:
    server = _TranscriptionServer()
    app = web.Application()
    app.router.add_get("/v1/realtime", server.handle)
    test_server = TestServer(app)
    await test_server.start_server()
    yield server, str(test_server.make_url("/v1/realtime")).replace("http://", "ws://")
    await test_server.close()


def _utterance() -> list[Frame]:
    audio = [InputAudioRawFrame(audio=SILENCE, sample_rate=24000, num_channels=1)] * 10
    return [VADUserStartedSpeakingFrame(), *audio, VADUserStoppedSpeakingFrame()]


def _transcribed(down: list[Frame], up: list[Frame]) -> bool:
    return any(isinstance(f, TranscriptionFrame) for f in down) or any(
        isinstance(f, ErrorFrame) for f in up
    )


async def test_transcribes_a_turn_ended_by_local_vad(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    stt = SpragSTTService(api_key="test-key", base_url=url)
    down, up = await run_until(
        stt,
        _utterance(),
        _transcribed,
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.01,
    )

    assert not [f for f in up if isinstance(f, ErrorFrame)]
    finals = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert finals == ["hello there"]
    assert any(isinstance(f, InterimTranscriptionFrame) for f in down)

    request = server.handshakes[0]
    assert request.query["intent"] == "transcription"
    assert request.headers["Authorization"] == "Bearer test-key"

    update = server.events("session.update")[0]["session"]
    assert update["type"] == "transcription"
    transcription_config = update["audio"]["input"]["transcription"]
    assert transcription_config["model"] == "rhythm"
    assert "language" not in transcription_config
    assert server.events("input_audio_buffer.append")
    assert server.events("input_audio_buffer.commit")


async def test_server_vad_is_requested_when_enabled(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    stt = SpragSTTService(api_key="test-key", base_url=url, turn_detection={"type": "server_vad"})
    await run_until(
        stt,
        [],
        lambda down, up: bool(server.events("session.update")),
        params=PipelineParams(audio_in_sample_rate=24000),
    )
    update = server.events("session.update")[0]["session"]
    assert update["audio"]["input"]["turn_detection"]["type"] == "server_vad"


async def test_model_can_be_overridden(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    stt = SpragSTTService(
        api_key="test-key", base_url=url, settings=SpragSTTService.Settings(model="rhythm-next")
    )
    await run_until(
        stt,
        [],
        lambda down, up: bool(server.events("session.update")),
        params=PipelineParams(audio_in_sample_rate=24000),
    )
    update = server.events("session.update")[0]["session"]
    assert update["audio"]["input"]["transcription"]["model"] == "rhythm-next"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRAG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPRAG_API_KEY"):
        SpragSTTService()
