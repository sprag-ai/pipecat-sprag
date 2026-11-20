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
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.pipeline.worker import PipelineParams

from pipecat_sprag.stt import SpragSTTService

SILENCE = b"\x00\x00" * 480  # 20 ms at 24 kHz


class _TranscriptionServer:
    """Answers each commit with ``committed`` and a transcript.

    ``answer_after`` holds the transcripts back until that many commits have arrived, then
    sends them in the ``answer_order`` given, as indexes into the commits. ``fail`` holds
    the indexes of the commits answered with a failure in place of a transcript.
    """

    def __init__(self, transcript: str = "hello there") -> None:
        self.transcript = transcript
        self.answer_after = 1
        self.answer_order: list[int] | None = None
        self.fail: set[int] = set()
        self.handshakes: list[web.Request] = []
        self.received: list[dict[str, Any]] = []

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.handshakes.append(request)
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {"type": "transcription"}})
        commits = 0
        async for msg in ws:
            if msg.type is not WSMsgType.TEXT:
                continue
            event = json.loads(msg.data)
            self.received.append(event)
            if event["type"] == "session.update":
                await ws.send_json({"type": "session.updated", "session": event["session"]})
            elif event["type"] == "input_audio_buffer.commit":
                await ws.send_json(
                    {"type": "input_audio_buffer.committed", "item_id": f"item_{commits}"}
                )
                commits += 1
                if commits == self.answer_after:
                    for index in self.answer_order or range(commits):
                        await self._answer(ws, index)
                elif commits > self.answer_after:
                    await self._answer(ws, commits - 1)
        return ws

    async def _answer(self, ws: web.WebSocketResponse, index: int) -> None:
        item = {"item_id": f"item_{index}", "content_index": 0}
        if index in self.fail:
            failed = {"type": "conversation.item.input_audio_transcription.failed"}
            await ws.send_json({**failed, **item, "error": {"message": "no transcript"}})
            return
        for word in self.transcript.split(" "):
            delta = {"type": "conversation.item.input_audio_transcription.delta"}
            await ws.send_json({**delta, **item, "delta": word + " "})
        done = {"type": "conversation.item.input_audio_transcription.completed"}
        await ws.send_json({**done, **item, "transcript": self.transcript})

    def events(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.received if e["type"] == event_type]


@pytest.fixture
async def transcription() -> AsyncIterator[tuple[_TranscriptionServer, str]]:
    server = _TranscriptionServer()
    app = web.Application()
    app.router.add_get("/v1/audio/transcriptions", server.handle)
    test_server = TestServer(app)
    await test_server.start_server()
    yield server, str(test_server.make_url("/v1/audio/transcriptions")).replace("http://", "ws://")
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


@pytest.mark.parametrize("turn_end", ["vad", "proposed"])
async def test_the_transcript_of_a_committed_turn_is_finalized(
    transcription: tuple[_TranscriptionServer, str], turn_end: str
) -> None:
    server, url = transcription
    stt = SpragSTTService(api_key="test-key", base_url=url, commit_on_turn_end=True)
    audio = [InputAudioRawFrame(audio=SILENCE, sample_rate=24000, num_channels=1)] * 10
    frames = _utterance() if turn_end == "vad" else [*audio, ProposedUserStoppedSpeakingFrame()]
    down, _ = await run_until(
        stt, frames, _transcribed, params=PipelineParams(audio_in_sample_rate=24000), between=0.01
    )
    assert _finalized(down) == [True]


def _finalized(frames: list[Frame]) -> list[bool | None]:
    return [f.finalized for f in frames if isinstance(f, TranscriptionFrame)]


@pytest.mark.parametrize("answer_order", [[0, 1], [1, 0]])
async def test_a_transcript_is_not_finalized_while_a_later_commit_is_unanswered(
    transcription: tuple[_TranscriptionServer, str], answer_order: list[int]
) -> None:
    server, url = transcription
    server.answer_after = 2
    server.answer_order = answer_order
    stt = SpragSTTService(api_key="test-key", base_url=url)
    down, _ = await run_until(
        stt,
        [*_utterance(), *_utterance()],
        lambda down, up: len(_finalized(down)) == 2,
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.01,
    )
    assert _finalized(down) == [False, True]


async def test_a_failed_transcription_does_not_hold_back_the_next_finalized_transcript(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.fail = {0}
    stt = SpragSTTService(api_key="test-key", base_url=url)
    down, _ = await run_until(
        stt,
        [*_utterance(), *_utterance()],
        lambda down, up: bool(_finalized(down)),
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.01,
    )
    assert _finalized(down) == [True]


@pytest.mark.parametrize("commit_on_turn_end", [True, False])
async def test_a_proposed_turn_end_commits_the_audio_when_enabled(
    transcription: tuple[_TranscriptionServer, str], commit_on_turn_end: bool
) -> None:
    server, url = transcription
    stt = SpragSTTService(api_key="test-key", base_url=url, commit_on_turn_end=commit_on_turn_end)
    audio = [InputAudioRawFrame(audio=SILENCE, sample_rate=24000, num_channels=1)] * 10
    down, _ = await run_until(
        stt,
        [*audio, ProposedUserStoppedSpeakingFrame(), *audio],
        lambda down, up: _transcribed(down, up) or not commit_on_turn_end,
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.01,
    )
    finals = [f.text for f in down if isinstance(f, TranscriptionFrame)]
    assert finals == (["hello there"] if commit_on_turn_end else [])
    assert len(server.events("input_audio_buffer.commit")) == (1 if commit_on_turn_end else 0)
