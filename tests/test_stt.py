import asyncio
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
    ProposedUserStartedSpeakingFrame,
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

    With ``hold`` set, a commit is only acknowledged with ``committed``; ``answer`` then
    sends the transcript of the commit at that index, and ``fail`` a failure in its place.
    ``commit`` commits an item no client asked for, as the server does on its own. With
    ``acknowledge`` off, a client's commit is left unacknowledged until ``commit`` is called.
    ``error`` sends an error event, which makes the client reconnect.
    """

    def __init__(self, transcript: str = "hello there") -> None:
        self.transcript = transcript
        self.hold = False
        self.acknowledge = True
        self.handshakes: list[web.Request] = []
        self.received: list[dict[str, Any]] = []
        self.items = 0
        self._outbox: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def handle(self, request: web.Request) -> web.WebSocketResponse:
        self.handshakes.append(request)
        # aiohttp fails to read a compressed second connection from the same client.
        ws = web.WebSocketResponse(compress=False)
        await ws.prepare(request)
        await ws.send_json({"type": "session.created", "session": {"type": "transcription"}})
        sender = asyncio.create_task(self._send(ws))
        try:
            async for msg in ws:
                if msg.type is not WSMsgType.TEXT:
                    continue
                event = json.loads(msg.data)
                self.received.append(event)
                if event["type"] == "session.update":
                    updated = {"type": "session.updated", "session": event["session"]}
                    self._outbox.put_nowait(updated)
                elif event["type"] == "input_audio_buffer.commit" and self.acknowledge:
                    index = self.commit()
                    if not self.hold:
                        self.answer(index)
        finally:
            sender.cancel()
        return ws

    async def _send(self, ws: web.WebSocketResponse) -> None:
        while True:
            await ws.send_json(await self._outbox.get())

    def commit(self) -> int:
        index = self.items
        self.items += 1
        committed = {"type": "input_audio_buffer.committed", "item_id": f"item_{index}"}
        self._outbox.put_nowait(committed)
        return index

    def answer(self, index: int, transcript: str | None = None) -> bool:
        item = {"item_id": f"item_{index}", "content_index": 0}
        transcript = self.transcript if transcript is None else transcript
        for word in transcript.split():
            delta = {"type": "conversation.item.input_audio_transcription.delta"}
            self._outbox.put_nowait({**delta, **item, "delta": word + " "})
        done = {"type": "conversation.item.input_audio_transcription.completed"}
        self._outbox.put_nowait({**done, **item, "transcript": transcript})
        return True

    def fail(self, index: int) -> bool:
        failed = {"type": "conversation.item.input_audio_transcription.failed"}
        item = {"item_id": f"item_{index}", "content_index": 0}
        self._outbox.put_nowait({**failed, **item, "error": {"message": "no transcript"}})
        return True

    def error(self) -> bool:
        self._outbox.put_nowait({"type": "error", "error": {"message": "try again"}})
        return True

    def committed(self, count: int) -> bool:
        return len(self.events("input_audio_buffer.commit")) >= count

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


AUDIO = [InputAudioRawFrame(audio=SILENCE, sample_rate=24000, num_channels=1)] * 10


async def _run(stt: SpragSTTService, frames: list[Any], transcripts: int) -> list[bool | None]:
    down, _ = await run_until(
        stt,
        frames,
        lambda down, up: len(_finalized(down)) == transcripts,
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.01,
    )
    return _finalized(down)


@pytest.mark.parametrize("answer_order", [[0, 1], [1, 0]])
async def test_a_transcript_is_not_finalized_while_a_later_commit_is_unanswered(
    transcription: tuple[_TranscriptionServer, str], answer_order: list[int]
) -> None:
    server, url = transcription
    server.hold = True
    first, second = answer_order
    frames = [
        *_utterance(),
        *_utterance(),
        lambda down, up: server.committed(2),
        lambda down, up: server.answer(first),
        lambda down, up: len(_finalized(down)) == 1,
        lambda down, up: server.answer(second),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, frames, 2) == [False, True]


async def test_a_transcript_that_arrives_after_the_caller_resumed_is_not_finalized(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.hold = True
    frames = [
        *_utterance(),
        lambda down, up: server.committed(1),
        VADUserStartedSpeakingFrame(),
        lambda down, up: server.answer(0),
        lambda down, up: len(_finalized(down)) == 1,
        *AUDIO,
        VADUserStoppedSpeakingFrame(),
        lambda down, up: server.committed(2),
        lambda down, up: server.answer(1),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, frames, 2) == [False, True]


async def test_a_transcript_of_audio_the_server_committed_itself_is_not_finalized(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.hold = True
    frames = [
        VADUserStartedSpeakingFrame(),
        *AUDIO,
        lambda down, up: server.commit() == 0,
        *AUDIO,
        VADUserStoppedSpeakingFrame(),
        lambda down, up: server.committed(1),
        lambda down, up: server.answer(0),
        lambda down, up: len(_finalized(down)) == 1,
        lambda down, up: server.answer(1),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, frames, 2) == [False, True]


async def test_a_transcript_is_not_finalized_while_a_commit_is_unacknowledged(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.hold = True
    server.acknowledge = False
    frames = [
        VADUserStartedSpeakingFrame(),
        *AUDIO,
        lambda down, up: server.commit() == 0,
        *AUDIO,
        VADUserStoppedSpeakingFrame(),
        lambda down, up: server.committed(1),
        lambda down, up: server.answer(0),
        lambda down, up: len(_finalized(down)) == 1,
        lambda down, up: server.answer(server.commit()),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, frames, 2) == [False, True]


async def test_a_transcript_is_not_finalized_when_the_caller_spoke_through_the_commit(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    frames = [VADUserStartedSpeakingFrame(), *AUDIO, ProposedUserStoppedSpeakingFrame(), *AUDIO]
    stt = SpragSTTService(api_key="test-key", base_url=url, commit_on_turn_end=True)
    assert await _run(stt, frames, 1) == [False]


async def test_a_transcript_is_not_finalized_after_a_turn_model_reports_the_caller_resumed(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.hold = True
    frames = [
        *AUDIO,
        ProposedUserStoppedSpeakingFrame(),
        lambda down, up: server.committed(1),
        ProposedUserStartedSpeakingFrame(),
        lambda down, up: server.answer(0),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url, commit_on_turn_end=True)
    assert await _run(stt, frames, 1) == [False]


async def test_a_turn_ended_by_a_vad_and_a_turn_model_is_committed_once(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    frames = [*_utterance(), *AUDIO[:2], ProposedUserStoppedSpeakingFrame()]
    stt = SpragSTTService(api_key="test-key", base_url=url, commit_on_turn_end=True)
    assert await _run(stt, frames, 1) == [True]
    assert len(server.events("input_audio_buffer.commit")) == 1


async def test_a_reconnect_forgets_the_commits_the_lost_connection_left_unanswered(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.hold = True
    frames = [
        *_utterance(),
        lambda down, up: server.committed(1),
        lambda down, up: server.error(),
        lambda down, up: len(server.events("session.update")) == 2,
        *_utterance(),
        lambda down, up: server.committed(2),
        lambda down, up: server.answer(1),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, frames, 1) == [True]


async def test_a_commit_made_while_disconnected_does_not_hold_back_later_transcripts(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    stt = SpragSTTService(api_key="test-key", base_url=url)
    await stt._commit_audio_buffer()
    assert await _run(stt, _utterance(), 1) == [True]


async def test_a_failed_transcription_does_not_hold_back_the_next_finalized_transcript(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.hold = True
    frames = [
        *_utterance(),
        lambda down, up: server.committed(1),
        lambda down, up: server.fail(0),
        *_utterance(),
        lambda down, up: server.committed(2),
        lambda down, up: server.answer(1),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, frames, 1) == [True]


async def test_a_transcript_without_words_is_not_finalized(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    server.transcript = " "
    stt = SpragSTTService(api_key="test-key", base_url=url)
    assert await _run(stt, _utterance(), 1) == [False]


async def test_a_transcript_is_not_finalized_when_sprag_detects_the_turns(
    transcription: tuple[_TranscriptionServer, str],
) -> None:
    server, url = transcription
    frames = [
        *AUDIO,
        lambda down, up: bool(server.events("input_audio_buffer.append")),
        lambda down, up: server.answer(server.commit()),
    ]
    stt = SpragSTTService(api_key="test-key", base_url=url, turn_detection={"type": "server_vad"})
    assert await _run(stt, frames, 1) == [False]


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
