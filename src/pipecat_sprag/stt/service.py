"""Sprag's Rhythm model as a Pipecat speech-to-text service."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from typing import Any, Literal

from pipecat.frames.frames import (
    Frame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
)
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.openai.stt import OpenAIRealtimeSTTService, OpenAIRealtimeSTTSettings

from pipecat_sprag.common import SPRAG_TRANSCRIPTION_URL, resolve_api_key

# The server refuses a commit of less audio with an error, which drops the session.
MIN_COMMIT_SECS = 0.1


@dataclass
class SpragSTTSettings(OpenAIRealtimeSTTSettings):
    """Settings for SpragSTTService."""


class SpragSTTService(OpenAIRealtimeSTTService):
    """Streaming transcription with Rhythm over Sprag's streaming transcription socket.

    By default the pipeline's own VAD decides when a turn ends. Pass
    ``turn_detection={"type": "server_vad"}`` to have Sprag detect turns instead, or
    ``commit_on_turn_end=True`` to transcribe each turn when a turn model such as
    ``SpragTurnDetector`` proposes its end.

    Unlike the LLM and TTS services, it sends no ``X-Sprag-Integration`` header: Pipecat's
    realtime STT connects with a fixed set of headers.
    """

    Settings = SpragSTTSettings
    _settings: SpragSTTSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SPRAG_TRANSCRIPTION_URL,
        turn_detection: dict[str, Any] | Literal[False] | None = False,
        commit_on_turn_end: bool = False,
        settings: SpragSTTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sprag STT service.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment
                variable.
            base_url: The Sprag streaming transcription WebSocket URL.
            turn_detection: ``False`` (the default) leaves turn detection to the
                pipeline's VAD. A dict such as ``{"type": "server_vad"}`` enables
                Sprag's server-side VAD.
            commit_on_turn_end: Whether a ``ProposedUserStoppedSpeakingFrame`` commits the
                audio for transcription. Applies only without server-side VAD.
            settings: Runtime-updatable settings. The model defaults to ``rhythm``.
            **kwargs: Additional arguments passed to OpenAIRealtimeSTTService.
        """
        # Rhythm takes no language parameter
        default_settings = self.Settings(model="rhythm", language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            turn_detection=turn_detection,
            settings=default_settings,
            **kwargs,
        )
        self._commit_on_turn_end = commit_on_turn_end
        self._unacknowledged_commits = 0
        self._open_items: set[str] = set()
        self._spoken_since_commit = False
        self._uncommitted_secs = 0.0
        self._finalizing = asyncio.Lock()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Process a frame, committing the audio when a proposed turn end arrives.

        Args:
            frame: The frame to process.
            direction: The frame's direction.
        """
        if isinstance(frame, (VADUserStartedSpeakingFrame, ProposedUserStartedSpeakingFrame)):
            async with self._finalizing:
                self._spoken_since_commit = True
                await super().process_frame(frame, direction)
            return
        await super().process_frame(frame, direction)
        if (
            self._commit_on_turn_end
            and not self._server_vad_enabled
            and isinstance(frame, ProposedUserStoppedSpeakingFrame)
        ):
            await self._commit_audio_buffer()

    async def push_frame(
        self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM
    ) -> None:
        """Push a frame, marking a transcript finalized when it covers all the caller said.

        Args:
            frame: The frame to push.
            direction: The direction to push it in.
        """
        if not isinstance(frame, TranscriptionFrame):
            await super().push_frame(frame, direction)
            return
        # Held so that a frame reporting the caller speaking again cannot pass a transcript
        # already judged to cover all their speech.
        async with self._finalizing:
            if self._covers_all_speech(frame):
                frame.finalized = True
            await super().push_frame(frame, direction)

    def _covers_all_speech(self, transcript: TranscriptionFrame) -> bool:
        return (
            not self._server_vad_enabled
            and not self._spoken_since_commit
            and not self._unacknowledged_commits
            and not self._open_items
            and bool(transcript.text.strip())
        )

    async def _websocket_connect(self, uri: str, **kwargs: Any) -> Any:
        return _CommittedItems(await super()._websocket_connect(uri, **kwargs), self._open_item)

    async def _disconnect_websocket(self) -> None:
        await super()._disconnect_websocket()
        self._unacknowledged_commits = 0
        self._open_items.clear()
        self._uncommitted_secs = 0.0

    async def _send_audio(self, audio: bytes) -> None:
        await super()._send_audio(audio)
        if self._websocket and not self._disconnecting:
            self._uncommitted_secs += len(audio) / (self.sample_rate * 2)

    async def _commit_audio_buffer(self) -> None:
        if self._uncommitted_secs < MIN_COMMIT_SECS:
            return
        self._uncommitted_secs = 0.0
        self._unacknowledged_commits += 1
        self._spoken_since_commit = self._user_speaking
        await super()._commit_audio_buffer()

    def _open_item(self, item_id: str) -> None:
        self._unacknowledged_commits = max(self._unacknowledged_commits - 1, 0)
        self._open_items.add(item_id)

    async def _handle_transcription_completed(self, evt: dict[str, Any]) -> None:
        self._open_items.discard(evt.get("item_id", ""))
        await super()._handle_transcription_completed(evt)

    async def _handle_transcription_failed(self, evt: dict[str, Any]) -> None:
        self._open_items.discard(evt.get("item_id", ""))
        await super()._handle_transcription_failed(evt)


class _CommittedItems:
    """A socket that reports the item of each ``input_audio_buffer.committed`` it receives."""

    def __init__(self, websocket: Any, on_committed: Callable[[str], None]) -> None:
        self._websocket = websocket
        self._on_committed = on_committed

    def __getattr__(self, name: str) -> Any:
        return getattr(self._websocket, name)

    async def __aiter__(self) -> AsyncIterator[Any]:
        async for message in self._websocket:
            try:
                event = json.loads(message)
            except ValueError:
                event = None
            if isinstance(event, dict) and event.get("type") == "input_audio_buffer.committed":
                self._on_committed(event.get("item_id", ""))
            yield message
