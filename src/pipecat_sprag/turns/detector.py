"""Sprag's turn model as a Pipecat processor that proposes the user's turn boundaries."""

from __future__ import annotations

import asyncio
import time
from typing import Any

import numpy as np
from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    MetricsFrame,
    ProposedUserStartedSpeakingFrame,
    ProposedUserStoppedSpeakingFrame,
    StartFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import TurnMetricsData
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_sprag.common import SPRAG_EOT_MODEL, SPRAG_TURNS_URL, resolve_api_key
from pipecat_sprag.turns.events import (
    EVENT_HANDLERS,
    TurnEnded,
    TurnsError,
    TurnsServerEvent,
    TurnStarted,
    TurnsUsage,
)
from pipecat_sprag.turns.session import SpragTurnParams, TurnsSession


class SpragTurnDetector(FrameProcessor):
    """Streams caller audio to Sprag's turn model and proposes the user's turn boundaries.

    Place it after the transport input and use Pipecat's ``ExternalUserTurnStrategies``: the
    model's ``turn.started`` becomes a ``ProposedUserStartedSpeakingFrame`` and its
    ``turn.ended`` a ``ProposedUserStoppedSpeakingFrame``, which the user aggregator closes on
    once the STT's transcript arrives. Every frame passes through unchanged, and the
    connection runs in the background so audio is never held back while it opens. Its
    ``on_turn_started``, ``on_turn_update``, ``on_eager_end``, ``on_turn_resumed``,
    ``on_turn_ended`` and ``on_session_closed`` events receive the detector and the server's
    event. With metrics enabled, it pushes a ``TurnMetricsData`` for each closed turn.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = SPRAG_TURNS_URL,
        model: str = SPRAG_EOT_MODEL,
        params: SpragTurnParams | None = None,
        connect_timeout: float = 10.0,
        **kwargs: Any,
    ) -> None:
        """Initialize the detector.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment variable.
            url: The Sprag turns WebSocket URL.
            model: The turn model.
            params: The model's thresholds. Defaults to ``SpragTurnParams()``.
            connect_timeout: Seconds the session may take to open, and again to be configured.
            **kwargs: Additional arguments passed to FrameProcessor.
        """
        super().__init__(**kwargs)
        self._session = TurnsSession(
            api_key=resolve_api_key(api_key),
            url=url,
            model=model,
            params=params or SpragTurnParams(),
            on_event=self._on_event,
            connect_timeout=connect_timeout,
        )
        self._connection: asyncio.Task[None] | None = None
        self._model = model
        self._speech_stopped_at: float | None = None
        for event_name in EVENT_HANDLERS.values():
            self._register_event_handler(event_name)

    @property
    def usage(self) -> TurnsUsage | None:
        """The usage of the last session the detector closed, if the server reported it."""
        return self._session.usage

    async def set_agent_text(self, text: str) -> None:
        """Set what the agent last said, which the model reads as context for later readings.

        Args:
            text: The agent's last utterance; empty clears it.
        """
        await self._session.send_agent_text(text)

    async def end_turn(self) -> None:
        """End the caller's open turn now; the model answers ``turn.ended``, reason ``client``."""
        await self._session.send_turn_end()

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Stream input audio to the model and pass every frame on.

        Args:
            frame: The frame to process.
            direction: The frame's direction.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            if self._connection is None:
                self._connection = self.create_task(self._session.maintain(self._report_error))
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._stop(wait=isinstance(frame, EndFrame))
        elif isinstance(frame, InputAudioRawFrame):
            self._session.push_audio(_mono(frame), frame.sample_rate)
        elif isinstance(frame, VADUserStartedSpeakingFrame):
            self._speech_stopped_at = None
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._speech_stopped_at = time.perf_counter()
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Close the session."""
        await self._stop(wait=False)
        await super().cleanup()

    async def _stop(self, *, wait: bool) -> None:
        if self._connection is not None:
            await self.cancel_task(self._connection)
            self._connection = None
        await self._session.close(wait=wait)

    def can_generate_metrics(self) -> bool:
        """Report that the detector produces turn metrics."""
        return True

    async def _push_turn_metrics(self, event: TurnEnded) -> None:
        stopped, self._speech_stopped_at = self._speech_stopped_at, None
        if not self.metrics_enabled:
            return
        metrics = TurnMetricsData(
            processor=self.name,
            model=self._model,
            is_complete=True,
            probability=event.probability or 0.0,
            e2e_processing_time_ms=(time.perf_counter() - stopped) * 1000
            if stopped is not None
            else 0.0,
        )
        await self.push_frame(MetricsFrame(data=[metrics]))

    async def _report_error(self, error: Exception) -> None:
        await self.push_error(
            error_msg=f"Error connecting to Sprag turns: {error}", exception=error
        )

    async def _on_event(self, event: TurnsServerEvent) -> None:
        if isinstance(event, TurnStarted):
            await self.broadcast_frame(ProposedUserStartedSpeakingFrame)
        elif isinstance(event, TurnEnded):
            await self.broadcast_frame(ProposedUserStoppedSpeakingFrame)
            await self._push_turn_metrics(event)
        elif isinstance(event, TurnsError):
            await self.push_error(error_msg=f"Sprag turns error: {event.description}")
        if event.type in EVENT_HANDLERS:
            await self._call_event_handler(EVENT_HANDLERS[event.type], event)


def _mono(frame: InputAudioRawFrame) -> bytes:
    if frame.num_channels == 1:
        return frame.audio
    samples = np.frombuffer(frame.audio, dtype=np.int16).reshape(-1, frame.num_channels)
    return bytes(samples.mean(axis=1).astype(np.int16).tobytes())
