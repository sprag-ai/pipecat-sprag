"""Sprag's turn model as a Pipecat turn analyzer."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnAnalyzer, EndOfTurnState
from pipecat.metrics.metrics import MetricsData, TurnMetricsData

from pipecat_sprag.common import SPRAG_TURNS_MODEL, SPRAG_TURNS_URL, resolve_api_key
from pipecat_sprag.turns.session import SpragTurnParams, TurnsSession, error_message


class SpragTurnAnalyzer(BaseTurnAnalyzer):
    """Streams the caller's audio to Sprag's turn model; the turn is complete on its ``turn.ended``.

    Pass it as ``TurnAnalyzerUserTurnStopStrategy(turn_analyzer=...)``, one analyzer per
    pipeline. It is a streaming analyzer: ``append_audio`` returns ``COMPLETE`` once the model
    has ended the turn, and ``analyze_end_of_turn`` keeps reporting ``COMPLETE`` until the turn
    is cleared or the model opens a new one. The analyzer sees no agent text of its own; call
    ``set_agent_text`` with what the agent last said, for example from
    ``SpragAgentTextObserver``.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = SPRAG_TURNS_URL,
        model: str = SPRAG_TURNS_MODEL,
        params: SpragTurnParams | None = None,
        connect_timeout: float = 10.0,
        sample_rate: int | None = None,
    ) -> None:
        """Initialize the analyzer.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment variable.
            url: The Sprag turns WebSocket URL.
            model: The turn model.
            params: The model's thresholds. Defaults to ``SpragTurnParams()``.
            connect_timeout: Seconds the session may take to open, and again to be configured.
            sample_rate: The input sample rate; defaults to the pipeline's.
        """
        super().__init__(sample_rate=sample_rate)
        self._params = params or SpragTurnParams()
        self._session = TurnsSession(
            api_key=resolve_api_key(api_key),
            url=url,
            model=model,
            params=self._params,
            on_event=self._on_event,
            connect_timeout=connect_timeout,
        )
        self._connection: asyncio.Task[None] | None = None
        self._speech_triggered = False
        self._turn_ended = False
        self._turn_reported = False
        self._last_metrics: TurnMetricsData | None = None
        self._speech_stopped_at: float | None = None

    @property
    def speech_triggered(self) -> bool:
        """Whether the model has opened a turn that has not ended yet."""
        return self._speech_triggered

    @property
    def params(self) -> SpragTurnParams:
        """The model's thresholds."""
        return self._params

    async def set_agent_text(self, text: str) -> None:
        """Set what the agent last said, which the model reads as context for later readings.

        Args:
            text: The agent's last utterance; empty clears it.
        """
        await self._session.send_agent_text(text)

    def append_audio(self, buffer: bytes, is_speech: bool) -> EndOfTurnState:
        """Queue caller audio for the model and report whether it has ended the turn.

        Args:
            buffer: Mono PCM16 audio at the analyzer's sample rate.
            is_speech: Whether the pipeline's VAD marks the audio as speech.

        Returns:
            ``COMPLETE`` once after the model ends a turn, otherwise ``INCOMPLETE``.
        """
        if self._connection is None:
            self._connection = asyncio.get_running_loop().create_task(
                self._session.maintain(self._report_error)
            )
        if is_speech:
            self._speech_stopped_at = None
        elif self._speech_stopped_at is None:
            self._speech_stopped_at = time.perf_counter()
        self._session.push_audio(buffer, self.sample_rate)
        if self._turn_ended and not self._turn_reported:
            self._turn_reported = True
            return EndOfTurnState.COMPLETE
        return EndOfTurnState.INCOMPLETE

    async def analyze_end_of_turn(self) -> tuple[EndOfTurnState, MetricsData | None]:
        """Report whether the model has ended the turn, with its ending reading's metrics.

        Returns:
            ``COMPLETE`` from the model's ``turn.ended`` until ``clear`` or its next
            ``turn.started``, and the ending reading's metrics on the first call that reports it.
        """
        state = EndOfTurnState.COMPLETE if self._turn_ended else EndOfTurnState.INCOMPLETE
        metrics, self._last_metrics = self._last_metrics, None
        return state, metrics

    def clear(self) -> None:
        """Forget the current turn's state; the session stays open."""
        self._speech_triggered = False
        self._turn_ended = False
        self._turn_reported = False
        self._last_metrics = None
        self._speech_stopped_at = None

    async def cleanup(self) -> None:
        """Close the session; the next ``append_audio`` opens a new one."""
        connection, self._connection = self._connection, None
        if connection is not None:
            connection.cancel()
            await asyncio.gather(connection, return_exceptions=True)
        await self._session.close()
        self.clear()

    async def _report_error(self, error: Exception) -> None:
        logger.error(f"Error connecting to Sprag turns: {error}")

    async def _on_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "turn.started":
            stopped = self._speech_stopped_at
            self.clear()
            self._speech_stopped_at = stopped
            self._speech_triggered = True
        elif event_type == "turn.ended":
            self._speech_triggered = False
            self._turn_ended = True
            self._turn_reported = False
            stopped = self._speech_stopped_at
            self._last_metrics = TurnMetricsData(
                processor="SpragTurnAnalyzer",
                is_complete=True,
                probability=float(event.get("p_eot") or 0.0),
                e2e_processing_time_ms=(time.perf_counter() - stopped) * 1000
                if stopped is not None
                else 0.0,
            )
        elif event_type == "error":
            logger.error(f"Sprag turns error: {error_message(event)}")
