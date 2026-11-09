"""Sprag's turn model as a Pipecat processor that signals when the user's turn is complete."""

from __future__ import annotations

import time
from typing import Any

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    UserTurnInferenceCompletedFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from pipecat_sprag.common import SPRAG_TURNS_URL, resolve_api_key
from pipecat_sprag.turns.session import SpragTurnParams, TurnsSession

RECONNECT_INTERVAL_SECS = 2.0


class SpragTurnDetector(FrameProcessor):
    """Streams the caller's audio to Sprag's turn model; pushes ``UserTurnInferenceCompletedFrame``.

    Place it after the transport input and pair it with
    ``ExternalUserTurnCompletionStopStrategy``, so the model's ``turn.ended`` ends the user's
    turn. Every frame passes through unchanged. Its ``on_turn_update`` event receives each
    end-of-turn reading and its ``on_turn_ended`` event each closed turn.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = SPRAG_TURNS_URL,
        model: str = "sprag-turns",
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
            connect_timeout: Seconds the session may take to open and be configured.
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
        self._agent_text = ""
        self._next_connect = 0.0
        self._register_event_handler("on_turn_update")
        self._register_event_handler("on_turn_ended")

    async def set_agent_text(self, text: str) -> None:
        """Set what the agent last said, which the model reads as context for later readings.

        Args:
            text: The agent's last utterance; empty clears it.
        """
        self._agent_text = text
        await self._session.send_agent_text(text)

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        """Stream input audio to the model and pass every frame on.

        Args:
            frame: The frame to process.
            direction: The frame's direction.
        """
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            await self._connect()
            return
        if isinstance(frame, (EndFrame, CancelFrame)):
            await self._session.close()
        elif isinstance(frame, InputAudioRawFrame) and frame.num_channels == 1:
            if not self._session.connected and time.monotonic() >= self._next_connect:
                await self._connect()
            await self._session.send_audio(frame.audio, frame.sample_rate)
        await self.push_frame(frame, direction)

    async def cleanup(self) -> None:
        """Close the session."""
        await super().cleanup()
        await self._session.close()

    async def _connect(self) -> None:
        self._next_connect = time.monotonic() + RECONNECT_INTERVAL_SECS
        try:
            await self._session.connect()
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Sprag turns: {e}", exception=e)
            return
        if self._agent_text:
            await self._session.send_agent_text(self._agent_text)

    async def _on_event(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "turn.update":
            await self._call_event_handler("on_turn_update", event)
        elif event_type == "turn.ended":
            await self.push_frame(UserTurnInferenceCompletedFrame())
            await self._call_event_handler("on_turn_ended", event)
        elif event_type == "error":
            error = event.get("error") or {}
            await self.push_error(
                error_msg=f"Sprag turns error: {error.get('message') or error.get('code')}"
            )
