"""Sprag's Chorus models as a Pipecat text-to-speech service."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode

from pipecat.frames.frames import (
    CancelFrame,
    EndFrame,
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from websockets.protocol import State

from pipecat_sprag.common import ATTRIBUTION_HEADERS, SPRAG_REALTIME_URL, resolve_api_key

SAMPLE_RATE = 24000


@dataclass
class SpragTTSSettings(TTSSettings):
    """Settings for SpragTTSService."""


class SpragTTSService(InterruptibleTTSService):
    """Streaming speech with Sprag's Chorus models over Sprag's realtime speech session.

    Each sentence is spoken as its own response. The session allows one response at
    a time, so a sentence waits for the previous one to finish generating. Sprag
    cannot stop a response once it has started, so an interruption replaces the
    connection rather than cancelling.
    """

    Settings = SpragTTSSettings
    _settings: SpragTTSSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = SPRAG_REALTIME_URL,
        response_timeout: float = 10.0,
        settings: SpragTTSSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sprag TTS service.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment
                variable.
            url: The Sprag realtime WebSocket URL.
            response_timeout: Seconds a sentence may wait for the previous response to
                finish, and for the session to be configured.
            settings: Runtime-updatable settings. Defaults to the ``chorus-clone`` model
                and the ``wade`` voice.
            **kwargs: Additional arguments passed to InterruptibleTTSService.
        """
        default_settings = self.Settings(model="chorus-clone", voice="wade", language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            pause_frame_processing=True,
            sample_rate=SAMPLE_RATE,
            settings=default_settings,
            **kwargs,
        )
        self._api_key = resolve_api_key(api_key)
        self._url = url
        self._response_timeout = response_timeout
        self._receive_task: asyncio.Task[None] | None = None
        # the session speaks one response at a time: set while none is in progress
        self._idle = asyncio.Event()
        self._idle.set()
        self._response_context: str | None = None
        self._flushed_context: str | None = None
        # run_tts and the receive loop's reconnect may both reach for a new socket
        self._connect_lock = asyncio.Lock()

    def can_generate_metrics(self) -> bool:
        """Report that this service emits TTFB and usage metrics.

        Returns:
            True.
        """
        return True

    async def start(self, frame: StartFrame) -> None:
        """Connect when the pipeline starts.

        Args:
            frame: The start frame.
        """
        await super().start(frame)
        await self._connect()

    async def stop(self, frame: EndFrame) -> None:
        """Disconnect when the pipeline ends.

        Args:
            frame: The end frame.
        """
        await super().stop(frame)
        await self._disconnect()

    async def cancel(self, frame: CancelFrame) -> None:
        """Disconnect when the pipeline is cancelled.

        Args:
            frame: The cancel frame.
        """
        await super().cancel(frame)
        await self._disconnect()

    async def _update_settings(self, delta: TTSSettings) -> dict[str, Any]:  # type: ignore[override]
        changed = await super()._update_settings(delta)
        if changed:
            # the model and voice belong to the session, so a change needs a new one
            await self._disconnect()
            await self._connect()
        return changed

    async def _connect(self) -> None:
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and not self._receive_task:
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        await super()._disconnect()
        if self._receive_task:
            await self.cancel_task(self._receive_task, timeout=1.0)
            self._receive_task = None
        await self._disconnect_websocket()

    async def _connect_websocket(self) -> None:
        async with self._connect_lock:
            await self._open_session()

    async def _open_session(self) -> None:
        if self._websocket and self._websocket.state is State.OPEN:
            return
        self._reset_response_state()
        url = f"{self._url}?{urlencode({'model': self._settings.model})}"
        try:
            self._websocket = await self._websocket_connect(
                url,
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    **ATTRIBUTION_HEADERS,
                },
            )
            await asyncio.wait_for(self._configure_session(), timeout=self._response_timeout)
            await self._call_event_handler("on_connected")
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Sprag TTS: {e}", exception=e)
            if self._websocket:
                await self._websocket.close()
            self._websocket = None

    async def _configure_session(self) -> None:
        assert self._websocket is not None
        session = {"type": "realtime", "audio": {"output": {"voice": self._settings.voice}}}
        await self._websocket.send(json.dumps({"type": "session.update", "session": session}))
        async for message in self._websocket:
            event = json.loads(message)
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(_error_message(event))
        raise ConnectionError("the session closed before it was configured")

    async def _disconnect_websocket(self) -> None:
        try:
            if self._websocket:
                await self._websocket.close()
        finally:
            self._websocket = None
            # a flush that was waiting on the lost response still has to end the turn
            flushed = self._flushed_context
            self._reset_response_state()
            if flushed is not None:
                await self._close_context(flushed)
            await self._call_event_handler("on_disconnected")

    def _reset_response_state(self) -> None:
        self._response_context = None
        self._flushed_context = None
        self._idle.set()

    async def flush_audio(self, context_id: str | None = None) -> None:
        """Close the turn's audio context once its last response has finished.

        Args:
            context_id: The turn's audio context.
        """
        if context_id and not self._idle.is_set() and self._response_context == context_id:
            self._flushed_context = context_id
        elif context_id:
            await self._close_context(context_id)

    async def _close_context(self, context_id: str) -> None:
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

    async def on_audio_context_interrupted(self, context_id: str) -> None:
        """Stop metrics when the caller interrupts.

        Args:
            context_id: The interrupted audio context.
        """
        await self.stop_all_metrics()
        await super().on_audio_context_interrupted(context_id)

    async def _receive_messages(self) -> None:
        assert self._websocket is not None
        async for message in self._websocket:
            if not isinstance(message, str):
                continue
            event = json.loads(message)
            event_type = event.get("type")
            if event_type == "response.output_audio.delta":
                await self._on_audio(event)
            elif event_type == "response.done":
                await self._on_response_done(event)
            elif event_type == "error":
                await self._on_error(event)

    async def _on_audio(self, event: dict[str, Any]) -> None:
        context_id = self._response_context
        if context_id is None:
            return
        await self.stop_ttfb_metrics()
        frame = TTSAudioRawFrame(
            audio=base64.b64decode(event["delta"]),
            sample_rate=self.sample_rate,
            num_channels=1,
            context_id=context_id,
        )
        await self.append_to_audio_context(context_id, frame)

    async def _on_response_done(self, event: dict[str, Any]) -> None:
        response = event.get("response") or {}
        if response.get("status") != "completed":
            error = (response.get("status_details") or {}).get("error") or {}
            await self.push_error(
                error_msg=f"Sprag TTS response ended as {response.get('status')}: "
                f"{error.get('code', 'unknown')}"
            )
        await self._finish_response()

    async def _on_error(self, event: dict[str, Any]) -> None:
        await self.push_error(error_msg=_error_message(event))
        await self._finish_response()

    async def _finish_response(self) -> None:
        context_id, flushed = self._response_context, self._flushed_context
        self._response_context = None
        self._flushed_context = None
        self._idle.set()
        if context_id is not None and flushed == context_id:
            await self._close_context(context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Speak one sentence as a response on the session.

        Args:
            text: The sentence to speak.
            context_id: The turn's audio context.

        Yields:
            Nothing directly; audio arrives on the receive task.
        """
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()
        if not self._websocket:
            yield ErrorFrame(error="Sprag TTS is not connected")
            return
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=self._response_timeout)
        except TimeoutError:
            # the stalled response holds the session, so this sentence goes out on a new one
            yield ErrorFrame(error="Sprag TTS: the previous response did not finish")
            await self._disconnect()
            await self._connect()
        # a dropped connection opens the gate while the receive loop reconnects
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect_websocket()
        if not self._websocket:
            yield ErrorFrame(error="Sprag TTS is not connected")
            return
        self._idle.clear()
        self._response_context = context_id
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        try:
            await self._websocket.send(
                json.dumps({"type": "conversation.item.create", "item": item})
            )
            await self._websocket.send(json.dumps({"type": "response.create"}))
        except Exception as e:
            self._reset_response_state()
            yield ErrorFrame(error=f"Sprag TTS send error: {e}")
            return
        await self.start_tts_usage_metrics(text)
        yield None


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error") or {}
    return f"Sprag TTS error [{error.get('code', 'unknown')}]: {error.get('message', '')}"
