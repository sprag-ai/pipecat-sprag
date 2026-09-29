"""Sprag's Chorus models as a Pipecat text-to-speech service."""

from __future__ import annotations

import asyncio
import base64
import json
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    StartFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.services.settings import TTSSettings
from pipecat.services.tts_service import InterruptibleTTSService
from websockets.asyncio.client import ClientConnection
from websockets.protocol import State

from pipecat_sprag.common import ATTRIBUTION_HEADERS, SPRAG_REALTIME_URL, resolve_api_key

SAMPLE_RATE = 24000
# Pipecat's default wait for audio between frames of a turn
FIRST_AUDIO_TIMEOUT = 3.0


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
            response_timeout: Seconds a response may go without sending audio before it
                counts as stalled, and seconds the session may take to be configured.
            settings: Runtime-updatable settings. Defaults to the ``chorus-clone`` model
                and the ``wade`` voice.
            **kwargs: Additional arguments passed to InterruptibleTTSService.
        """
        default_settings = self.Settings(model="chorus-clone", voice="wade", language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        # a turn's audio context must outlast the longest gap between two of its responses:
        # a stall, the reconnect that replaces it, and the next response's first audio
        kwargs.setdefault("stop_frame_timeout_s", 2 * response_timeout + FIRST_AUDIO_TIMEOUT)
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
        self._stall_watch: asyncio.Task[None] | None = None
        # the session speaks one response at a time: set while none is in progress
        self._idle = asyncio.Event()
        self._idle.set()
        self._response_context: str | None = None
        self._last_activity = 0.0
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
        if self._websocket and (not self._receive_task or self._receive_task.done()):
            self._receive_task = self.create_task(self._receive_task_handler(self._report_error))

    async def _disconnect(self) -> None:
        if self._stall_watch and self._stall_watch is not asyncio.current_task():
            await self.cancel_task(self._stall_watch)
        self._stall_watch = None
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
        self._websocket = None
        parts = urlsplit(self._url)
        query = "&".join(q for q in (parts.query, urlencode({"model": self._settings.model})) if q)
        websocket: ClientConnection | None = None
        try:
            websocket = await self._websocket_connect(
                urlunsplit(parts._replace(query=query)),
                additional_headers={
                    "Authorization": f"Bearer {self._api_key}",
                    **ATTRIBUTION_HEADERS,
                },
            )
            await asyncio.wait_for(
                self._configure_session(websocket), timeout=self._response_timeout
            )
        except Exception as e:
            await self.push_error(error_msg=f"Error connecting to Sprag TTS: {e}", exception=e)
            if websocket:
                await websocket.close()
            return
        # run_tts may send as soon as the socket is visible, so it becomes visible configured
        self._websocket = websocket
        await self._call_event_handler("on_connected")

    async def _configure_session(self, websocket: ClientConnection) -> None:
        session = {"type": "realtime", "audio": {"output": {"voice": self._settings.voice}}}
        await websocket.send(json.dumps({"type": "session.update", "session": session}))
        async for message in websocket:
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
            self._stall_watch = self.create_task(self._end_turn_if_stalled(context_id))
        elif context_id:
            await self._close_context(context_id)

    async def _close_context(self, context_id: str) -> None:
        if self.audio_context_available(context_id):
            await self.append_to_audio_context(context_id, TTSStoppedFrame(context_id=context_id))
            await self.remove_audio_context(context_id)

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
        self._last_activity = asyncio.get_running_loop().time()
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
        if not await self._wait_until_idle():
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
        self._last_activity = asyncio.get_running_loop().time()
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

    async def _end_turn_if_stalled(self, context_id: str) -> None:
        if await self._wait_until_idle() or self._flushed_context != context_id:
            return
        await self.push_error(error_msg="Sprag TTS: the last response did not finish")
        # the disconnect ends the flushed turn, and the stalled response holds the session
        await self._disconnect()
        await self._connect()

    async def _wait_until_idle(self) -> bool:
        """Wait for the active response to finish, as long as it keeps sending audio."""
        loop = asyncio.get_running_loop()
        while not self._idle.is_set():
            remaining = self._last_activity + self._response_timeout - loop.time()
            if remaining <= 0:
                return False
            try:
                await asyncio.wait_for(self._idle.wait(), timeout=remaining)
            except TimeoutError:
                pass
        return True


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error") or {}
    return f"Sprag TTS error [{error.get('code', 'unknown')}]: {error.get('message', '')}"
