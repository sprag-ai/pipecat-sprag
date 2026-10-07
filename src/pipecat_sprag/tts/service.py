"""Sprag's Chorus models as a Pipecat text-to-speech service."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from collections import Counter
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

from pipecat_sprag.common import ATTRIBUTION_HEADERS, SPRAG_SPEECH_URL, resolve_api_key

SAMPLE_RATE = 24000


@dataclass
class SpragTTSSettings(TTSSettings):
    """Settings for SpragTTSService."""


class SpragTTSService(InterruptibleTTSService):
    """Streaming speech with Sprag's Chorus models over Sprag's streaming speech socket.

    Each sentence is spoken as its own response, sent as soon as Pipecat hands it over;
    the session queues responses and speaks them in order. Each response carries its
    turn's context id as metadata, which routes its audio back to that turn. Sprag
    cannot stop a response once it has started, so an interruption replaces the
    connection rather than cancelling.
    """

    Settings = SpragTTSSettings
    _settings: SpragTTSSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        url: str = SPRAG_SPEECH_URL,
        connect_timeout: float = 10.0,
        settings: SpragTTSSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sprag TTS service.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment
                variable.
            url: The Sprag streaming speech WebSocket URL.
            connect_timeout: Seconds the session may take to be configured.
            settings: Runtime-updatable settings. Defaults to the ``chorus`` model
                and the ``wade`` voice.
            **kwargs: Additional arguments passed to InterruptibleTTSService.
        """
        default_settings = self.Settings(model="chorus", voice="wade", language=None)
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
        self._connect_timeout = connect_timeout
        self._receive_task: asyncio.Task[None] | None = None
        self._unfinished: Counter[str] = Counter()
        self._flushed: set[str] = set()
        self._unstarted: dict[str, tuple[str, str]] = {}
        self._resend: list[tuple[str, str]] = []
        self._speaking_context: str | None = None
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
            # model and voice are fixed for the life of a session
            await self._disconnect()
            await self._connect()
        return changed

    async def _connect(self) -> None:
        await super()._connect()
        await self._connect_websocket()
        if self._websocket and (not self._receive_task or self._receive_task.done()):
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
        if self._websocket is not None:
            # closed without passing through _disconnect_websocket, e.g. with no reconnect
            await self._strand()
        self._websocket = None
        parts = urlsplit(self._url)
        query = "&".join(q for q in (parts.query, urlencode({"model": self._settings.model})) if q)
        # an interrupted turn's stranded sentences are dropped
        stranded, self._resend = self._resend, []
        resend = []
        for context_id, text in stranded:
            if self.audio_context_available(context_id):
                resend.append((context_id, text))
            else:
                await self._finish_response(context_id)
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
                self._configure_session(websocket), timeout=self._connect_timeout
            )
            # sent before the socket is visible, so they keep their place ahead of new sentences
            for context_id, text in resend:
                await self._request(websocket, context_id, text)
        except BaseException as e:
            if websocket:
                await websocket.close()
            self._unstarted.clear()
            self._resend = resend + self._resend
            if not isinstance(e, Exception):
                raise
            await self.push_error(error_msg=f"Error connecting to Sprag TTS: {e}", exception=e)
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
            await self._strand()
            await self._call_event_handler("on_disconnected")

    async def _strand(self) -> None:
        # the response being spoken is lost; queued ones go out on the next session
        self._resend += self._unstarted.values()
        self._unstarted.clear()
        speaking, self._speaking_context = self._speaking_context, None
        if speaking is not None:
            await self._finish_response(speaking)

    async def flush_audio(self, context_id: str | None = None) -> None:
        """Close the turn's audio context once its last response has finished.

        Args:
            context_id: The turn's audio context.
        """
        if not context_id:
            return
        if self._unfinished[context_id]:
            self._flushed.add(context_id)
        else:
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
            if event_type == "response.created":
                self._on_response_created(event)
            elif event_type == "response.output_audio.delta":
                await self._on_audio(event)
            elif event_type == "response.done":
                await self._on_response_done(event)
            elif event_type == "error":
                await self._on_error(event)

    def _on_response_created(self, event: dict[str, Any]) -> None:
        context_id = _context_of(event)
        self._speaking_context = context_id
        # responses start in the order they were asked for, so this answers the turn's oldest
        for event_id, (owner, _) in self._unstarted.items():
            if owner == context_id:
                del self._unstarted[event_id]
                break

    async def _on_audio(self, event: dict[str, Any]) -> None:
        context_id = self._speaking_context
        if context_id is None or not self.audio_context_available(context_id):
            return  # an interrupted turn's queued responses still play out on the session
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
        self._speaking_context = None
        context_id = _context_of(event)
        if context_id is not None:
            await self._finish_response(context_id)

    async def _on_error(self, event: dict[str, Any]) -> None:
        await self.push_error(error_msg=_error_message(event))
        # a refused response.create opens no response
        refused = self._unstarted.pop((event.get("error") or {}).get("event_id") or "", None)
        if refused is not None:
            await self._finish_response(refused[0])

    async def _finish_response(self, context_id: str) -> None:
        self._unfinished[context_id] -= 1
        if self._unfinished[context_id] > 0:
            return
        del self._unfinished[context_id]
        if context_id in self._flushed:
            self._flushed.discard(context_id)
            await self._close_context(context_id)

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        """Speak one sentence as a response on the session.

        Args:
            text: The sentence to speak.
            context_id: The turn's audio context.

        Yields:
            Nothing directly; audio arrives on the receive task.
        """
        reconnecting = self._receive_task is not None and not self._receive_task.done()
        if self._websocket and self._websocket.state is not State.OPEN and reconnecting:
            # the receive loop's reconnect sends it, behind the sentences the drop stranded
            self._unfinished[context_id] += 1
            self._resend.append((context_id, text))
            await self.start_tts_usage_metrics(text)
            yield None
            return
        if not self._websocket or self._websocket.state is not State.OPEN:
            await self._connect()
        if not self._websocket:
            yield ErrorFrame(error="Sprag TTS is not connected")
            return
        self._unfinished[context_id] += 1
        websocket = self._websocket
        try:
            await self._request(websocket, context_id, text)
        except Exception as e:
            if websocket.state is State.OPEN:
                await self._finish_response(context_id)
                yield ErrorFrame(error=f"Sprag TTS send error: {e}")
                return
            self._resend.append((context_id, text))
        await self.start_tts_usage_metrics(text)
        yield None

    async def _request(self, websocket: ClientConnection, context_id: str, text: str) -> None:
        item = {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": text}],
        }
        event_id = f"evt_{uuid.uuid4().hex}"
        create = {
            "type": "response.create",
            "event_id": event_id,
            "response": {"metadata": {_CONTEXT_KEY: context_id}},
        }
        # recorded first: the response can start before the send returns
        self._unstarted[event_id] = (context_id, text)
        try:
            await websocket.send(json.dumps({"type": "conversation.item.create", "item": item}))
            await websocket.send(json.dumps(create))
        except Exception:
            if self._unstarted.pop(event_id, None) is None:
                return  # the drop already queued it for the next session
            raise


_CONTEXT_KEY = "context_id"


def _context_of(event: dict[str, Any]) -> str | None:
    metadata = (event.get("response") or {}).get("metadata") or {}
    context_id = metadata.get(_CONTEXT_KEY)
    return context_id if isinstance(context_id, str) else None


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error") or {}
    return f"Sprag TTS error [{error.get('code', 'unknown')}]: {error.get('message', '')}"
