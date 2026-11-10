"""A client for Sprag's ``/v1/turns`` socket, shared by the turn detector and the turn analyzer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnParams
from pipecat.audio.utils import create_stream_resampler
from pydantic import Field
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from pipecat_sprag.common import ATTRIBUTION_HEADERS

SAMPLE_RATE = 16000
RECONNECT_INTERVAL_SECS = 2.0
MAX_RECONNECT_INTERVAL_SECS = 30.0
AUDIO_QUEUE_FRAMES = 25
MAX_MESSAGE_BYTES = 2**16
CLOSE_TIMEOUT_SECS = 2.0

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]


class SpragTurnParams(BaseTurnParams):
    """Thresholds the Sprag turn model applies to its readings."""

    eot_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="end-of-turn probability that ends the turn"
    )
    min_silence_ms: int = Field(
        default=200, ge=0, description="silence the turn needs before the model can end it"
    )
    eot_timeout_ms: int = Field(
        default=3000,
        ge=100,
        le=60000,
        description="silence that ends the turn whatever the model scores",
    )


class TurnsSession:
    """One ``/v1/turns`` session: streams caller audio and agent context, and reports turn events.

    Audio of any sample rate is resampled to the 16 kHz PCM16 the socket reads. Events reach
    ``on_event`` in the order the server sends them. The agent text is kept across reconnects
    and sent again on every new connection.
    """

    def __init__(
        self,
        *,
        api_key: str,
        url: str,
        model: str,
        params: SpragTurnParams,
        on_event: EventHandler,
        connect_timeout: float = 10.0,
    ) -> None:
        """Prepare a session; nothing is opened until ``connect`` or ``maintain``.

        Args:
            api_key: The Sprag API key.
            url: The ``/v1/turns`` WebSocket URL.
            model: The turn model to open the session on, unless ``url`` already names one.
            params: The thresholds sent as the session's configuration.
            on_event: Called with every server event after the session is configured.
            connect_timeout: Seconds the socket may take to open, and again to be configured.
        """
        self._api_key = api_key
        self._url = url
        self._model = model
        self._params = params
        self._on_event = on_event
        self._connect_timeout = connect_timeout
        self._websocket: ClientConnection | None = None
        self._tasks: list[asyncio.Task[None]] = []
        self._closing: asyncio.Task[None] | None = None
        self._audio: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=AUDIO_QUEUE_FRAMES)
        self._lost = asyncio.Event()
        self._agent_text = ""
        self._resampler = create_stream_resampler()

    @property
    def connected(self) -> bool:
        """Whether the session is open and configured."""
        return self._websocket is not None

    async def maintain(self, on_error: ErrorHandler) -> None:
        """Keep the session connected until the calling task is cancelled.

        A failed attempt goes to ``on_error`` and is retried after a delay that starts at
        ``RECONNECT_INTERVAL_SECS`` and doubles up to ``MAX_RECONNECT_INTERVAL_SECS``.

        Args:
            on_error: Called with the error of each failed connection attempt.
        """
        delay = RECONNECT_INTERVAL_SECS
        while True:
            try:
                await self.connect()
            except Exception as e:
                await on_error(e)
                await asyncio.sleep(delay)
                delay = min(delay * 2, MAX_RECONNECT_INTERVAL_SECS)
                continue
            delay = RECONNECT_INTERVAL_SECS
            await self._lost.wait()

    async def connect(self) -> None:
        """Open the socket, apply the session's thresholds and send the agent text.

        Raises:
            RuntimeError: The server refused the configuration.
            ConnectionError: The socket closed before the session was configured.
            TimeoutError: The session took longer than ``connect_timeout`` to configure.
        """
        if self._websocket is not None:
            return
        websocket = await connect(
            self._session_url(),
            additional_headers={"Authorization": f"Bearer {self._api_key}", **ATTRIBUTION_HEADERS},
            open_timeout=self._connect_timeout,
            close_timeout=CLOSE_TIMEOUT_SECS,
            max_size=MAX_MESSAGE_BYTES,
        )
        try:
            await asyncio.wait_for(self._configure(websocket), timeout=self._connect_timeout)
            if self._agent_text:
                await websocket.send(_context_update(self._agent_text))
        except BaseException:
            await websocket.close()
            raise
        while not self._audio.empty():
            self._audio.get_nowait()
        self._resampler = create_stream_resampler()
        self._websocket = websocket
        self._lost.clear()
        self._tasks = [
            asyncio.create_task(self._receive(websocket)),
            asyncio.create_task(self._send_audio(websocket)),
        ]

    def push_audio(self, audio: bytes, sample_rate: int) -> None:
        """Queue caller audio for the socket without waiting on it.

        Audio is dropped while the session is not connected, and the oldest queued audio is
        dropped once ``AUDIO_QUEUE_FRAMES`` buffers are waiting.

        Args:
            audio: Mono PCM16 little-endian samples.
            sample_rate: The audio's sample rate in hertz.
        """
        if self._websocket is None:
            return
        if self._audio.full():
            self._audio.get_nowait()
        self._audio.put_nowait((audio, sample_rate))

    async def send_agent_text(self, text: str) -> None:
        """Set what the agent last said, which every later reading takes as context.

        Args:
            text: The agent's last utterance; empty clears the context.
        """
        self._agent_text = text
        websocket = self._websocket
        if websocket is None:
            return
        try:
            await websocket.send(_context_update(text))
        except ConnectionClosed:
            self._drop(websocket)

    async def close(self) -> None:
        """End the session and close the socket; a later ``connect`` opens a new one."""
        websocket, self._websocket = self._websocket, None
        tasks, self._tasks = self._tasks, []
        current = asyncio.current_task()
        others = [task for task in tasks if task is not current]
        for task in others:
            task.cancel()
        await asyncio.gather(*others, return_exceptions=True)
        if websocket is not None:
            try:
                await websocket.send(json.dumps({"type": "session.close"}))
            except ConnectionClosed:
                pass
            await websocket.close()

    def _session_url(self) -> str:
        parts = urlsplit(self._url)
        if parts.scheme == "ws" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
            logger.warning("The Sprag turns URL is not encrypted; the API key is sent in plaintext")
        if "model" in parse_qs(parts.query):
            return self._url
        query = "&".join(q for q in (parts.query, urlencode({"model": self._model})) if q)
        return urlunsplit(parts._replace(query=query))

    async def _configure(self, websocket: ClientConnection) -> None:
        await websocket.send(
            json.dumps({"type": "session.update", "session": self._params.model_dump()})
        )
        async for message in websocket:
            if not isinstance(message, str):
                continue
            event = json.loads(message)
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(error_message(event))
        raise ConnectionError("the turns session closed before it was configured")

    async def _receive(self, websocket: ClientConnection) -> None:
        try:
            async for message in websocket:
                if not isinstance(message, str):
                    continue
                try:
                    event = json.loads(message)
                    if isinstance(event, dict):
                        await self._on_event(event)
                except Exception:
                    logger.exception("Error handling a Sprag turns event")
        except ConnectionClosed:
            pass
        finally:
            self._drop(websocket)

    async def _send_audio(self, websocket: ClientConnection) -> None:
        try:
            while True:
                audio, sample_rate = await self._audio.get()
                if sample_rate != SAMPLE_RATE:
                    audio = await self._resampler.resample(audio, sample_rate, SAMPLE_RATE)
                if audio:
                    await websocket.send(audio)
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("Error streaming audio to Sprag turns")
        finally:
            self._drop(websocket)

    def _drop(self, websocket: ClientConnection) -> None:
        if self._websocket is not websocket:
            return
        logger.warning("Sprag turns session lost")
        self._websocket = None
        current = asyncio.current_task()
        for task in self._tasks:
            if task is not current:
                task.cancel()
        self._tasks = []
        self._closing = asyncio.create_task(websocket.close())
        self._lost.set()


def error_message(event: dict[str, Any]) -> str:
    """Describe a server ``error`` event.

    Args:
        event: The server's ``error`` event.

    Returns:
        The error's message, else its code, else a generic description.
    """
    error = event.get("error") or {}
    return str(error.get("message") or error.get("code") or "the turns session reported an error")


def _context_update(text: str) -> str:
    return json.dumps({"type": "context.update", "agent_text": text})
