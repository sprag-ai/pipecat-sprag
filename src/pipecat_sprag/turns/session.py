"""A client for Sprag's ``/v1/turns`` socket, shared by the turn detector and the turn analyzer."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Callable
from urllib.parse import parse_qs, urlencode, urlsplit, urlunsplit

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnParams
from pipecat.audio.utils import create_stream_resampler
from pydantic import Field
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from pipecat_sprag.common import ATTRIBUTION_HEADERS
from pipecat_sprag.turns.events import (
    SessionClosed,
    TurnsError,
    TurnsServerEvent,
    TurnsUsage,
    parse_event,
)

SAMPLE_RATE = 16000
# A reading scores at most this much audio, so older audio cannot change one.
SCORING_WINDOW_SECS = 8
MAX_BUFFERED_AUDIO_BYTES = SCORING_WINDOW_SECS * SAMPLE_RATE * 2
RECONNECT_INTERVAL_SECS = 2.0
MAX_RECONNECT_INTERVAL_SECS = 30.0
AUDIO_QUEUE_FRAMES = 25
MAX_MESSAGE_BYTES = 2**16
CLOSE_TIMEOUT_SECS = 2.0
SESSION_CLOSE_TIMEOUT_SECS = 2.0

EventHandler = Callable[[TurnsServerEvent], Awaitable[None]]
ErrorHandler = Callable[[Exception], Awaitable[None]]


class SpragTurnParams(BaseTurnParams):
    """Thresholds the Sprag turn model applies to its readings."""

    eot_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0, description="end-of-turn probability that ends the turn"
    )
    eager_eot_threshold: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="end-of-turn probability that sends turn.eager_end; None sends none",
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
        self._receiver: asyncio.Task[None] | None = None
        self._sender: asyncio.Task[None] | None = None
        self._closing: asyncio.Task[None] | None = None
        self._audio: asyncio.Queue[tuple[bytes, int]] = asyncio.Queue(maxsize=AUDIO_QUEUE_FRAMES)
        self._pending: deque[tuple[bytes, int]] = deque()
        self._pending_bytes = 0
        self._lost = asyncio.Event()
        self._agent_text = ""
        self._resampler = create_stream_resampler()
        self._usage: TurnsUsage | None = None

    @property
    def connected(self) -> bool:
        """Whether the session is open and configured."""
        return self._websocket is not None

    @property
    def usage(self) -> TurnsUsage | None:
        """The usage of the last session the server closed with ``session.closed``, if any."""
        return self._usage

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
        """Open the socket, apply the thresholds, then send the agent text and buffered audio.

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
            self._resampler = create_stream_resampler()
            while self._pending:
                await self._send(websocket, *self._pop_pending())
        except BaseException:
            await websocket.close()
            raise
        self._websocket = websocket
        self._lost.clear()
        self._receiver = asyncio.create_task(self._receive(websocket))
        self._sender = asyncio.create_task(self._send_audio(websocket))

    def push_audio(self, audio: bytes, sample_rate: int) -> None:
        """Queue caller audio for the socket without waiting on it.

        While the session is not connected, audio is buffered and sent once it is, keeping
        only the latest ``SCORING_WINDOW_SECS``. While it is connected, the oldest queued
        audio is dropped once ``AUDIO_QUEUE_FRAMES`` buffers are waiting.

        Args:
            audio: Mono PCM16 little-endian samples.
            sample_rate: The audio's sample rate in hertz.

        Raises:
            ValueError: ``sample_rate`` is not positive.
        """
        if sample_rate <= 0:
            raise ValueError(f"the audio's sample rate must be positive, not {sample_rate}")
        if self._websocket is None:
            self._buffer(audio, sample_rate)
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

    async def send_turn_end(self) -> None:
        """Ask the server to end the open turn now; a no-op while disconnected or with no turn."""
        websocket = self._websocket
        if websocket is None:
            return
        try:
            await websocket.send(json.dumps({"type": "turn.end"}))
        except ConnectionClosed:
            self._drop(websocket)

    async def close(self, *, wait: bool = True) -> None:
        """End the session and close the socket; a later ``connect`` opens a new one.

        Unsent audio is discarded. Events that arrive while waiting reach ``on_event``.

        Args:
            wait: Whether to wait up to ``SESSION_CLOSE_TIMEOUT_SECS`` for the server's
                ``session.closed`` before closing the socket.
        """
        websocket, self._websocket = self._websocket, None
        receiver, self._receiver = self._receiver, None
        sender, self._sender = self._sender, None
        self._clear_audio()
        current = asyncio.current_task()
        if sender is not None and sender is not current:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
        if websocket is None:
            return
        try:
            await websocket.send(json.dumps({"type": "session.close"}))
            if wait and receiver is not None and receiver is not current:
                await asyncio.wait({receiver}, timeout=SESSION_CLOSE_TIMEOUT_SECS)
        except ConnectionClosed:
            pass
        finally:
            if receiver is not None and receiver is not current and not receiver.done():
                receiver.cancel()
                await asyncio.gather(receiver, return_exceptions=True)
            await websocket.close()

    def _buffer(self, audio: bytes, sample_rate: int) -> None:
        self._pending.append((audio, sample_rate))
        self._pending_bytes += _bytes_at_16k(audio, sample_rate)
        while self._pending_bytes > MAX_BUFFERED_AUDIO_BYTES:
            self._pop_pending()

    def _pop_pending(self) -> tuple[bytes, int]:
        audio, sample_rate = self._pending.popleft()
        self._pending_bytes -= _bytes_at_16k(audio, sample_rate)
        return audio, sample_rate

    def _clear_audio(self) -> None:
        self._pending.clear()
        self._pending_bytes = 0
        while not self._audio.empty():
            self._audio.get_nowait()

    def _session_url(self) -> str:
        parts = urlsplit(self._url)
        if parts.scheme == "ws" and parts.hostname not in ("localhost", "127.0.0.1", "::1"):
            logger.warning("The Sprag turns URL is not encrypted; the API key is sent in plaintext")
        if "model" in parse_qs(parts.query):
            return self._url
        query = "&".join(q for q in (parts.query, urlencode({"model": self._model})) if q)
        return urlunsplit(parts._replace(query=query))

    async def _configure(self, websocket: ClientConnection) -> None:
        session = self._params.model_dump(exclude_none=True)
        await websocket.send(json.dumps({"type": "session.update", "session": session}))
        async for message in websocket:
            if not isinstance(message, str):
                continue
            event = json.loads(message)
            if event.get("type") == "session.updated":
                return
            if event.get("type") == "error":
                raise RuntimeError(TurnsError.model_validate(event).description)
        raise ConnectionError("the turns session closed before it was configured")

    async def _receive(self, websocket: ClientConnection) -> None:
        try:
            async for message in websocket:
                if not isinstance(message, str):
                    continue
                try:
                    raw = json.loads(message)
                    event = parse_event(raw) if isinstance(raw, dict) else None
                    if event is None:
                        continue
                    if isinstance(event, SessionClosed):
                        self._usage = event.usage
                    await self._on_event(event)
                    if isinstance(event, SessionClosed):
                        return
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
                await self._send(websocket, audio, sample_rate)
        except ConnectionClosed:
            pass
        except Exception:
            logger.exception("Error streaming audio to Sprag turns")
        finally:
            self._drop(websocket)

    async def _send(self, websocket: ClientConnection, audio: bytes, sample_rate: int) -> None:
        if sample_rate != SAMPLE_RATE:
            audio = await self._resampler.resample(audio, sample_rate, SAMPLE_RATE)
        if audio:
            await websocket.send(audio)

    def _drop(self, websocket: ClientConnection) -> None:
        if self._websocket is not websocket:
            return
        logger.warning("Sprag turns session lost")
        self._websocket = None
        # The next connection's stream clock restarts at its first byte, so only audio this
        # connection never sent is carried over to it.
        while not self._audio.empty():
            self._buffer(*self._audio.get_nowait())
        current = asyncio.current_task()
        for task in (self._receiver, self._sender):
            if task is not None and task is not current:
                task.cancel()
        self._receiver = self._sender = None
        self._closing = asyncio.create_task(websocket.close())
        self._lost.set()


def _bytes_at_16k(audio: bytes, sample_rate: int) -> int:
    return len(audio) * SAMPLE_RATE // sample_rate


def _context_update(text: str) -> str:
    return json.dumps({"type": "context.update", "agent_text": text})
