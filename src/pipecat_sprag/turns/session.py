"""A client for Sprag's ``/v1/turns`` socket, shared by the turn detector and the turn analyzer."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from loguru import logger
from pipecat.audio.turn.base_turn_analyzer import BaseTurnParams
from pipecat.audio.utils import create_stream_resampler
from pydantic import Field
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from pipecat_sprag.common import ATTRIBUTION_HEADERS

SAMPLE_RATE = 16000

EventHandler = Callable[[dict[str, Any]], Awaitable[None]]


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
    ``on_event`` in the order the server sends them; ``connect`` returns once the session is
    configured.
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
        """Prepare a session; nothing is opened until ``connect``.

        Args:
            api_key: The Sprag API key.
            url: The ``/v1/turns`` WebSocket URL.
            model: The turn model to open the session on.
            params: The thresholds sent as the session's configuration.
            on_event: Called with every server event after the session is configured.
            connect_timeout: Seconds the session may take to open and be configured.
        """
        self._api_key = api_key
        self._url = url
        self._model = model
        self._params = params
        self._on_event = on_event
        self._connect_timeout = connect_timeout
        self._websocket: ClientConnection | None = None
        self._receive_task: asyncio.Task[None] | None = None
        self._resampler = create_stream_resampler()

    @property
    def connected(self) -> bool:
        """Whether the session is open and configured."""
        return self._websocket is not None

    async def connect(self) -> None:
        """Open the socket and apply the session's thresholds.

        Raises:
            RuntimeError: The server refused the configuration.
            ConnectionError: The socket closed before the session was configured.
            TimeoutError: The session took longer than ``connect_timeout`` to configure.
        """
        if self._websocket is not None:
            return
        parts = urlsplit(self._url)
        query = "&".join(q for q in (parts.query, urlencode({"model": self._model})) if q)
        websocket = await connect(
            urlunsplit(parts._replace(query=query)),
            additional_headers={"Authorization": f"Bearer {self._api_key}", **ATTRIBUTION_HEADERS},
            open_timeout=self._connect_timeout,
            max_size=None,
        )
        try:
            await asyncio.wait_for(self._configure(websocket), timeout=self._connect_timeout)
        except BaseException:
            await websocket.close()
            raise
        self._websocket = websocket
        self._receive_task = asyncio.create_task(self._receive(websocket))

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
                raise RuntimeError(_error_message(event))
        raise ConnectionError("the turns session closed before it was configured")

    async def send_audio(self, audio: bytes, sample_rate: int) -> None:
        """Stream caller audio.

        Args:
            audio: Mono PCM16 little-endian samples.
            sample_rate: The audio's sample rate in hertz.
        """
        websocket = self._websocket
        if websocket is None:
            return
        if sample_rate != SAMPLE_RATE:
            audio = await self._resampler.resample(audio, sample_rate, SAMPLE_RATE)
        if not audio:
            return
        try:
            await websocket.send(audio)
        except ConnectionClosed:
            await self._lost(websocket)

    async def send_agent_text(self, text: str) -> None:
        """Set what the agent last said, which every later reading takes as context.

        Args:
            text: The agent's last utterance; empty clears the context.
        """
        websocket = self._websocket
        if websocket is None:
            return
        try:
            await websocket.send(json.dumps({"type": "context.update", "agent_text": text}))
        except ConnectionClosed:
            await self._lost(websocket)

    async def close(self) -> None:
        """End the session and close the socket."""
        websocket, self._websocket = self._websocket, None
        if websocket is not None:
            try:
                await websocket.send(json.dumps({"type": "session.close"}))
            except ConnectionClosed:
                pass
            await websocket.close()
        task, self._receive_task = self._receive_task, None
        if task is not None and task is not asyncio.current_task():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    async def _receive(self, websocket: ClientConnection) -> None:
        try:
            async for message in websocket:
                if isinstance(message, str):
                    await self._on_event(json.loads(message))
        except ConnectionClosed:
            pass
        await self._lost(websocket)

    async def _lost(self, websocket: ClientConnection) -> None:
        if self._websocket is websocket:
            logger.warning("Sprag turns session closed by the server")
            self._websocket = None


def _error_message(event: dict[str, Any]) -> str:
    error = event.get("error") or {}
    return str(
        error.get("message") or error.get("code") or "the turns session refused the configuration"
    )
