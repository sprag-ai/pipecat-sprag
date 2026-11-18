"""Server events of Sprag's ``/v1/turns`` socket, as typed models."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter


class TurnEvent(BaseModel):
    """Fields every turn event carries."""

    turn_index: int = Field(
        description="the caller's turn, counted from 0 on each connection; restarts on reconnect"
    )
    audio_window_start: float = Field(
        description="seconds from the connection's first audio byte to where the turn's speech "
        "began; restarts on reconnect"
    )
    audio_window_end: float = Field(
        description="seconds from the connection's first audio byte to the point the event "
        "refers to; restarts on reconnect"
    )


class TurnStarted(TurnEvent):
    """The model opened a turn; ``audio_window_end`` is the speech onset."""

    type: Literal["turn.started"] = Field(description="the event type")


class TurnReading(TurnEvent):
    """An end-of-turn reading taken during a pause."""

    probability: float = Field(description="the model's end-of-turn probability, 0 to 1")
    silence: float = Field(
        description="seconds of silence since the caller's last speech, up to audio_window_end"
    )


class TurnUpdate(TurnReading):
    """One reading; every reading is sent as one, including a reading ``turn.ended`` follows."""

    type: Literal["turn.update"] = Field(description="the event type")


class TurnEagerEnd(TurnReading):
    """The pause's first reading at or above ``eager_eot_threshold``; at most one per pause.

    When the same reading ends the turn, ``turn.ended`` follows it.
    """

    type: Literal["turn.eager_end"] = Field(description="the event type")


class TurnResumed(TurnEvent):
    """The caller spoke again after an eager end; ``audio_window_end`` is the renewed onset."""

    type: Literal["turn.resumed"] = Field(description="the event type")


class TurnEnded(TurnEvent):
    """The turn closed; ``audio_window_end`` is where it closed."""

    type: Literal["turn.ended"] = Field(description="the event type")
    reason: str = Field(
        description="'model' at eot_threshold, 'timeout' after eot_timeout_ms of silence, "
        "'client' after end_turn()"
    )
    probability: float | None = Field(
        default=None, description="the pause's last reading; None when the pause was not read"
    )
    silence: float | None = Field(
        default=None, description="seconds of silence since the caller's last speech"
    )


class TurnsUsage(BaseModel):
    """What a turns connection was billed for."""

    invocations: int = Field(description="model invocations: the first reading of each pause")


class SessionClosed(BaseModel):
    """The connection's session ended after the client's ``session.close``."""

    type: Literal["session.closed"] = Field(description="the event type")
    usage: TurnsUsage = Field(description="the usage over this connection")


class TurnsErrorDetail(BaseModel):
    """The body of a server ``error`` event."""

    code: str | None = Field(default=None, description="machine-readable error code")
    message: str | None = Field(default=None, description="human-readable error message")


class TurnsError(BaseModel):
    """The server reported an error."""

    type: Literal["error"] = Field(description="the event type")
    error: TurnsErrorDetail = Field(
        default_factory=TurnsErrorDetail, description="the error's code and message"
    )

    @property
    def description(self) -> str:
        """The error's message, else its code, else a generic description."""
        return self.error.message or self.error.code or "the turns session reported an error"


TurnsServerEvent = Annotated[
    TurnStarted | TurnUpdate | TurnEagerEnd | TurnResumed | TurnEnded | SessionClosed | TurnsError,
    Field(discriminator="type"),
]

_EVENT_TYPES = frozenset(
    {
        "turn.started",
        "turn.update",
        "turn.eager_end",
        "turn.resumed",
        "turn.ended",
        "session.closed",
        "error",
    }
)
_ADAPTER: TypeAdapter[TurnsServerEvent] = TypeAdapter(TurnsServerEvent)

EVENT_HANDLERS = {
    "turn.started": "on_turn_started",
    "turn.update": "on_turn_update",
    "turn.eager_end": "on_eager_end",
    "turn.resumed": "on_turn_resumed",
    "turn.ended": "on_turn_ended",
    "session.closed": "on_session_closed",
}


def parse_event(raw: dict[str, Any]) -> TurnsServerEvent | None:
    """Parse one server event.

    Args:
        raw: The decoded JSON event.

    Returns:
        The typed event, or None for an event type this client does not read.

    Raises:
        ValidationError: A known event type is missing a field or has one of the wrong type.
    """
    if raw.get("type") not in _EVENT_TYPES:
        return None
    return _ADAPTER.validate_python(raw)


__all__ = [
    "EVENT_HANDLERS",
    "SessionClosed",
    "TurnEagerEnd",
    "TurnEnded",
    "TurnEvent",
    "TurnReading",
    "TurnResumed",
    "TurnStarted",
    "TurnUpdate",
    "TurnsError",
    "TurnsErrorDetail",
    "TurnsServerEvent",
    "TurnsUsage",
    "parse_event",
]
