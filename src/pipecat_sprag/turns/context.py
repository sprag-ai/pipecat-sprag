"""An observer that hands the agent's last utterance to a Sprag turn detector or analyzer."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed

from pipecat_sprag.turns.analyzer import SpragTurnAnalyzer
from pipecat_sprag.turns.detector import SpragTurnDetector


class SpragAgentTextObserver(BaseObserver):
    """Sets the text the bot spoke as the turn model's context each time the bot stops speaking.

    Pass it to the pipeline worker's ``observers``. An interrupted reply sets only the text that
    was spoken before the interruption.
    """

    def __init__(self, target: SpragTurnDetector | SpragTurnAnalyzer) -> None:
        """Initialize the observer.

        Args:
            target: The detector or analyzer whose context this observer keeps current.
        """
        super().__init__()
        self._set: Callable[[str], Awaitable[None] | None] = target.set_agent_text
        self._spoken: list[str] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        """Track the bot's spoken text across one reply.

        Args:
            data: The pushed frame and its route.
        """
        if not data.first_push:
            return
        frame = data.frame
        if isinstance(frame, BotStartedSpeakingFrame):
            self._spoken = []
        elif isinstance(frame, TTSTextFrame):
            self._spoken.append(frame.text)
        elif isinstance(frame, BotStoppedSpeakingFrame) and self._spoken:
            text = " ".join(part.strip() for part in self._spoken if part.strip())
            self._spoken = []
            result = self._set(text)
            if inspect.isawaitable(result):
                await result
