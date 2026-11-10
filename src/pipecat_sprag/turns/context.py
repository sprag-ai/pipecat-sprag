"""An observer that hands the agent's last utterance to a Sprag turn detector or analyzer."""

from __future__ import annotations

from pipecat.frames.frames import BotStartedSpeakingFrame, BotStoppedSpeakingFrame, TTSTextFrame
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.processors.frame_processor import FrameDirection
from pipecat.transports.base_output import BaseOutputTransport

from pipecat_sprag.turns.analyzer import SpragTurnAnalyzer
from pipecat_sprag.turns.detector import SpragTurnDetector


class SpragAgentTextObserver(BaseObserver):
    """Sets the text the bot spoke as the turn model's context each time the bot stops speaking.

    Pass it to the pipeline worker's ``observers``. Text is collected as the output transport
    plays it, so an interrupted reply sets only the words that were played, and a reply with
    no text clears the context.
    """

    def __init__(self, target: SpragTurnDetector | SpragTurnAnalyzer) -> None:
        """Initialize the observer.

        Args:
            target: The detector or analyzer whose context this observer keeps current.
        """
        super().__init__()
        self._target = target
        self._spoken: list[str] = []

    async def on_push_frame(self, data: FramePushed) -> None:
        """Track the text the output transport plays across one reply.

        Args:
            data: The pushed frame and its route.
        """
        if data.direction is not FrameDirection.DOWNSTREAM:
            return
        if not isinstance(data.source, BaseOutputTransport):
            return
        frame = data.frame
        if isinstance(frame, BotStartedSpeakingFrame):
            self._spoken = []
        elif isinstance(frame, TTSTextFrame):
            self._spoken.append(frame.text)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            text = " ".join(part.strip() for part in self._spoken if part.strip())
            self._spoken = []
            await self._target.set_agent_text(text)
