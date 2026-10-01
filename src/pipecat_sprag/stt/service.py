"""Sprag's Rhythm model as a Pipecat speech-to-text service."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pipecat.services.openai.stt import OpenAIRealtimeSTTService, OpenAIRealtimeSTTSettings

from pipecat_sprag.common import SPRAG_REALTIME_URL, resolve_api_key


@dataclass
class SpragSTTSettings(OpenAIRealtimeSTTSettings):
    """Settings for SpragSTTService."""


class SpragSTTService(OpenAIRealtimeSTTService):
    """Streaming transcription with Sprag's Rhythm model over Sprag's realtime API.

    By default the pipeline's own VAD decides when a turn ends. Pass
    ``turn_detection={"type": "server_vad"}`` to have Sprag detect turns instead.

    Unlike the LLM and TTS services, it sends no ``X-Sprag-Integration`` header: Pipecat's
    realtime STT connects with a fixed set of headers.
    """

    Settings = SpragSTTSettings
    _settings: SpragSTTSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SPRAG_REALTIME_URL,
        turn_detection: dict[str, Any] | Literal[False] | None = False,
        settings: SpragSTTSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sprag STT service.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment
                variable.
            base_url: The Sprag realtime WebSocket URL.
            turn_detection: ``False`` (the default) leaves turn detection to the
                pipeline's VAD. A dict such as ``{"type": "server_vad"}`` enables
                Sprag's server-side VAD.
            settings: Runtime-updatable settings. The model defaults to ``rhythm``.
            **kwargs: Additional arguments passed to OpenAIRealtimeSTTService.
        """
        # Rhythm takes no language parameter
        default_settings = self.Settings(model="rhythm", language=None)
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            turn_detection=turn_detection,
            settings=default_settings,
            **kwargs,
        )
