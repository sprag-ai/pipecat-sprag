"""Sprag's Symphony model as a Pipecat LLM service."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import PromptTokensDetails
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_sprag.common import ATTRIBUTION_HEADERS, SPRAG_BASE_URL, resolve_api_key


@dataclass
class SpragLLMSettings(OpenAILLMSettings):
    """Settings for SpragLLMService."""


class SpragLLMService(OpenAILLMService):
    """Sprag's Symphony model through Sprag's OpenAI-compatible chat completions API.

    Sprag reports token usage per modality (``usage.input.text.tokens`` and so on)
    rather than as ``prompt_tokens`` and ``completion_tokens``. This service maps it
    onto the fields Pipecat reads, so usage metrics and function calls work.
    """

    Settings = SpragLLMSettings
    _settings: SpragLLMSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SPRAG_BASE_URL,
        settings: SpragLLMSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sprag LLM service.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment
                variable.
            base_url: The Sprag API base URL.
            settings: Runtime-updatable settings. The model defaults to ``symphony``.
            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="symphony")
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            default_headers=ATTRIBUTION_HEADERS,
            settings=default_settings,
            **kwargs,
        )

    async def get_chat_completions(  # type: ignore[override]
        self, context: LLMContext
    ) -> _OpenAIUsageStream:
        """Stream chat completions, with Sprag's usage mapped onto OpenAI's fields.

        Args:
            context: The context to complete.

        Returns:
            The chunk stream.
        """
        return _OpenAIUsageStream(await super().get_chat_completions(context))


class _OpenAIUsageStream:
    """Wraps a chunk stream so each chunk's usage carries OpenAI's token fields."""

    def __init__(self, stream: AsyncStream[ChatCompletionChunk]) -> None:
        self._stream = stream

    def __aiter__(self) -> AsyncIterator[ChatCompletionChunk]:
        return self._iterate()

    async def _iterate(self) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in self._stream:
            yield _with_openai_usage(chunk)

    async def close(self) -> None:
        await self._stream.close()


def _with_openai_usage(chunk: ChatCompletionChunk) -> ChatCompletionChunk:
    usage = chunk.usage
    if usage is None or usage.prompt_tokens is not None:
        return chunk
    extra = usage.model_extra or {}
    inputs: dict[str, Any] = extra.get("input") or {}
    outputs: dict[str, Any] = extra.get("output") or {}
    prompt = _tokens(inputs)
    completion = _tokens(outputs)
    usage.prompt_tokens = prompt
    usage.completion_tokens = completion
    usage.total_tokens = prompt + completion
    cached = (inputs.get("text") or {}).get("cached_tokens")
    if cached:
        usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=cached)
    return chunk


def _tokens(modalities: dict[str, Any]) -> int:
    return sum(value.get("tokens") or 0 for value in modalities.values() if isinstance(value, dict))
