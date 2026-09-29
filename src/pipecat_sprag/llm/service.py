"""Sprag's Symphony model as a Pipecat LLM service."""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass
from typing import Any

from openai import AsyncStream
from openai.types.chat import ChatCompletionChunk
from openai.types.completion_usage import CompletionTokensDetails, PromptTokensDetails
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.openai.base_llm import OpenAILLMSettings
from pipecat.services.openai.llm import OpenAILLMService

from pipecat_sprag.common import ATTRIBUTION_HEADERS, SPRAG_BASE_URL, resolve_api_key


@dataclass
class SpragLLMSettings(OpenAILLMSettings):
    """Settings for SpragLLMService."""


class SpragLLMService(OpenAILLMService):
    """Sprag's Symphony model through Sprag's OpenAI-compatible chat completions API.

    Sprag reports usage per billing axis (``usage.input.text.tokens``,
    ``usage.input.audio.duration`` and so on) rather than as ``prompt_tokens`` and
    ``completion_tokens``. This service reports the text tokens in the fields Pipecat
    reads, so usage metrics and function calls work. Media is billed on its own axes,
    which Pipecat's token metrics have no place for; Sprag's full usage stays on the
    chunk for anyone who needs it.
    """

    Settings = SpragLLMSettings
    _settings: SpragLLMSettings

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = SPRAG_BASE_URL,
        default_headers: Mapping[str, str] | None = None,
        settings: SpragLLMSettings | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the Sprag LLM service.

        Args:
            api_key: Your Sprag API key. Defaults to the ``SPRAG_API_KEY`` environment
                variable.
            base_url: The Sprag API base URL.
            default_headers: Extra headers sent with every request.
            settings: Runtime-updatable settings. The model defaults to ``symphony``.
            **kwargs: Additional arguments passed to OpenAILLMService.
        """
        default_settings = self.Settings(model="symphony")
        if settings is not None:
            default_settings.apply_update(settings)
        super().__init__(
            api_key=resolve_api_key(api_key),
            base_url=base_url,
            default_headers={**ATTRIBUTION_HEADERS, **(default_headers or {})},
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
    text_in: dict[str, Any] = (extra.get("input") or {}).get("text") or {}
    text_out: dict[str, Any] = (extra.get("output") or {}).get("text") or {}
    usage.prompt_tokens = text_in.get("tokens") or 0
    usage.completion_tokens = text_out.get("tokens") or 0
    usage.total_tokens = usage.prompt_tokens + usage.completion_tokens
    if text_in.get("cached_tokens"):
        usage.prompt_tokens_details = PromptTokensDetails(cached_tokens=text_in["cached_tokens"])
    if text_out.get("reasoning_tokens"):
        usage.completion_tokens_details = CompletionTokensDetails(
            reasoning_tokens=text_out["reasoning_tokens"]
        )
    return chunk
