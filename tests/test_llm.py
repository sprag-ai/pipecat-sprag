import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, LLMTextFrame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import run_test

from pipecat_sprag.common import ATTRIBUTION_HEADERS
from pipecat_sprag.llm import SpragLLMService

# Sprag reports usage per modality, not as prompt_tokens/completion_tokens
SPRAG_USAGE = {
    "input": {
        "text": {"tokens": 12, "characters": 0, "cached_tokens": 4, "reasoning_tokens": 0},
        "audio": {"duration": "2.5"},
        "image": {"count": 1, "megapixels": "1.0", "tiles": 4, "tokens": 700},
        "video": {"duration": "0", "frames": 0, "tokens": 0},
    },
    "output": {
        "text": {"tokens": 5, "characters": 0, "cached_tokens": 0, "reasoning_tokens": 2},
        "audio": {"duration": "0"},
        "tools": {"call_count": 1, "tokens": 9},
    },
}


def _chunk(**fields: Any) -> dict[str, Any]:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "symphony",
        "choices": [],
        **fields,
    }


class _ChatServer:
    """Streams a Symphony reply: text, then a tool call on the first request, then usage."""

    def __init__(self) -> None:
        self.requests: list[tuple[web.Request, dict[str, Any]]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        self.requests.append((request, body))
        chunks = [_chunk(choices=[{"index": 0, "delta": {"role": "assistant", "content": "Hi"}}])]
        if len(self.requests) == 1:
            call = {"index": 0, "id": "call_1", "type": "function"}
            call["function"] = {"name": "get_weather", "arguments": '{"city": "Paris"}'}
            chunks.append(_chunk(choices=[{"index": 0, "delta": {"tool_calls": [call]}}]))
        chunks.append(_chunk(usage=SPRAG_USAGE))
        text = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return web.Response(text=text, content_type="text/event-stream")


@pytest.fixture
async def chat() -> AsyncIterator[tuple[_ChatServer, str]]:
    server = _ChatServer()
    app = web.Application()
    app.router.add_post("/v1/chat/completions", server.handle)
    test_server = TestServer(app)
    await test_server.start_server()
    yield server, str(test_server.make_url("/v1"))
    await test_server.close()


def _weather_context() -> LLMContext:
    weather = FunctionSchema(
        name="get_weather",
        description="Look up the weather for a city.",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    return LLMContext(
        messages=[{"role": "user", "content": "What's the weather in Paris?"}],
        tools=ToolsSchema(standard_tools=[weather]),
    )


async def test_reply_usage_and_function_call(chat: tuple[_ChatServer, str]) -> None:
    server, base_url = chat
    llm = SpragLLMService(api_key="test-key", base_url=base_url)
    calls: list[dict[str, Any]] = []

    async def get_weather(params: FunctionCallParams) -> None:
        calls.append(dict(params.arguments))
        await params.result_callback({"conditions": "sunny"})

    llm.register_function("get_weather", get_weather)
    down, up = await run_test(
        llm,
        frames_to_send=[LLMContextFrame(context=_weather_context())],
        pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    assert not [f for f in (*down, *up) if isinstance(f, ErrorFrame)]
    assert calls == [{"city": "Paris"}]
    assert "".join(f.text for f in down if isinstance(f, LLMTextFrame)).startswith("Hi")
    usage = [
        d.value
        for f in down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, LLMUsageMetricsData)
    ]
    assert usage, "no usage metrics were reported"
    # text tokens only: image, audio, and tool usage bill on their own axes
    assert usage[0].prompt_tokens == 12
    assert usage[0].completion_tokens == 5
    assert usage[0].total_tokens == 17
    assert usage[0].cache_read_input_tokens == 4
    assert usage[0].reasoning_tokens == 2

    request, body = server.requests[0]
    assert request.headers["Authorization"] == "Bearer test-key"
    assert request.headers["X-Sprag-Integration"] == ATTRIBUTION_HEADERS["X-Sprag-Integration"]
    assert body["model"] == "symphony"
    assert body["stream"] is True


async def test_model_can_be_overridden(chat: tuple[_ChatServer, str]) -> None:
    server, base_url = chat
    llm = SpragLLMService(
        api_key="test-key",
        base_url=base_url,
        settings=SpragLLMService.Settings(model="symphony-notes"),
    )
    context = LLMContext(messages=[{"role": "user", "content": "Hello"}])
    await run_test(llm, frames_to_send=[LLMContextFrame(context=context)])
    assert server.requests[0][1]["model"] == "symphony-notes"


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SPRAG_API_KEY", raising=False)
    with pytest.raises(ValueError, match="SPRAG_API_KEY"):
        SpragLLMService()
