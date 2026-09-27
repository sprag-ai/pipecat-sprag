"""End-to-end checks against the Sprag API. Run with ``SPRAG_API_KEY=... uv run pytest -m live``.

``SPRAG_TEST_BASE_URL`` points them at another environment (default: production).
"""

import os
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import ErrorFrame, LLMContextFrame, MetricsFrame
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import run_test

from pipecat_sprag.common import SPRAG_BASE_URL
from pipecat_sprag.llm import SpragLLMService

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("SPRAG_API_KEY"), reason="needs SPRAG_API_KEY"),
]

BASE_URL = os.environ.get("SPRAG_TEST_BASE_URL", SPRAG_BASE_URL)


async def test_llm_calls_a_tool_and_reports_usage() -> None:
    llm = SpragLLMService(base_url=BASE_URL)
    calls: list[dict[str, Any]] = []

    async def get_weather(params: FunctionCallParams) -> None:
        calls.append(dict(params.arguments))
        await params.result_callback({"conditions": "sunny"})

    llm.register_function("get_weather", get_weather)
    weather = FunctionSchema(
        name="get_weather",
        description="Look up the current weather for a city.",
        properties={"city": {"type": "string"}},
        required=["city"],
    )
    context = LLMContext(
        messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
        tools=ToolsSchema(standard_tools=[weather]),
    )
    down, up = await run_test(
        llm,
        frames_to_send=[LLMContextFrame(context=context)],
        pipeline_params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
    )

    assert not [f for f in (*down, *up) if isinstance(f, ErrorFrame)]
    assert calls and calls[0]["city"].lower() == "paris"
    usage = [
        d.value
        for f in down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, LLMUsageMetricsData)
    ]
    assert usage and usage[0].prompt_tokens > 0 and usage[0].completion_tokens > 0
