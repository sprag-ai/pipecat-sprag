"""End-to-end checks against the Sprag API. Run with ``SPRAG_API_KEY=... uv run pytest -m live``.

``SPRAG_TEST_BASE_URL`` points them at another environment (default: production).
"""

import os
from typing import Any

import httpx
import pytest
from helpers import run_until
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    InputAudioRawFrame,
    LLMContextFrame,
    MetricsFrame,
    TranscriptionFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import run_test

from pipecat_sprag.common import SPRAG_BASE_URL
from pipecat_sprag.llm import SpragLLMService
from pipecat_sprag.stt import SpragSTTService

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.environ.get("SPRAG_API_KEY"), reason="needs SPRAG_API_KEY"),
]

BASE_URL = os.environ.get("SPRAG_TEST_BASE_URL", SPRAG_BASE_URL)
PHRASE = "The tide came in slowly over the flat grey sand."


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


def _speech_pcm(text: str) -> bytes:
    """Synthesize ``text`` with Chorus over REST, as 24 kHz PCM, to feed the STT."""
    response = httpx.post(
        f"{BASE_URL}/audio/speech",
        headers={"Authorization": f"Bearer {os.environ['SPRAG_API_KEY']}"},
        json={"model": "chorus-clone", "voice": "wade", "input": text, "response_format": "pcm"},
        timeout=60,
    )
    response.raise_for_status()
    return response.content


def _audio_frames(pcm: bytes, trailing_silence_s: float = 0.0) -> list[Frame]:
    pcm += b"\x00\x00" * int(24000 * trailing_silence_s)
    step = 480 * 2  # 20 ms
    return [
        InputAudioRawFrame(audio=pcm[i : i + step], sample_rate=24000, num_channels=1)
        for i in range(0, len(pcm), step)
    ]


def _transcript(down: list[Frame]) -> str:
    return " ".join(f.text for f in down if isinstance(f, TranscriptionFrame)).lower()


def _heard_phrase(down: list[Frame], up: list[Frame]) -> bool:
    # server VAD may split off a short leading segment first, so wait for the phrase
    return "tide came in slowly" in _transcript(down) or any(isinstance(f, ErrorFrame) for f in up)


@pytest.mark.parametrize("turn_detection", [False, {"type": "server_vad"}])
async def test_stt_transcribes_speech(turn_detection: dict[str, str] | bool) -> None:
    realtime_url = BASE_URL.replace("https://", "wss://") + "/realtime"
    stt = SpragSTTService(base_url=realtime_url, turn_detection=turn_detection)  # type: ignore[arg-type]
    audio = _audio_frames(_speech_pcm(PHRASE), trailing_silence_s=1.0)
    frames: list[Frame] = (
        [VADUserStartedSpeakingFrame(), *audio, VADUserStoppedSpeakingFrame()]
        if turn_detection is False
        else audio
    )
    down, up = await run_until(
        stt,
        frames,
        _heard_phrase,
        params=PipelineParams(audio_in_sample_rate=24000),
        timeout=30,
        between=0.02,
    )
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert "tide came in slowly" in _transcript(down)
