"""End-to-end checks against the Sprag API. Run with ``SPRAG_API_KEY=... uv run pytest -m live``.

``SPRAG_TEST_BASE_URL`` points them at another environment (default: production).
"""

import asyncio
import io
import os
import wave
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
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    ProposedUserStoppedSpeakingFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
    TTSTextFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMUsageMetricsData, TurnMetricsData
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.services.llm_service import FunctionCallParams
from pipecat.tests.utils import run_test
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy, TurnAnalyzerUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies

from pipecat_sprag.common import SPRAG_BASE_URL
from pipecat_sprag.llm import SpragLLMService
from pipecat_sprag.stt import SpragSTTService
from pipecat_sprag.tts import SpragTTSService
from pipecat_sprag.turns import SpragTurnAnalyzer, SpragTurnDetector

pytestmark = [
    pytest.mark.live,
    pytest.mark.timeout(120),
    pytest.mark.skipif(not os.environ.get("SPRAG_API_KEY"), reason="needs SPRAG_API_KEY"),
]

BASE_URL = os.environ.get("SPRAG_TEST_BASE_URL", SPRAG_BASE_URL)
PHRASE = "The tide came in slowly over the flat grey sand."
SECOND_PHRASE = "A lighthouse stood at the far end of the bay."
# The longest turn the server buffers before it commits the audio on its own.
SERVER_TURN_SECS = 120


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
    """Chorus speech for ``text`` as 24 kHz PCM."""
    response = httpx.post(
        f"{BASE_URL}/audio/speech",
        headers={"Authorization": f"Bearer {os.environ['SPRAG_API_KEY']}"},
        json={"model": "chorus", "voice": "wade", "input": text, "response_format": "pcm"},
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


def _turn_metrics(down: list[Frame], processor: str) -> list[TurnMetricsData]:
    return [
        d
        for f in down
        if isinstance(f, MetricsFrame)
        for d in f.data
        if isinstance(d, TurnMetricsData) and d.processor.startswith(processor)
    ]


def _transcript(down: list[Frame]) -> str:
    return " ".join(f.text for f in down if isinstance(f, TranscriptionFrame)).lower()


def _heard_phrase(down: list[Frame], up: list[Frame]) -> bool:
    # server VAD may split off a short leading segment first, so wait for the phrase
    return "tide came in slowly" in _transcript(down) or any(isinstance(f, ErrorFrame) for f in up)


@pytest.mark.parametrize("turn_detection", [False, {"type": "server_vad"}])
async def test_stt_transcribes_speech(turn_detection: dict[str, str] | bool) -> None:
    transcription_url = BASE_URL.replace("https://", "wss://") + "/audio/transcriptions"
    stt = SpragSTTService(base_url=transcription_url, turn_detection=turn_detection)  # type: ignore[arg-type]
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
    assert _finalized(down)[-1] is (turn_detection is False)


def _finalized(down: list[Frame]) -> list[bool | None]:
    return [f.finalized for f in down if isinstance(f, TranscriptionFrame)]


def _stt(**kwargs: Any) -> SpragSTTService:
    url = BASE_URL.replace("https://", "wss://") + "/audio/transcriptions"
    return SpragSTTService(base_url=url, **kwargs)


def _errors(up: list[Frame]) -> list[Frame]:
    return [f for f in up if isinstance(f, ErrorFrame)]


def _transcripts(count: int) -> Any:
    return lambda down, up: len(_finalized(down)) >= count or bool(_errors(up))


async def _transcribe(stt: SpragSTTService, frames: list[Any], count: int) -> list[Frame]:
    down, up = await run_until(
        stt,
        frames,
        _transcripts(count),
        params=PipelineParams(audio_in_sample_rate=24000),
        timeout=90,
        between=0.02,
    )
    assert not _errors(up)
    return down


async def test_stt_finalizes_only_the_transcript_after_a_caller_who_resumed() -> None:
    first = _audio_frames(_speech_pcm(PHRASE))
    second = _audio_frames(_speech_pcm(SECOND_PHRASE), trailing_silence_s=1.0)
    down = await _transcribe(
        _stt(),
        [
            VADUserStartedSpeakingFrame(),
            *first,
            VADUserStoppedSpeakingFrame(),
            VADUserStartedSpeakingFrame(),
            *second,
            VADUserStoppedSpeakingFrame(),
        ],
        2,
    )
    assert _finalized(down) == [False, True]
    assert "tide came in slowly" in _transcript(down)
    assert "lighthouse" in _transcript(down)


async def test_stt_commits_a_turn_ended_by_a_vad_and_a_turn_model_once() -> None:
    speech = _audio_frames(_speech_pcm(PHRASE), trailing_silence_s=0.3)
    down = await _transcribe(
        _stt(commit_on_turn_end=True),
        [
            VADUserStartedSpeakingFrame(),
            *speech,
            VADUserStoppedSpeakingFrame(),
            *_audio_frames(b"", trailing_silence_s=0.04),
            ProposedUserStoppedSpeakingFrame(),
            *_audio_frames(b"", trailing_silence_s=2.0),
        ],
        1,
    )
    assert _finalized(down) == [True]


async def test_stt_keeps_its_session_through_a_turn_too_short_to_commit() -> None:
    speech = _audio_frames(_speech_pcm(PHRASE), trailing_silence_s=1.0)
    stt = _stt()
    down = await _transcribe(
        stt,
        [
            VADUserStartedSpeakingFrame(),
            *_audio_frames(b"", trailing_silence_s=0.04),
            VADUserStoppedSpeakingFrame(),
            *_audio_frames(b"", trailing_silence_s=1.0),
            VADUserStartedSpeakingFrame(),
            *speech,
            VADUserStoppedSpeakingFrame(),
        ],
        1,
    )
    assert _finalized(down) == [True]
    assert "tide came in slowly" in _transcript(down)


async def test_stt_finalizes_again_after_it_reconnects() -> None:
    speech = _audio_frames(_speech_pcm(PHRASE), trailing_silence_s=1.0)
    turn = [VADUserStartedSpeakingFrame(), *speech, VADUserStoppedSpeakingFrame()]
    stt = _stt()
    sessions: list[bool] = []

    @stt.event_handler("on_connected")
    async def on_connected(service: SpragSTTService) -> None:
        sessions.append(True)

    def drop(down: list[Frame], up: list[Frame]) -> bool:
        asyncio.get_running_loop().create_task(stt._websocket.close(code=1011))
        return True

    down, up = await run_until(
        stt,
        [
            *turn,
            _transcripts(1),
            drop,
            lambda down, up: len(sessions) == 2 and stt._session_ready,
            *turn,
        ],
        _transcripts(2),
        params=PipelineParams(audio_in_sample_rate=24000),
        timeout=90,
        between=0.02,
    )
    assert len(sessions) == 2
    assert _finalized(down) == [True, True]


@pytest.mark.timeout(300)
async def test_stt_finalizes_only_the_end_of_a_turn_longer_than_the_server_holds() -> None:
    phrase = _speech_pcm(PHRASE) + b"\x00\x00" * 12000
    spoken = phrase * (int(SERVER_TURN_SECS * 48000 / len(phrase)) + 2)
    down, up = await run_until(
        _stt(),
        [
            VADUserStartedSpeakingFrame(),
            *_audio_frames(spoken),
            VADUserStoppedSpeakingFrame(),
            *_audio_frames(b"", trailing_silence_s=1.0),
        ],
        _transcripts(2),
        params=PipelineParams(audio_in_sample_rate=24000),
        timeout=280,
        between=0.002,
    )
    assert not _errors(up)
    assert _finalized(down) == [False, True]


@pytest.mark.parametrize("rate", [24000, 16000])
async def test_tts_speaks_a_multi_sentence_turn(rate: int) -> None:
    speech_url = BASE_URL.replace("https://", "wss://") + "/audio/speech"
    tts = SpragTTSService(url=speech_url)
    turn = [
        LLMFullResponseStartFrame(),
        LLMTextFrame("Thanks for calling today, it is good to hear from you. "),
        LLMTextFrame("Your package shipped this morning and should arrive on Friday."),
        LLMFullResponseEndFrame(),
    ]
    down, up = await run_until(
        tts,
        turn,
        lambda down, up: (
            any(isinstance(f, TTSStoppedFrame) for f in down)
            or any(isinstance(f, ErrorFrame) for f in up)
        ),
        params=PipelineParams(audio_out_sample_rate=rate),
        timeout=30,
        play_audio=True,
    )
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    frames = [f for f in down if isinstance(f, TTSAudioRawFrame)]
    assert {f.sample_rate for f in frames} == {rate}
    pcm = b"".join(f.audio for f in frames)
    assert len(pcm) / 2 / rate > 3

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm)
    response = httpx.post(
        f"{BASE_URL}/audio/transcriptions",
        headers={"Authorization": f"Bearer {os.environ['SPRAG_API_KEY']}"},
        data={"model": "rhythm"},
        files={"file": ("reply.wav", buffer.getvalue(), "audio/wav")},
        timeout=60,
    )
    response.raise_for_status()
    heard = response.json()["text"].lower()
    assert "thanks for calling" in heard and "arrive on friday" in heard


async def test_voice_agent_answers_a_spoken_question() -> None:
    transcription_url = BASE_URL.replace("https://", "wss://") + "/audio/transcriptions"
    context = LLMContext(
        messages=[{"role": "system", "content": "Answer in one short sentence."}],
    )
    user_aggregator, _ = LLMContextAggregatorPair(context)
    agent = Pipeline(
        [
            SpragSTTService(base_url=transcription_url),
            user_aggregator,
            SpragLLMService(base_url=BASE_URL),
            SpragTTSService(url=BASE_URL.replace("https://", "wss://") + "/audio/speech"),
        ]
    )
    question = _audio_frames(_speech_pcm("What is the tallest mountain in the world?"), 0.5)
    down, up = await run_until(
        agent,
        [VADUserStartedSpeakingFrame(), *question, VADUserStoppedSpeakingFrame()],
        lambda down, up: (
            any(isinstance(f, TTSStoppedFrame) for f in down)
            or any(isinstance(f, ErrorFrame) for f in up)
        ),
        params=PipelineParams(audio_in_sample_rate=24000),
        timeout=45,
        between=0.02,
        play_audio=True,
    )
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    reply = " ".join(f.text for f in down if isinstance(f, TTSTextFrame)).lower()
    assert "everest" in reply
    assert sum(len(f.audio) for f in down if isinstance(f, TTSAudioRawFrame)) > 24000 * 2


async def test_turn_detector_proposes_the_end_of_a_spoken_turn() -> None:
    turns_url = BASE_URL.replace("https://", "wss://") + "/turns"
    detector = SpragTurnDetector(url=turns_url)
    await detector.set_agent_text("What can I help you with today?")
    speech = _audio_frames(_speech_pcm(PHRASE))
    silence = _audio_frames(b"", trailing_silence_s=4.0)
    down, up = await run_until(
        detector,
        [
            lambda down, up: detector._session.connected,
            VADUserStartedSpeakingFrame(),
            *speech,
            VADUserStoppedSpeakingFrame(),
            *silence,
        ],
        lambda down, up: (
            bool(_turn_metrics(down, "SpragTurnDetector"))
            or any(isinstance(f, ErrorFrame) for f in up)
        ),
        params=PipelineParams(audio_in_sample_rate=24000, enable_metrics=True),
        between=0.02,
        timeout=60,
    )
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert sum(isinstance(f, ProposedUserStoppedSpeakingFrame) for f in down) == 1
    verdicts = _turn_metrics(down, "SpragTurnDetector")
    assert len(verdicts) == 1 and verdicts[0].e2e_processing_time_ms > 0


async def test_turn_analyzer_ends_a_spoken_turn() -> None:
    transcription_url = BASE_URL.replace("https://", "wss://") + "/audio/transcriptions"
    analyzer = SpragTurnAnalyzer(url=BASE_URL.replace("https://", "wss://") + "/turns")
    await analyzer.set_agent_text("What can I help you with today?")
    context = LLMContext()
    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                stop=[TurnAnalyzerUserTurnStopStrategy(turn_analyzer=analyzer)]
            )
        ),
    )
    lead_in = _audio_frames(b"", trailing_silence_s=0.5)
    speech = _audio_frames(_speech_pcm(PHRASE))
    silence = _audio_frames(b"", trailing_silence_s=4.0)
    down, up = await run_until(
        Pipeline([SpragSTTService(base_url=transcription_url), user_aggregator]),
        [
            *lead_in,
            lambda down, up: analyzer._session.connected,
            VADUserStartedSpeakingFrame(),
            *speech,
            VADUserStoppedSpeakingFrame(),
            *silence,
        ],
        lambda down, up: (
            any(isinstance(f, LLMContextFrame) for f in down)
            or any(isinstance(f, ErrorFrame) for f in up)
        ),
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.02,
        timeout=60,
    )
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert sum(isinstance(f, UserStoppedSpeakingFrame) for f in down) == 1
    verdicts = _turn_metrics(down, "SpragTurnAnalyzer")
    assert len(verdicts) == 1 and verdicts[0].is_complete
    assert verdicts[0].e2e_processing_time_ms > 0
    heard = " ".join(str(m.get("content")) for m in context.get_messages()).lower()
    assert "tide came in slowly" in heard


async def test_turn_detector_ends_a_turn_with_its_whole_transcript_without_a_vad() -> None:
    turns = SpragTurnDetector(url=BASE_URL.replace("https://", "wss://") + "/turns")
    stt = SpragSTTService(
        base_url=BASE_URL.replace("https://", "wss://") + "/audio/transcriptions",
        commit_on_turn_end=True,
    )
    context = LLMContext()
    user_aggregator, _ = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[ExternalUserTurnStopStrategy(timeout=0.1)],
            )
        ),
    )
    speech = _audio_frames(_speech_pcm(PHRASE))
    silence = _audio_frames(b"", trailing_silence_s=4.0)
    down, up = await run_until(
        Pipeline([turns, stt, user_aggregator]),
        [lambda down, up: turns._session.connected, *speech, *silence],
        lambda down, up: (
            any(isinstance(f, LLMContextFrame) for f in down)
            or any(isinstance(f, ErrorFrame) for f in up)
        ),
        params=PipelineParams(audio_in_sample_rate=24000),
        between=0.02,
        timeout=60,
    )
    assert not [f for f in up if isinstance(f, ErrorFrame)]
    assert sum(isinstance(f, UserStoppedSpeakingFrame) for f in down) == 1
    user = [m for m in context.get_messages() if m.get("role") == "user"]
    assert len(user) == 1 and "tide came in slowly" in str(user[0]["content"]).lower()
