"""A Sprag voice agent behind a Sprag-branded browser client, for recording a demo.

Rhythm transcribes, Symphony answers, Chorus speaks, and Sprag's turn model decides when the
caller has finished:

    SPRAG_API_KEY=... uv run examples/demo.py

Then open http://localhost:7860.
"""

from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frameworks.rtvi import RTVIProcessor
from pipecat.runner.run import app, main
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop import ExternalUserTurnStopStrategy
from pipecat.turns.user_turn_strategies import UserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_sprag.llm import SpragLLMService
from pipecat_sprag.stt import SpragSTTService
from pipecat_sprag.tts import SpragTTSService
from pipecat_sprag.turns import SpragAgentTextObserver, SpragTurnDetector

load_dotenv(override=True)

app.mount("/sprag", StaticFiles(directory=Path(__file__).parent / "demo", html=True), name="sprag")


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Send the browser to the Sprag client."""
    return RedirectResponse(url="/sprag/")


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point for Pipecat's runner."""
    transport = await create_transport(
        runner_args,
        {"webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True)},
    )
    turns = SpragTurnDetector()
    stt = SpragSTTService(commit_on_turn_end=True)
    llm = SpragLLMService(
        settings=SpragLLMService.Settings(
            system_instruction=(
                "Your name is Sprag, and you were built by Sprag. You are a helpful assistant "
                "in a voice conversation. Your replies are spoken aloud, so avoid emojis, lists, "
                "and formatting. Answer in two or three sentences."
            ),
        ),
    )
    tts = SpragTTSService(settings=SpragTTSService.Settings(voice="dominic-conversational"))

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            user_turn_strategies=UserTurnStrategies(
                start=[ExternalUserTurnStartStrategy()],
                stop=[ExternalUserTurnStopStrategy(timeout=0.1)],
            ),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            turns,
            stt,
            user_aggregator,
            llm,
            tts,
            transport.output(),
            assistant_aggregator,
        ]
    )
    worker = PipelineWorker(
        pipeline,
        params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
        idle_timeout_secs=runner_args.pipeline_idle_timeout_secs,
        observers=[SpragAgentTextObserver(turns)],
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

    @turns.event_handler("on_turn_update")
    async def on_turn_update(detector: SpragTurnDetector, event: dict[str, Any]) -> None:
        logger.debug(f"p_eot {event['p_eot']:.2f} after {event['silence_ms']} ms of silence")

    @turns.event_handler("on_turn_ended")
    async def on_turn_ended(detector: SpragTurnDetector, event: dict[str, Any]) -> None:
        logger.info(f"Turn ended: {event}")

    @worker.rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi: RTVIProcessor) -> None:
        logger.info("Client ready")
        await worker.queue_frames(
            [TTSSpeakFrame("Hi, I'm a voice assistant running on Sprag. Ask me anything.")]
        )

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: object) -> None:
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


if __name__ == "__main__":
    main()
