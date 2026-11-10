"""A Sprag voice agent whose turns are decided by Sprag's turn model.

Rhythm hears, Symphony thinks, Chorus speaks, and ``sprag-turns`` decides when the caller has
finished. Run it with Pipecat's development runner and open the printed URL in a browser:

    SPRAG_API_KEY=... uv run examples/turn_detection.py

The log shows each end-of-turn reading and the moment the model ends a turn. Pass
``--transport daily`` or ``--transport twilio`` to use another transport; see
https://docs.pipecat.ai/server/utilities/runner/guide.
"""

from typing import Any

from dotenv import load_dotenv
from loguru import logger
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import LLMRunFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.transports.daily.transport import DailyParams
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams
from pipecat.turns.user_turn_strategies import ExternalUserTurnStrategies
from pipecat.workers.runner import WorkerRunner

from pipecat_sprag.llm import SpragLLMService
from pipecat_sprag.stt import SpragSTTService
from pipecat_sprag.tts import SpragTTSService
from pipecat_sprag.turns import SpragAgentTextObserver, SpragTurnDetector, SpragTurnParams

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
    turns = SpragTurnDetector(params=SpragTurnParams(eot_threshold=0.6, eot_timeout_ms=3000))
    stt = SpragSTTService()
    llm = SpragLLMService(
        settings=SpragLLMService.Settings(
            system_instruction=(
                "You are a helpful assistant in a voice conversation. Your replies are "
                "spoken aloud, so avoid emojis, lists, and formatting. Keep them brief."
            ),
        ),
    )
    tts = SpragTTSService(settings=SpragTTSService.Settings(voice="wade"))

    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            # The VAD tells Rhythm when to finalize a transcript; the turn model decides the turn.
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=ExternalUserTurnStrategies(),
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

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport: BaseTransport, client: object) -> None:
        logger.info("Client connected")
        context.add_message({"role": "developer", "content": "Introduce yourself briefly."})
        await worker.queue_frames([LLMRunFrame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport: BaseTransport, client: object) -> None:
        logger.info("Client disconnected")
        await runner.cancel()

    await runner.run()


async def bot(runner_args: RunnerArguments) -> None:
    """Entry point for Pipecat's runner and Pipecat Cloud."""
    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport, runner_args)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
