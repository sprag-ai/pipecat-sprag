"""A voice agent that runs entirely on Sprag: Rhythm hears, Symphony thinks, Chorus speaks.

Run it with Pipecat's development runner and open the printed URL in a browser:

    SPRAG_API_KEY=... uv run examples/voice_agent.py

Pass ``--transport daily`` or ``--transport twilio`` to use another transport; see
https://docs.pipecat.ai/server/utilities/runner/guide.
"""

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
from pipecat.workers.runner import WorkerRunner

from pipecat_sprag.llm import SpragLLMService
from pipecat_sprag.stt import SpragSTTService
from pipecat_sprag.tts import SpragTTSService

load_dotenv(override=True)

transport_params = {
    "daily": lambda: DailyParams(audio_in_enabled=True, audio_out_enabled=True),
    "twilio": lambda: FastAPIWebsocketParams(audio_in_enabled=True, audio_out_enabled=True),
    "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
}


async def run_bot(transport: BaseTransport, runner_args: RunnerArguments) -> None:
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
        user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer()),
    )
    pipeline = Pipeline(
        [
            transport.input(),
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
    )
    runner = WorkerRunner(handle_sigint=runner_args.handle_sigint)
    await runner.add_workers(worker)

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
