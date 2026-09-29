import asyncio
from collections.abc import Callable, Sequence

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStoppedFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.workers.runner import WorkerRunner


async def run_until(
    processor: FrameProcessor,
    frames: Sequence[Frame | Callable[[list[Frame], list[Frame]], bool]],
    done: Callable[[list[Frame], list[Frame]], bool],
    *,
    params: PipelineParams | None = None,
    timeout: float = 10.0,
    between: float = 0.0,
    play_audio: bool = False,
) -> tuple[list[Frame], list[Frame]]:
    """Run ``processor`` in a pipeline, send ``frames``, and collect output until ``done``.

    An entry in ``frames`` may also be a condition on the output so far; sending pauses
    until it holds, which lets a test interrupt at a chosen moment.

    With ``play_audio``, the helper stands in for an output transport: like one, it
    reports the bot speaking once a turn's audio arrives and stopped when that turn ends,
    which TTS services wait on to resume processing after a turn. A turn without audio
    is never reported as speech.

    Unlike ``run_test``, the pipeline stays up until ``done`` holds, which services that
    answer asynchronously (websocket STT and TTS) need.

    Returns:
        The downstream and upstream frames received.
    """
    down_queue: asyncio.Queue[Frame] = asyncio.Queue()
    up_queue: asyncio.Queue[Frame] = asyncio.Queue()
    sink = QueuedFrameProcessor(
        queue=down_queue, queue_direction=FrameDirection.DOWNSTREAM, ignore_start=True
    )
    source = QueuedFrameProcessor(
        queue=up_queue, queue_direction=FrameDirection.UPSTREAM, ignore_start=True
    )
    worker = PipelineWorker(
        Pipeline([source, processor, sink]),
        params=params or PipelineParams(),
        cancel_on_idle_timeout=False,
    )
    started = asyncio.Event()

    @worker.event_handler("on_pipeline_started")
    async def _started(worker: PipelineWorker, frame: Frame) -> None:
        started.set()

    down: list[Frame] = []
    up: list[Frame] = []
    speaking: list[bool] = []

    async def drive() -> None:
        await asyncio.wait_for(started.wait(), timeout=5)
        for entry in frames:
            if isinstance(entry, Frame):
                await worker.queue_frame(entry)
                if between:
                    await asyncio.sleep(between)
            else:
                await collect_until(entry)
        await collect_until(done)
        await worker.queue_frame(EndFrame())
        # a paused TTS only reaches the EndFrame once playback is reported finished
        await collect_until(lambda down, up: running.done())

    async def collect_until(condition: Callable[[list[Frame], list[Frame]], bool]) -> None:
        while not condition(down, up):
            while not up_queue.empty():
                up.append(up_queue.get_nowait())
            try:
                frame = await asyncio.wait_for(down_queue.get(), timeout=0.05)
            except TimeoutError:
                continue
            down.append(frame)
            if not play_audio:
                continue
            if isinstance(frame, TTSAudioRawFrame) and not speaking:
                speaking.append(True)
                await sink.push_frame(BotStartedSpeakingFrame(), FrameDirection.UPSTREAM)
            elif isinstance(frame, TTSStoppedFrame) and speaking:
                speaking.clear()
                await sink.push_frame(BotStoppedSpeakingFrame(), FrameDirection.UPSTREAM)

    runner = WorkerRunner()
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(drive(), timeout=timeout)
        await running
    finally:
        if not running.done():
            await worker.cancel()
            await asyncio.gather(running, return_exceptions=True)
    while not up_queue.empty():
        up.append(up_queue.get_nowait())
    return down, up
