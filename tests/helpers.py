import asyncio
from collections.abc import Callable, Sequence

from pipecat.frames.frames import EndFrame, Frame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.worker import PipelineParams, PipelineWorker
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.tests.utils import QueuedFrameProcessor
from pipecat.workers.runner import WorkerRunner


async def run_until(
    processor: FrameProcessor,
    frames: Sequence[Frame],
    done: Callable[[list[Frame], list[Frame]], bool],
    *,
    params: PipelineParams | None = None,
    timeout: float = 10.0,
    between: float = 0.0,
) -> tuple[list[Frame], list[Frame]]:
    """Run ``processor`` in a pipeline, send ``frames``, and collect output until ``done``.

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

    async def drive() -> None:
        await asyncio.wait_for(started.wait(), timeout=5)
        for frame in frames:
            await worker.queue_frame(frame)
            if between:
                await asyncio.sleep(between)
        while not done(down, up):
            while not up_queue.empty():
                up.append(up_queue.get_nowait())
            try:
                down.append(await asyncio.wait_for(down_queue.get(), timeout=0.05))
            except TimeoutError:
                pass
        await worker.queue_frame(EndFrame())

    runner = WorkerRunner()
    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    try:
        await asyncio.wait_for(drive(), timeout=timeout)
        await asyncio.wait_for(running, timeout=5)
    finally:
        if not running.done():
            await worker.cancel()
            await asyncio.gather(running, return_exceptions=True)
    while not up_queue.empty():
        up.append(up_queue.get_nowait())
    return down, up
