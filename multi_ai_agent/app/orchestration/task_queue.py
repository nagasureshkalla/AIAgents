"""Task store and priority task queue with a bounded worker pool."""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import time
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path

from app.core.exceptions import TaskNotFoundError
from app.schemas.common import TaskStatus
from app.schemas.task import TaskRecord, TaskRequest
from app.utils.files import atomic_write_json

logger = logging.getLogger(__name__)


class TaskStore:
    """In-memory task records with a JSON snapshot per task under data/tasks/."""

    def __init__(self, tasks_dir: Path | None = None, max_tasks: int = 5000) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._done: dict[str, asyncio.Event] = {}
        self._tasks_dir = tasks_dir
        self._max_tasks = max_tasks

    @staticmethod
    def new_id() -> str:
        return f"task_{uuid.uuid4().hex[:12]}"

    def create(self, request: TaskRequest) -> TaskRecord:
        record = TaskRecord(task_id=self.new_id(), request=request)
        self._tasks[record.task_id] = record
        self._done[record.task_id] = asyncio.Event()
        self._prune()
        self.save(record)
        return record

    def get(self, task_id: str) -> TaskRecord:
        record = self._tasks.get(task_id)
        if record is None:
            raise TaskNotFoundError(f"Task '{task_id}' not found")
        return record

    def list(self, status: TaskStatus | None = None, limit: int = 100) -> list[TaskRecord]:
        records = [r for r in self._tasks.values() if status is None or r.status == status]
        return sorted(records, key=lambda r: r.created_at, reverse=True)[:limit]

    def save(self, record: TaskRecord) -> None:
        if self._tasks_dir is None:
            return
        try:
            atomic_write_json(self._tasks_dir / f"{record.task_id}.json", record.model_dump(mode="json"))
        except OSError:
            logger.warning("could not persist task %s", record.task_id, exc_info=True)

    def mark_done(self, task_id: str) -> None:
        event = self._done.get(task_id)
        if event is not None:
            event.set()

    async def wait(self, task_id: str, timeout: float | None = None) -> TaskRecord:
        record = self.get(task_id)
        event = self._done.get(task_id)
        if event is not None and not record.status.is_terminal:
            await asyncio.wait_for(event.wait(), timeout)
        return self.get(task_id)

    def counts(self) -> dict[str, int]:
        counts = {s.value: 0 for s in TaskStatus}
        for record in self._tasks.values():
            counts[record.status.value] += 1
        return counts

    def _prune(self) -> None:
        if len(self._tasks) <= self._max_tasks:
            return
        finished = sorted((r for r in self._tasks.values() if r.status.is_terminal), key=lambda r: r.created_at)
        for record in finished[: len(self._tasks) - self._max_tasks]:
            self._tasks.pop(record.task_id, None)
            self._done.pop(record.task_id, None)


TaskRunner = Callable[[TaskRecord], Awaitable[object]]


class TaskQueue:
    """Priority queue (CRITICAL first, FIFO within a priority) drained by `workers` concurrent workers."""

    def __init__(self, store: TaskStore, runner: TaskRunner, workers: int, on_cancel: Callable[[TaskRecord], None]) -> None:
        self._store = store
        self._runner = runner
        self._workers_count = workers
        self._on_cancel = on_cancel
        self._queue: asyncio.PriorityQueue[tuple[int, int, str]] = asyncio.PriorityQueue()
        self._seq = itertools.count()
        self._workers: list[asyncio.Task] = []
        self._running: dict[str, asyncio.Task] = {}

    @property
    def running(self) -> int:
        return len(self._running)

    @property
    def queued(self) -> int:
        return self._queue.qsize()

    async def start(self) -> None:
        for i in range(self._workers_count):
            self._workers.append(asyncio.create_task(self._worker(), name=f"task-worker-{i}"))

    async def stop(self) -> None:
        for task in list(self._running.values()):
            task.cancel()
        for worker in self._workers:
            worker.cancel()
        for worker in self._workers:
            with contextlib.suppress(asyncio.CancelledError):
                await worker
        self._workers.clear()

    def submit(self, record: TaskRecord) -> None:
        self._queue.put_nowait((record.request.priority.rank, next(self._seq), record.task_id))

    def cancel(self, task_id: str) -> bool:
        record = self._store.get(task_id)
        if record.status.is_terminal:
            return False
        running = self._running.get(task_id)
        if running is not None:
            running.cancel()  # the runner records the cancellation and emits the event
            return True
        record.status = TaskStatus.CANCELLED
        record.error = "Task cancelled before it started"
        record.completed_at = time.time()
        self._store.save(record)
        self._store.mark_done(task_id)
        self._on_cancel(record)
        return True

    async def _worker(self) -> None:
        while True:
            _, _, task_id = await self._queue.get()
            try:
                try:
                    record = self._store.get(task_id)
                except TaskNotFoundError:
                    continue
                if record.status != TaskStatus.QUEUED:
                    continue
                job = asyncio.create_task(self._runner(record), name=f"run-{task_id}")
                self._running[task_id] = job
                try:
                    # asyncio.wait does not raise when `job` is cancelled, so a cancelled task
                    # does not take the worker down; only cancelling the worker itself propagates.
                    await asyncio.wait({job})
                except asyncio.CancelledError:
                    job.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await job
                    raise
                finally:
                    self._running.pop(task_id, None)
                if not job.cancelled() and job.exception() is not None:
                    logger.error("task %s crashed", task_id, exc_info=job.exception())
            finally:
                self._queue.task_done()
