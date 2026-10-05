"""Resource-aware inference scheduler.

Agents run concurrently as asyncio tasks, but model inference is admitted through per-device
priority pools. With the default `gpu_concurrency: 1` only one model runs on the GPU at a time,
so independent agents overlap their CPU-side work (prompting, parsing, I/O) without stacking
several large models' activations in VRAM.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import itertools
import logging
import time
from collections.abc import AsyncIterator, Callable
from typing import Any, TypeVar

from app.core.config import Settings
from app.core.logging import log_event
from app.models.base import LocalModel
from app.models.model_manager import ModelManager
from app.schemas.common import Priority

logger = logging.getLogger(__name__)

T = TypeVar("T")


class PriorityPool:
    """A semaphore that admits waiters by (priority, model-already-loaded, FIFO)."""

    def __init__(self, name: str, capacity: int) -> None:
        self.name = name
        self.capacity = capacity
        self._in_use = 0
        self._waiters: list[tuple[int, int, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    @property
    def in_use(self) -> int:
        return self._in_use

    @property
    def waiting(self) -> int:
        return sum(1 for *_, fut in self._waiters if not fut.done())

    async def acquire(self, priority: Priority = Priority.NORMAL, warm: bool = False) -> None:
        if self._in_use < self.capacity and self.waiting == 0:
            self._in_use += 1
            return
        fut: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (priority.rank, 0 if warm else 1, next(self._seq), fut))
        try:
            await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # slot was granted but the waiter went away
            raise

    def release(self) -> None:
        self._in_use = max(0, self._in_use - 1)
        while self._waiters and self._in_use < self.capacity:
            *_, fut = heapq.heappop(self._waiters)
            if not fut.done():
                self._in_use += 1
                fut.set_result(None)


class Lease:
    """Grants use of one loaded model. Inference runs in a worker thread via `run`."""

    def __init__(self, model: LocalModel, key: str) -> None:
        self.model = model
        self.key = key
        self._inflight: asyncio.Future[Any] | None = None

    @property
    def device(self) -> str:
        return self.model.device

    @property
    def busy(self) -> bool:
        return self._inflight is not None and not self._inflight.done()

    async def run(self, fn: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        # Shielded: if the caller is cancelled (timeout / task cancel) the thread cannot be killed,
        # so the scheduler keeps the slot reserved until the thread actually finishes.
        fut = asyncio.ensure_future(asyncio.to_thread(fn, *args, **kwargs))
        self._inflight = fut
        return await asyncio.shield(fut)


class ResourceScheduler:
    def __init__(self, model_manager: ModelManager, settings: Settings) -> None:
        self.models = model_manager
        self.settings = settings
        self._pools = {
            "gpu": PriorityPool("gpu", settings.runtime.gpu_concurrency),
            "cpu": PriorityPool("cpu", settings.runtime.cpu_concurrency),
        }

    @staticmethod
    def _pool_name(device: str) -> str:
        return "cpu" if device == "cpu" else "gpu"

    @contextlib.asynccontextmanager
    async def acquire(
        self,
        model_name: str,
        *,
        priority: Priority = Priority.NORMAL,
        device: str | None = None,
        task_id: str | None = None,
        agent: str | None = None,
    ) -> AsyncIterator[Lease]:
        name = self.models.resolve(model_name)
        target = self.models.device_for(name, device)
        pool = self._pools[self._pool_name(target)]
        waited = time.perf_counter()
        await pool.acquire(priority, warm=self.models.is_loaded(name, device))
        wait_time = time.perf_counter() - waited
        if wait_time > 0.05:
            log_event(
                logger,
                "scheduler_admitted",
                task_id=task_id,
                agent=agent,
                model=name,
                device=target,
                duration=wait_time,
                priority=priority.value,
            )
        try:
            if pool.name == "gpu":
                await self._wait_for_memory(name, device)
            key, model = await self.models.acquire(name, device)
        except BaseException:
            pool.release()
            raise

        lease = Lease(model, key)
        try:
            yield lease
        finally:
            if lease.busy:
                assert lease._inflight is not None
                lease._inflight.add_done_callback(lambda _f: self._finish(pool, key))
            else:
                self._finish(pool, key)

    def _finish(self, pool: PriorityPool, key: str) -> None:
        self.models.release(key)
        pool.release()

    async def _wait_for_memory(self, name: str, device: str | None) -> None:
        """When GPU concurrency > 1, hold back a job whose model cannot fit beside the running ones."""
        target = self.models.device_for(name, device)
        while not self.models.fits(name, device) and self.models.active_on(target) > 0:
            await self.models.wait_for_release(timeout=2.0)

    def stats(self) -> dict[str, dict[str, int]]:
        return {
            name: {"capacity": pool.capacity, "in_use": pool.in_use, "waiting": pool.waiting}
            for name, pool in self._pools.items()
        }
