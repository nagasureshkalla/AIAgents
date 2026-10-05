"""Model lifecycle: discovery, loading, caching, LRU eviction and device placement."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from enum import StrEnum

from app.core.config import Settings
from app.core.device import DeviceInfo, empty_cache, is_oom_error, memory_probe
from app.core.exceptions import (
    GPUOutOfMemoryError,
    ModelInUseError,
    ModelLoadError,
    ModelNotFoundError,
)
from app.core.logging import log_event
from app.models.base import LocalModel
from app.models.model_registry import ModelRegistry, ModelSpec, ModelType
from app.schemas.result import ModelInfo

logger = logging.getLogger(__name__)

ModelFactory = Callable[[ModelSpec, str, Settings], LocalModel]
MemoryProbe = Callable[[str], tuple[float, float]]  # device -> (free_gb, total_gb)


class ModelState(StrEnum):
    NOT_LOADED = "NOT_LOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    ACTIVE = "ACTIVE"
    IDLE = "IDLE"
    UNLOADED = "UNLOADED"
    FAILED = "FAILED"


def create_local_model(spec: ModelSpec, device: str, settings: Settings) -> LocalModel:
    """Default factory: build the right wrapper for the model type and load weights (blocking)."""
    if spec.type == ModelType.TEXT:
        from app.models.text_model import TextModel

        model: LocalModel = TextModel(spec, device, settings)
    else:
        from app.models.vision_model import VisionModel

        model = VisionModel(spec, device, settings)
    model.load()
    return model


@dataclass
class _Entry:
    key: str
    spec: ModelSpec
    pinned_device: str | None = None  # set for explicit CPU-fallback copies
    state: ModelState = ModelState.NOT_LOADED
    model: LocalModel | None = None
    device: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active: int = 0
    last_used: float = 0.0
    load_seconds: float | None = None
    error: str | None = None

    @property
    def evictable(self) -> bool:
        return self.model is not None and self.active == 0 and not self.lock.locked()


class ModelManager:
    def __init__(
        self,
        registry: ModelRegistry,
        settings: Settings,
        device_info: DeviceInfo,
        factory: ModelFactory = create_local_model,
        probe: MemoryProbe = memory_probe,
    ) -> None:
        self.registry = registry
        self.settings = settings
        self.device_info = device_info
        self._factory = factory
        self._probe = probe
        self._entries: dict[str, _Entry] = {}
        self._load_gate = asyncio.Lock()  # one load at a time so capacity checks stay consistent
        self._release_event = asyncio.Event()
        self._idle_task: asyncio.Task | None = None

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> None:
        if self.settings.runtime.idle_unload_seconds > 0 and self._idle_task is None:
            self._idle_task = asyncio.create_task(self._idle_reaper(), name="model-idle-reaper")

    async def shutdown(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._idle_task
            self._idle_task = None
        for entry in list(self._entries.values()):
            if entry.model is not None:
                await self._unload_entry(entry)

    # ------------------------------------------------------------------ resolution
    def resolve(self, name: str) -> str:
        """Return `name` if installed, otherwise a configured or same-type installed fallback."""
        spec = self.registry.get(name)
        if spec.installed:
            return name
        candidates = list(self.settings.runtime.model_fallbacks.get(name, []))
        compatible = {ModelType.VISION, ModelType.MULTIMODAL} if spec.is_visual else {spec.type}
        candidates += [s.name for s in self.registry.installed() if s.type in compatible and s.name != name]
        for candidate in candidates:
            if candidate in self.registry and self.registry.get(candidate).installed:
                logger.warning("model '%s' not installed; using fallback '%s'", name, candidate)
                return candidate
        raise self._missing_error(spec, fallback_checked=True)

    @staticmethod
    def _missing_error(spec: ModelSpec, fallback_checked: bool = False) -> ModelNotFoundError:
        model_id = spec.hf_id or "<HF_MODEL_ID>"
        hint = f"python scripts/download_models.py --model {model_id} --type {spec.type.value}"
        suffix = " and no installed fallback is available" if fallback_checked else ""
        return ModelNotFoundError(
            f"Model '{spec.name}' is not installed at {spec.path}{suffix}. Download it with: {hint}",
            details={"model": spec.name, "path": str(spec.path)},
        )

    def default_device(self, spec: ModelSpec) -> str:
        return self.device_info.torch_device

    def device_for(self, name: str, device: str | None = None) -> str:
        """The device a model is (or would be) placed on."""
        entry = self._entries.get(self._key(*self._normalize(name, device)))
        if entry is not None and entry.device:
            return entry.device
        return device or self.default_device(self.registry.get(name))

    # ------------------------------------------------------------------ public API
    async def load_model(self, name: str, device: str | None = None) -> LocalModel:
        entry = self._entry(name, device)
        async with entry.lock:
            await self._ensure_loaded(entry)
            entry.last_used = time.time()
            return entry.model  # type: ignore[return-value]

    async def get_model(self, name: str, device: str | None = None) -> LocalModel:
        return await self.load_model(name, device)

    async def unload_model(self, name: str, *, force: bool = False) -> bool:
        unloaded = False
        for key in [k for k, e in self._entries.items() if e.spec.name == name]:
            entry = self._entries[key]
            async with entry.lock:
                if entry.model is None:
                    continue
                if entry.active and not force:
                    raise ModelInUseError(f"Model '{key}' is in use by {entry.active} running inference(s)")
                await self._unload_entry(entry)
                unloaded = True
        return unloaded

    async def acquire(self, name: str, device: str | None = None) -> tuple[str, LocalModel]:
        """Load if necessary and mark the model ACTIVE. Pair with `release(key)`."""
        entry = self._entry(name, device)
        async with entry.lock:
            await self._ensure_loaded(entry)
            entry.active += 1
            entry.state = ModelState.ACTIVE
            entry.last_used = time.time()
            return entry.key, entry.model  # type: ignore[return-value]

    def release(self, key: str) -> None:
        """Mark one inference on `key` finished. Synchronous so it can run from future callbacks."""
        entry = self._entries.get(key)
        if entry is None:
            return
        entry.active = max(0, entry.active - 1)
        entry.last_used = time.time()
        if entry.active == 0 and entry.model is not None:
            entry.state = ModelState.IDLE
        event, self._release_event = self._release_event, asyncio.Event()
        event.set()

    @contextlib.asynccontextmanager
    async def use(self, name: str, device: str | None = None) -> AsyncIterator[LocalModel]:
        key, model = await self.acquire(name, device)
        try:
            yield model
        finally:
            self.release(key)

    async def wait_for_release(self, timeout: float) -> None:
        """Wait until any model finishes an inference (or `timeout` elapses)."""
        event = self._release_event
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(event.wait(), timeout)

    def active_on(self, device: str) -> int:
        return sum(e.active for e in self._entries.values() if e.device == device)

    async def list_models(self) -> list[ModelInfo]:
        infos: list[ModelInfo] = []
        for spec in self.registry.list():
            entries = [e for e in self._entries.values() if e.spec.name == spec.name] or [None]
            for entry in entries:
                infos.append(
                    ModelInfo(
                        name=entry.key if entry else spec.name,
                        type=spec.type.value,
                        path=str(spec.path),
                        hf_id=spec.hf_id,
                        installed=spec.installed,
                        state=(entry.state if entry else ModelState.NOT_LOADED).value,
                        device=entry.device if entry else None,
                        priority=spec.priority,
                        source=spec.source,
                        disk_size_gb=spec.disk_size_gb(),
                        memory_gb=entry.model.memory_gb if entry and entry.model else None,
                        last_used=entry.last_used if entry and entry.last_used else None,
                        error=entry.error if entry else None,
                    )
                )
        return infos

    def memory(self, device: str) -> tuple[float, float]:
        """(free_gb, total_gb) for a device, via the configured probe."""
        return self._probe(device)

    def loaded_models(self) -> list[str]:
        return [e.key for e in self._entries.values() if e.model is not None]

    def is_loaded(self, name: str, device: str | None = None) -> bool:
        entry = self._entries.get(self._key(*self._normalize(name, device)))
        return entry is not None and entry.model is not None

    def state(self, name: str, device: str | None = None) -> ModelState:
        entry = self._entries.get(self._key(*self._normalize(name, device)))
        return entry.state if entry else ModelState.NOT_LOADED

    def fits(self, name: str, device: str | None = None) -> bool:
        """Whether the model is loaded or could be loaded now (after evicting idle models)."""
        if self.is_loaded(name, device):
            return True
        spec = self.registry.get(name)
        target = device or self.default_device(spec)
        free, total = self._probe(target)
        reclaimable = sum(
            self._entry_memory(e) for e in self._entries.values() if e.evictable and e.device == target
        )
        threshold = self._threshold(target)
        projected_used = (total - free - reclaimable) + spec.estimated_memory_gb()
        return projected_used <= total * threshold

    # ------------------------------------------------------------------ internals
    def _key(self, name: str, device: str | None) -> str:
        return f"{name}@{device}" if device else name

    def _normalize(self, name: str, device: str | None) -> tuple[str, str | None]:
        """A device equal to the default placement maps to the primary entry (key without suffix)."""
        if device is not None and device == self.default_device(self.registry.get(name)):
            device = None
        return name, device

    def _entry(self, name: str, device: str | None) -> _Entry:
        spec = self.registry.get(name)
        name, device = self._normalize(name, device)
        key = self._key(name, device)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry(key=key, spec=spec, pinned_device=device)
            self._entries[key] = entry
        return entry

    def _threshold(self, device: str) -> float:
        rt = self.settings.runtime
        return rt.vram_threshold if device.startswith("cuda") else rt.ram_threshold

    @staticmethod
    def _entry_memory(entry: _Entry) -> float:
        if entry.model is not None and entry.model.memory_gb:
            return entry.model.memory_gb
        return entry.spec.estimated_memory_gb()

    async def _ensure_loaded(self, entry: _Entry) -> None:
        if entry.model is not None:
            return
        spec = entry.spec
        if not spec.installed:
            raise self._missing_error(spec)
        target = entry.pinned_device or self.default_device(spec)
        async with self._load_gate:
            await self._make_room(spec, target, exclude=entry.key)
            entry.state = ModelState.LOADING
            entry.error = None
            log_event(logger, "model_loading", model=entry.key, device=target)
            started = time.perf_counter()
            try:
                model = await self._load_with_recovery(entry, target)
            except Exception as exc:
                entry.state = ModelState.FAILED
                entry.error = str(exc)
                log_event(logger, "model_load_failed", level=logging.ERROR, model=entry.key, message=str(exc))
                raise
            entry.model = model
            entry.device = model.device
            entry.load_seconds = round(time.perf_counter() - started, 2)
            entry.state = ModelState.LOADED
            log_event(
                logger,
                "model_loaded",
                model=entry.key,
                device=model.device,
                duration=entry.load_seconds,
                status="ok",
            )

    async def _load_with_recovery(self, entry: _Entry, target: str) -> LocalModel:
        spec = entry.spec
        try:
            return await asyncio.to_thread(self._factory, spec, target, self.settings)
        except Exception as exc:
            if not is_oom_error(exc):
                if isinstance(exc, (ModelLoadError, ModelNotFoundError)):
                    raise
                raise ModelLoadError(f"Failed to load model '{spec.name}': {exc}") from exc
            logger.warning("OOM while loading %s on %s; evicting idle models and retrying", spec.name, target)

        empty_cache(target)
        await self._evict(target, exclude=entry.key, evict_all=True)
        try:
            return await asyncio.to_thread(self._factory, spec, target, self.settings)
        except Exception as exc:
            if not is_oom_error(exc):
                raise ModelLoadError(f"Failed to load model '{spec.name}': {exc}") from exc
            empty_cache(target)
            if self.settings.runtime.allow_cpu_fallback and target != "cpu":
                logger.warning("OOM persists for %s; falling back to CPU", spec.name)
                try:
                    return await asyncio.to_thread(self._factory, spec, "cpu", self.settings)
                except Exception as cpu_exc:
                    raise ModelLoadError(f"CPU fallback for '{spec.name}' failed: {cpu_exc}") from cpu_exc
            raise GPUOutOfMemoryError(f"Not enough memory to load model '{spec.name}' on {target}") from exc

    async def _make_room(self, spec: ModelSpec, target: str, exclude: str) -> None:
        """Evict LRU idle models until the new model fits under the memory threshold and model cap."""
        max_loaded = self.settings.runtime.max_loaded_models
        while len(self.loaded_models()) >= max_loaded:
            if not await self._evict_one(None, exclude):
                break
        need = spec.estimated_memory_gb()
        threshold = self._threshold(target)
        while True:
            free, total = self._probe(target)
            if (total - free) + need <= total * threshold:
                return
            if not await self._evict_one(target, exclude):
                logger.warning(
                    "model %s (~%.2f GB) may not fit on %s (free %.2f/%.2f GB); attempting load anyway",
                    spec.name,
                    need,
                    target,
                    free,
                    total,
                )
                return

    async def _evict_one(self, device: str | None, exclude: str) -> bool:
        candidates = [
            e
            for e in self._entries.values()
            if e.key != exclude and e.evictable and (device is None or e.device == device)
        ]
        if not candidates:
            return False
        victim = min(candidates, key=lambda e: e.last_used)
        async with victim.lock:
            if victim.active or victim.model is None:
                return False
            log_event(logger, "model_evicted", model=victim.key, device=victim.device, status="lru")
            await self._unload_entry(victim)
        return True

    async def _evict(self, device: str, exclude: str, evict_all: bool = False) -> None:
        while await self._evict_one(device, exclude):
            if not evict_all:
                return

    async def _unload_entry(self, entry: _Entry) -> None:
        model, device = entry.model, entry.device
        entry.model = None
        entry.device = None
        entry.state = ModelState.UNLOADED
        if model is not None:
            await asyncio.to_thread(model.unload)
            empty_cache(device)
            log_event(logger, "model_unloaded", model=entry.key, device=device)

    async def _idle_reaper(self) -> None:
        ttl = self.settings.runtime.idle_unload_seconds
        while True:
            await asyncio.sleep(max(5.0, ttl / 4))
            now = time.time()
            for entry in list(self._entries.values()):
                if entry.evictable and entry.spec.name not in self.settings.runtime.preload and now - entry.last_used > ttl:
                    async with entry.lock:
                        if entry.model is not None and entry.active == 0:
                            log_event(logger, "model_idle_unload", model=entry.key, status="idle")
                            await self._unload_entry(entry)
