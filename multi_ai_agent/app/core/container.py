"""Dependency container: builds and owns every long-lived service (no module-level singletons)."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.agents.base_agent import AgentContext
from app.agents.registry import AgentRegistry
from app.core.config import Settings
from app.core.device import DeviceInfo, get_device, memory_probe
from app.core.events import EventBus, EventType
from app.core.exceptions import InvalidRequestError
from app.core.harness import HarnessStore
from app.core.local_guard import enforce_local_inference
from app.models.downloader import DownloadManager
from app.models.model_manager import (
    MemoryProbe,
    ModelFactory,
    ModelManager,
    create_local_model,
)
from app.models.model_registry import ModelRegistry
from app.models.speech_model import SpeechToText, TextToSpeech
from app.orchestration.orchestrator import Orchestrator
from app.orchestration.result_aggregator import ResultAggregator
from app.orchestration.scheduler import ResourceScheduler
from app.orchestration.task_queue import TaskQueue, TaskStore
from app.schemas.common import InputType
from app.schemas.task import TaskRecord, TaskRequest
from app.utils.files import ensure_dirs, resolve_input_path
from app.utils.image import validate_image_file
from app.utils.performance import PerformanceMonitor

logger = logging.getLogger(__name__)


def apply_offline_env(settings: Settings) -> None:
    """Make the HF libraries refuse network access when offline_mode is on."""
    if settings.runtime.offline_mode:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ.setdefault("HF_HOME", str(settings.cache_dir / "huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class Container:
    settings: Settings
    device_info: DeviceInfo
    model_registry: ModelRegistry
    models: ModelManager
    scheduler: ResourceScheduler
    events: EventBus
    monitor: PerformanceMonitor
    agents: AgentRegistry
    aggregator: ResultAggregator
    store: TaskStore
    orchestrator: Orchestrator
    queue: TaskQueue
    downloads: DownloadManager | None = None
    speech: SpeechToText | None = None
    tts: TextToSpeech | None = None
    cloud_keys_ignored: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    @classmethod
    def build(
        cls,
        settings: Settings,
        *,
        device_info: DeviceInfo | None = None,
        model_factory: ModelFactory = create_local_model,
        probe: MemoryProbe = memory_probe,
        persist_tasks: bool = True,
    ) -> "Container":
        cloud_keys = enforce_local_inference(settings)
        apply_offline_env(settings)
        ensure_dirs(settings.input_dir, settings.output_dir, settings.cache_dir, settings.tasks_dir)
        device_info = device_info or get_device(settings.runtime.device, settings.inference.cuda_device)
        model_registry = ModelRegistry.from_settings(settings)
        models = ModelManager(model_registry, settings, device_info, factory=model_factory, probe=probe)
        scheduler = ResourceScheduler(models, settings)
        events = EventBus()
        monitor = PerformanceMonitor()
        harness = None
        if settings.harness.enabled:
            harness = HarnessStore(
                settings.resolve_path(settings.harness.root),
                {agent: settings.resolve_path(d) for agent, d in settings.harness.dirs.items()},
            )
        tts = TextToSpeech(settings)
        ctx = AgentContext(
            settings=settings, scheduler=scheduler, models=models, monitor=monitor, events=events, harness=harness,
            tts=tts,
        )
        agents = AgentRegistry.build(ctx)
        aggregator = ResultAggregator(ctx)
        store = TaskStore(settings.tasks_dir if persist_tasks else None)
        orchestrator = Orchestrator(
            settings=settings,
            agents=agents,
            aggregator=aggregator,
            events=events,
            store=store,
            device_info=device_info,
        )

        def on_cancel(record: TaskRecord) -> None:
            events.emit(record.task_id, EventType.TASK_CANCELLED, reason="cancelled before start")

        queue = TaskQueue(store, orchestrator.run_task, settings.runtime.max_concurrent_tasks, on_cancel)
        return cls(
            settings=settings,
            device_info=device_info,
            model_registry=model_registry,
            models=models,
            scheduler=scheduler,
            events=events,
            monitor=monitor,
            agents=agents,
            aggregator=aggregator,
            store=store,
            orchestrator=orchestrator,
            queue=queue,
            downloads=DownloadManager(models.refresh, os.environ.get("MAA_CONFIG")),
            cloud_keys_ignored=cloud_keys,
            speech=SpeechToText(settings, device_info),
            tts=tts,
        )

    async def start(self) -> None:
        await self.models.start()
        await self.queue.start()
        for name in self.settings.runtime.preload:
            try:
                await self.models.load_model(self.models.resolve(name))
            except Exception as exc:
                logger.warning("preload of model %s failed: %s", name, exc)
        logger.info(
            "system ready: device=%s agents=%s models=%s",
            self.device_info.type,
            ",".join(self.agents.names()),
            ",".join(s.name for s in self.model_registry.installed()) or "none installed",
        )

    async def stop(self) -> None:
        if self.downloads is not None:
            await self.downloads.shutdown()
        await self.queue.stop()
        if self.speech is not None:
            self.speech.unload()
        await self.models.shutdown()
        from app.agents.control_panel_agent import ControlPanelClient

        ControlPanelClient.shutdown()  # the computer_ai_agent service, if this process started it

    # ------------------------------------------------------------------ task API
    def prepare_request(self, request: TaskRequest) -> TaskRequest:
        """Validate and normalize a request (image path sandboxing, prompt size, input type)."""
        security = self.settings.security
        if len(request.prompt) > security.max_prompt_chars:
            raise InvalidRequestError(f"Prompt exceeds {security.max_prompt_chars} characters")
        update: dict = {}
        if request.image_path:
            path: Path = resolve_input_path(request.image_path, self.settings.input_dir, self.settings.base_dir)
            validate_image_file(path, security)
            update["image_path"] = str(path)
            if request.input_type in (None, InputType.TEXT):
                update["input_type"] = InputType.IMAGE
        elif request.input_type in (InputType.IMAGE, InputType.MULTIMODAL):
            raise InvalidRequestError(f"input_type '{request.input_type.value}' requires image_path")
        elif request.input_type is None:
            update["input_type"] = InputType.TEXT
        if request.options.agents:
            for name in request.options.agents:
                if not self.agents.has(name) or self.agents.get(name).role != "worker":
                    raise InvalidRequestError(f"Unknown worker agent '{name}'")
        return request.model_copy(update=update)

    def submit(self, request: TaskRequest) -> TaskRecord:
        record = self.store.create(self.prepare_request(request))
        self.events.emit(record.task_id, EventType.TASK_QUEUED, priority=record.request.priority.value)
        self.queue.submit(record)
        return record

    async def run_and_wait(self, request: TaskRequest, timeout: float) -> TaskRecord:
        record = self.submit(request)
        try:
            return await self.store.wait(record.task_id, timeout)
        except TimeoutError:
            self.queue.cancel(record.task_id)
            raise

    def cancel(self, task_id: str) -> bool:
        return self.queue.cancel(task_id)
