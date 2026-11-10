"""Wires the whole system together once per process. Agents receive this instead of a dozen arguments."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from actions.action_engine import ActionEngine
from applications.application_manager import ApplicationManager
from config.settings import Settings
from core.events import EventBus
from core.logging_setup import ActionLogger
from memory.task_memory import TaskMemory
from models.model_manager import ModelManager
from planner.action_planner import ActionPlanner
from planner.task_planner import TaskPlanner
from planner.verifier import Verifier
from safety.confirmation import ConfirmationBroker
from safety.permission_manager import PermissionManager
from vision.ui_detector import UIDetector

if TYPE_CHECKING:
    from actions.platform import Desktop

logger = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    bus: EventBus
    desktop: Desktop
    detector: UIDetector
    engine: ActionEngine
    apps: ApplicationManager
    models: ModelManager
    task_planner: TaskPlanner
    action_planner: ActionPlanner
    verifier: Verifier
    permissions: PermissionManager
    broker: ConfirmationBroker
    task_memory: TaskMemory
    action_log: ActionLogger


def build_runtime(settings: Settings, bus: EventBus | None = None, desktop: Desktop | None = None,
                  models: ModelManager | None = None, apps: ApplicationManager | None = None,
                  detector: UIDetector | None = None) -> Runtime:
    bus = bus or EventBus()
    if desktop is None:
        from actions.platform import create_desktop

        desktop = create_desktop(settings.display, settings.accessibility_enabled)
    if detector is None:
        ocr = None
        if settings.ocr_enabled:
            from vision.ocr import OcrEngine

            ocr = OcrEngine()
        a11y = None
        if desktop.accessibility:
            from vision.accessibility import AccessibilityReader

            a11y = AccessibilityReader()
        detector = UIDetector(desktop, settings, ocr, a11y)
    models = models or ModelManager(settings)
    apps = apps or ApplicationManager(desktop.windows, desktop.display, settings.launch_timeout)
    engine = ActionEngine(desktop, settings.action_delay)
    return Runtime(
        settings=settings,
        bus=bus,
        desktop=desktop,
        detector=detector,
        engine=engine,
        apps=apps,
        models=models,
        task_planner=TaskPlanner(models.llm),
        action_planner=ActionPlanner(models.llm, harness_dir=settings.harness_dir),
        verifier=Verifier(models.vision.provider),
        permissions=PermissionManager(settings.require_confirmation),
        broker=ConfirmationBroker(bus, settings.confirmation_timeout),
        task_memory=TaskMemory(settings.data_dir),
        action_log=ActionLogger(settings.logs_dir),
    )
