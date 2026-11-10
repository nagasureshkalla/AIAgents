"""Test doubles: a simulated desktop with one calculator-like app, and a scripted local model."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from actions.platform import Desktop
from actions.window_manager import WindowManager
from applications.application_detector import AppEntry, ApplicationDetector
from applications.application_manager import ApplicationManager
from config.settings import Settings
from core.types import Element, Observation, WindowInfo
from models.llm import Backend, ChatMessage, Generation
from models.model_manager import ModelManager
from vision.screen_capture import image_hash

SCREEN = (1280, 800)
WIN = WindowInfo(id=101, title="Calculator", wm_class="gnome-calculator", pid=4242, geometry=(400, 150, 360, 500),
                 active=True)


class FakeApp:
    """A tiny calculator: typing goes into the display, Enter evaluates, the 'Delete all' button needs approval."""

    def __init__(self) -> None:
        self.display = ""
        self.running = True
        self.clicks: list[tuple[int, int]] = []
        self.deleted = False
        self.lock = threading.Lock()

    def type(self, text: str) -> None:
        with self.lock:
            for ch in text:
                if ch == "\n":
                    self.enter()
                else:
                    self.display += ch

    def enter(self) -> None:
        try:
            self.display = str(eval(self.display.replace("×", "*"), {"__builtins__": {}}))  # noqa: S307 - test only
        except Exception:  # noqa: BLE001
            self.display = "Error"

    def elements(self) -> list[Element]:
        x, y, w, _ = WIN.geometry
        els = [
            Element(0, "textbox", self.display, (x + 10, y + 60, w - 20, 40), "accessibility", ["focused", "editable"]),
            Element(0, "button", "=", (x + 280, y + 420, 60, 60), "accessibility"),
            Element(0, "button", "Delete all history", (x + 10, y + 420, 120, 40), "accessibility"),
        ]
        for i, d in enumerate("789456123"):
            els.append(Element(0, "button", d, (x + 10 + (i % 3) * 70, y + 200 + (i // 3) * 60, 60, 50), "accessibility"))
        return els

    def render(self) -> Image.Image:
        img = Image.new("RGB", SCREEN, (30, 30, 30))
        d = ImageDraw.Draw(img)
        x, y, w, h = WIN.geometry
        d.rectangle([x, y, x + w, y + h], fill=(60, 60, 60))
        d.rectangle([x + 10, y + 60, x + w - 10, y + 100], fill=(255, 255, 255))
        d.text((x + 20, y + 70), self.display or " ", fill=(0, 0, 0))
        for i, (cx, cy) in enumerate(self.clicks[-3:]):  # clicks leave a visible mark so the verifier sees change
            d.ellipse([cx - 3 + i, cy - 3, cx + 3 + i, cy + 3], fill=(250, 200, 0))
        return img


class FakeScreen:
    def __init__(self, app: FakeApp) -> None:
        self.app = app

    def grab(self) -> Image.Image:
        return self.app.render()

    def size(self) -> tuple[int, int]:
        return SCREEN


class FakeMouse:
    def __init__(self, app: FakeApp) -> None:
        self.app = app
        self.position = (0, 0)
        self.log: list[tuple] = []

    def move(self, x: int, y: int) -> None:
        self.position = (x, y)
        self.log.append(("move", x, y))

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> None:
        self.position = (x, y)
        self.log.append(("click", x, y, button, count))
        self.app.clicks.append((x, y))
        for e in self.app.elements():
            ex, ey, ew, eh = e.bbox
            if ex <= x <= ex + ew and ey <= y <= ey + eh:
                if e.text == "Delete all history":
                    self.app.deleted = True
                elif e.text == "=":
                    self.app.enter()
                elif e.type == "button":
                    self.app.type(e.text)

    def drag(self, *a: Any, **k: Any) -> None:
        self.log.append(("drag", *a))

    def scroll(self, x: int, y: int, clicks: int) -> None:
        self.log.append(("scroll", x, y, clicks))


class FakeKeyboard:
    def __init__(self, app: FakeApp) -> None:
        self.app = app
        self.log: list[tuple] = []

    def press(self, name: str) -> None:
        self.log.append(("press", name))
        if name.lower() in ("enter", "return"):
            self.app.enter()
        elif name.lower() in ("escape", "esc"):
            self.app.display = ""

    def hotkey(self, names: list[str]) -> None:
        self.log.append(("hotkey", tuple(names)))

    def type_text(self, text: str) -> None:
        self.log.append(("type", text))
        self.app.type(text)


class FakeWindows(WindowManager):
    def __init__(self, app: FakeApp) -> None:
        self.app = app
        self.closed: list[int] = []

    def list_windows(self) -> list[WindowInfo]:
        return [WIN] if self.app.running else []

    def active_window(self) -> WindowInfo | None:
        return WIN if self.app.running else None

    def activate(self, window_id: int) -> bool:
        return True

    def close(self, window_id: int) -> bool:
        self.closed.append(window_id)
        self.app.running = False
        return True


class FakeDetector:
    """Observation straight from the fake app (stands in for AT-SPI + OCR)."""

    def __init__(self, desktop: Desktop, app: FakeApp) -> None:
        self.desktop = desktop
        self.app = app
        self.last_timing: dict[str, float] = {}

    def observe(self, force_ocr: bool = False, use_cache: bool = True, target: Any = None) -> Observation:
        img = self.app.render()
        els = self.app.elements()
        for i, e in enumerate(els, start=1):
            e.id = i
        return Observation(0.0, SCREEN, WIN if self.app.running else None, els, img, image_hash(img),
                           focused=els[0], sources=["accessibility"])

    def invalidate(self) -> None:
        pass


class ScriptedBackend(Backend):
    """Answers by prompt type. `actions` is consumed one per action-planner call."""

    def __init__(self, actions: list[dict[str, Any]], plan: list[dict[str, Any]] | None = None,
                 goal_complete: bool = True) -> None:
        self.actions = deque(actions)
        self.plan = plan or [{"app": "calculator", "instruction": "Calculate 25 * 89", "steps": ["result shown"]}]
        self.goal_complete = goal_complete
        self.calls: list[str] = []

    async def chat(self, model: str, messages: list[ChatMessage], *, json_mode: bool, max_tokens: int,
                   temperature: float) -> Generation:
        system = messages[0].content
        if "planning module" in system:
            kind, out = "plan", {"subtasks": self.plan}
        elif "operate the application" in system:
            kind = "action"
            out = self.actions.popleft() if self.actions else {"action": "done", "reason": "nothing left"}
        elif "verify whether" in system:
            kind, out = "goal", {"complete": self.goal_complete, "evidence": "display shows the result"}
        else:
            kind, out = "other", {}
        self.calls.append(kind)
        return Generation(json.dumps(out), model, 1.0)

    async def health(self, model: str) -> dict[str, Any]:
        return {"ok": True, "backend": "scripted", "model": model, "loaded": True}


def make_runtime(tmp: Path, actions: list[dict[str, Any]], **backend_kw: Any):  # noqa: ANN201
    from core.runtime import build_runtime

    app = FakeApp()
    settings = Settings(data_dir=tmp / "data", logs_dir=tmp / "logs", harness_dir=tmp / "harness", action_delay=0.0, settle_timeout=0.0, nav_timeout=0.0, vision_mode="never",
                        max_steps_per_subtask=10, confirmation_timeout=5)
    desktop = Desktop("fake", ":99", FakeScreen(app), FakeMouse(app), FakeKeyboard(app), FakeWindows(app), True)
    detector_apps = ApplicationDetector(dirs=[])
    detector_apps.apps = {"org.gnome.Calculator": AppEntry("org.gnome.Calculator", "Calculator", "gnome-calculator",
                                                           wm_class="gnome-calculator")}
    apps = ApplicationManager(desktop.windows, ":99", 2.0, detector_apps)
    backend = ScriptedBackend(actions, **backend_kw)
    models = ModelManager(settings, backend=backend)
    rt = build_runtime(settings, desktop=desktop, models=models, apps=apps, detector=FakeDetector(desktop, app))
    return rt, app, backend
