"""Application Manager: discover, launch, detect, switch and close applications; tracks PIDs and window ids."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from applications.application_detector import AppEntry, ApplicationDetector
from applications.application_launcher import ApplicationLauncher
from core.types import WindowInfo

if TYPE_CHECKING:
    from actions.window_manager import WindowManager

logger = logging.getLogger(__name__)


class ApplicationNotFound(RuntimeError):
    pass


class LaunchFailed(RuntimeError):
    pass


@dataclass
class RunningApp:
    entry: AppEntry
    pids: set[int] = field(default_factory=set)
    window_ids: list[int] = field(default_factory=list)
    launched_by_agent: bool = False
    last_seen: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"app": self.entry.name, "id": self.entry.id, "pids": sorted(self.pids),
                "windows": self.window_ids, "launched_by_agent": self.launched_by_agent}


class ApplicationManager:
    def __init__(self, windows: WindowManager, display: str = "", launch_timeout: float = 25.0,
                 detector: ApplicationDetector | None = None) -> None:
        self.windows = windows
        self.detector = detector or ApplicationDetector()
        self.launcher = ApplicationLauncher(display)
        self.launch_timeout = launch_timeout
        self.running: dict[str, RunningApp] = {}

    # ---- discovery ------------------------------------------------------------------------------------------
    def installed(self) -> list[AppEntry]:
        if not self.detector.apps:
            self.detector.discover()
        return sorted((a for a in self.detector.apps.values() if not a.no_display), key=lambda a: a.name.lower())

    def find(self, name: str) -> AppEntry:
        entry = self.detector.find(name)
        if entry is None:
            raise ApplicationNotFound(f"No installed application matches {name!r}")
        return entry

    # ---- detection ------------------------------------------------------------------------------------------
    def _track(self, entry: AppEntry) -> RunningApp:
        ra = self.running.get(entry.id)
        if ra is None:
            ra = self.running[entry.id] = RunningApp(entry)
        return ra

    def app_windows(self, entry: AppEntry) -> list[WindowInfo]:
        ra = self.running.get(entry.id)
        pids: set[int] = set()
        if ra:
            for pid in list(ra.pids):
                pids |= self.launcher.process_tree(pid)
        wins = [w for w in self.windows.list_windows() if self.detector.matches_window(entry, w, pids)]
        if wins:
            ra = self._track(entry)
            ra.window_ids = [w.id for w in wins]
            ra.pids |= {w.pid for w in wins if w.pid}
            ra.last_seen = time.time()
        return wins

    def is_running(self, entry: AppEntry) -> bool:
        return bool(self.app_windows(entry)) or bool(self.launcher.running_pids(entry))

    def app_for_window(self, win: WindowInfo | None) -> AppEntry | None:
        if win is None:
            return None
        for ra in self.running.values():
            if win.id in ra.window_ids or (win.pid and win.pid in ra.pids):
                return ra.entry
        if not self.detector.apps:
            self.detector.discover()
        for entry in self.detector.apps.values():
            if not entry.no_display and self.detector.matches_window(entry, win):
                return entry
        return None

    # ---- actions --------------------------------------------------------------------------------------------
    def ensure_open(self, name_or_entry: str | AppEntry, extra_args: list[str] | None = None,
                    env: dict[str, str] | None = None, new_window: bool = False) -> tuple[AppEntry, WindowInfo, bool]:
        """Bring the app's window to the front, launching it first if needed. Returns (entry, window, launched)."""
        entry = name_or_entry if isinstance(name_or_entry, AppEntry) else self.find(name_or_entry)
        existing = self.app_windows(entry)
        if existing and not new_window:
            win = existing[-1]
            self.windows.activate(win.id)
            return entry, self._refresh(win), False

        before = {w.id for w in self.windows.list_windows()}
        res = self.launcher.launch(entry, extra_args, env)
        if not res.ok:
            raise LaunchFailed(res.message)
        ra = self._track(entry)
        ra.launched_by_agent = True
        if res.pid:
            ra.pids.add(res.pid)

        def new_window_of_app(w: WindowInfo) -> bool:
            if w.id in before:
                return False
            tree: set[int] = set()
            for pid in ra.pids:
                tree |= self.launcher.process_tree(pid)
            return self.detector.matches_window(entry, w, tree)

        win = self.windows.wait_for(new_window_of_app, self.launch_timeout)
        if win is None:
            # Single-instance apps may hand the request to an already-running process (new window, other pid).
            wins = self.app_windows(entry)
            if wins:
                win = wins[-1]
            else:
                raise LaunchFailed(
                    f"{entry.name} was started but no window appeared within {self.launch_timeout:.0f}s"
                )
        time.sleep(0.4)  # let the first frame render before activation/observation
        self.windows.activate(win.id)
        ra.window_ids = sorted(set(ra.window_ids) | {win.id})
        return entry, self._refresh(win), True

    def _refresh(self, win: WindowInfo) -> WindowInfo:
        for w in self.windows.list_windows():
            if w.id == win.id:
                return w
        return win

    def activate(self, name_or_entry: str | AppEntry) -> WindowInfo:
        entry = name_or_entry if isinstance(name_or_entry, AppEntry) else self.find(name_or_entry)
        wins = self.app_windows(entry)
        if not wins:
            raise ApplicationNotFound(f"{entry.name} has no open window")
        self.windows.activate(wins[-1].id)
        return self._refresh(wins[-1])

    def close(self, name_or_entry: str | AppEntry) -> int:
        entry = name_or_entry if isinstance(name_or_entry, AppEntry) else self.find(name_or_entry)
        wins = self.app_windows(entry)
        for w in wins:
            self.windows.close(w.id)
        self.running.pop(entry.id, None)
        return len(wins)

    def snapshot(self) -> list[dict[str, Any]]:
        return [ra.to_dict() for ra in self.running.values()]

    # ---- async wrappers (window/process calls block) --------------------------------------------------------
    async def aensure_open(self, *args: Any, **kwargs: Any) -> tuple[AppEntry, WindowInfo, bool]:
        return await asyncio.to_thread(self.ensure_open, *args, **kwargs)

    async def aactivate(self, name: str | AppEntry) -> WindowInfo:
        return await asyncio.to_thread(self.activate, name)

    async def aclose(self, name: str | AppEntry) -> int:
        return await asyncio.to_thread(self.close, name)
