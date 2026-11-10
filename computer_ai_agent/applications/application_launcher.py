"""Starts applications detached from the agent process and waits for their window to appear."""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass

import psutil

from applications.application_detector import AppEntry, ApplicationDetector

logger = logging.getLogger(__name__)


@dataclass
class LaunchResult:
    ok: bool
    pid: int | None
    message: str
    argv: list[str]


class ApplicationLauncher:
    def __init__(self, display: str = "") -> None:
        self.display = display

    def launch(self, app: AppEntry, extra_args: list[str] | None = None, env: dict[str, str] | None = None) -> LaunchResult:
        try:
            argv = ApplicationDetector.resolve_exec(app)
        except FileNotFoundError as exc:
            return LaunchResult(False, None, str(exc), [])
        argv = argv + list(extra_args or [])
        child_env = dict(os.environ)
        if self.display:
            child_env["DISPLAY"] = self.display
        child_env.update(env or {})
        try:
            proc = subprocess.Popen(  # noqa: S603 - argv comes from the installed .desktop entry
                argv, env=child_env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,  # survive the agent; never tied to our terminal
            )
        except OSError as exc:
            return LaunchResult(False, None, f"failed to start {argv[0]}: {exc}", argv)
        logger.info("launched %s pid=%s: %s", app.name, proc.pid, " ".join(argv))
        return LaunchResult(True, proc.pid, f"started {app.name}", argv)

    @staticmethod
    def process_tree(pid: int) -> set[int]:
        try:
            p = psutil.Process(pid)
            return {pid} | {c.pid for c in p.children(recursive=True)}
        except psutil.Error:
            return {pid}

    @staticmethod
    def running_pids(app: AppEntry) -> set[int]:
        """PIDs of processes that belong to the app (by executable name)."""
        names = {app.binary.lower(), app.binary.lower().removesuffix("-stable")}
        names.discard("")
        out = set()
        for p in psutil.process_iter(["pid", "name", "exe", "cmdline"]):
            try:
                pname = (p.info["name"] or "").lower()
                exe = os.path.basename(p.info["exe"] or "").lower()
                if pname in names or exe in names:
                    out.add(p.info["pid"])
            except (psutil.Error, TypeError):
                continue
        return out

    @staticmethod
    def wait_until(predicate, timeout: float, interval: float = 0.3) -> bool:  # noqa: ANN001
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return True
            time.sleep(interval)
        return False
