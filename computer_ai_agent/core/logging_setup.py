"""Log layout: logs/master/master.log, logs/applications/<app>.log, logs/tasks/<task_id>/ (debug artifacts)."""

from __future__ import annotations

import json
import logging
import re
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

FMT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(logs_dir: Path, debug: bool = False) -> None:
    for sub in ("master", "applications", "tasks"):
        (logs_dir / sub).mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG if debug else logging.INFO)
    if not any(getattr(h, "_cua", False) for h in root.handlers):
        fh = RotatingFileHandler(logs_dir / "master" / "master.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter(FMT))
        fh._cua = True  # type: ignore[attr-defined]
        sh = logging.StreamHandler()
        sh.setFormatter(logging.Formatter(FMT))
        sh.setLevel(logging.DEBUG if debug else logging.INFO)
        sh._cua = True  # type: ignore[attr-defined]
        root.addHandler(fh)
        root.addHandler(sh)
    for noisy in ("httpx", "httpcore", "PIL", "urllib3", "asyncio", "multipart", "RapidOCR"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "app"


class ActionLogger:
    """Writes the human-readable per-application action log required by the spec."""

    def __init__(self, logs_dir: Path) -> None:
        self.dir = logs_dir / "applications"
        self.dir.mkdir(parents=True, exist_ok=True)

    def log(self, app: str, agent: str, action: str, target: str, point: Any, ok: bool, message: str = "") -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        coords = f"{point[0]}, {point[1]}" if point else "-"
        entry = (
            f"{stamp}\nApplication: {app}\nAgent: {agent}\nAction: {action}\nTarget: {target or '-'}\n"
            f"Coordinates: {coords}\nResult: {'success' if ok else 'failed'}"
            + (f"\nDetail: {message}" if message else "")
            + "\n\n"
        )
        with open(self.dir / f"{slug(app)}.log", "a", encoding="utf-8") as f:
            f.write(entry)


class TaskDebugRecorder:
    """In DEBUG mode, keeps screenshots, OCR, model decisions, actions and timing for each step of a task."""

    def __init__(self, logs_dir: Path, task_id: str, enabled: bool) -> None:
        self.enabled = enabled
        self.dir = logs_dir / "tasks" / task_id
        self.journal = logs_dir / "tasks" / f"{task_id}.jsonl"
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        if enabled:
            self.dir.mkdir(parents=True, exist_ok=True)

    def record(self, kind: str, **data: Any) -> None:
        """Always appended to logs/tasks/<id>.jsonl (actions, decisions, results, errors, timing)."""
        entry = {"ts": time.time(), "kind": kind, **data}
        with open(self.journal, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, default=str) + "\n")

    def screenshot(self, step: int, image: Any, tag: str = "") -> str | None:
        if not self.enabled or image is None:
            return None
        path = self.dir / f"step_{step:03d}{('_' + tag) if tag else ''}.png"
        image.save(path)
        return str(path)

    def dump(self, step: int, name: str, payload: Any) -> None:
        if not self.enabled:
            return
        (self.dir / f"step_{step:03d}_{name}.json").write_text(json.dumps(payload, indent=2, default=str))
