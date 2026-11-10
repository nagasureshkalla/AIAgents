"""Task execution history (data/tasks/<task_id>.json) and lookup of earlier successful workflows."""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


class TaskMemory:
    def __init__(self, data_dir: Path) -> None:
        self.dir = data_dir / "tasks"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def save(self, record: dict[str, Any]) -> Path:
        path = self.dir / f"{record['task_id']}.json"
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(record, indent=2, default=str, ensure_ascii=False))
            tmp.replace(path)
        return path

    def load(self, task_id: str) -> dict[str, Any] | None:
        try:
            return json.loads((self.dir / f"{task_id}.json").read_text())
        except (OSError, json.JSONDecodeError):
            return None

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        files = sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
        out = []
        for f in files:
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            out.append({k: d.get(k) for k in ("task_id", "user_command", "application", "result", "duration",
                                              "started_at", "summary")})
        return out

    def similar_successes(self, command: str, limit: int = 2, min_score: float = 0.4) -> list[dict[str, Any]]:
        q = _tokens(command)
        if not q:
            return []
        scored = []
        for f in sorted(self.dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)[:300]:
            try:
                d = json.loads(f.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if d.get("result") != "success":
                continue
            t = _tokens(d.get("user_command", ""))
            score = len(q & t) / len(q | t) if t else 0.0
            if score >= min_score:
                scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:limit]]

    @staticmethod
    def new_record(task_id: str, command: str) -> dict[str, Any]:
        return {"task_id": task_id, "user_command": command, "application": None, "subtasks": [], "actions": [],
                "messages": [], "result": "running", "summary": "", "started_at": time.time(), "duration": None}
