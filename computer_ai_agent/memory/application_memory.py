"""Per-application knowledge that persists across runs and improves from successful interactions.

data/applications/<app>/
    profile.json             identity, launch info, counters
    knowledge.json           editable tips/shortcuts (seeded from the plugin's knowledge.json, never overwritten)
    ui_memory.json           UI elements seen before (text, type, position relative to the window)
    successful_actions.json  action sequences that completed a goal
    failed_patterns.json     actions that did not work, with the reason
"""

from __future__ import annotations

import json
import re
import threading
import time
from pathlib import Path
from typing import Any

from core.logging_setup import slug
from core.types import Observation

MAX_UI = 400
MAX_SUCCESS = 60
MAX_FAIL = 80


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) > 2}


def _similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


class ApplicationMemory:
    def __init__(self, data_dir: Path, app_name: str, seed_knowledge: dict[str, Any] | None = None) -> None:
        self.app_name = app_name
        self.dir = data_dir / "applications" / slug(app_name)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.profile = self._load("profile.json", {"application": app_name, "created": time.time(), "tasks": 0,
                                                   "successes": 0, "failures": 0})
        kpath = self.dir / "knowledge.json"
        if not kpath.exists():  # seed once; afterwards the file belongs to the user (and to learning below)
            self._save("knowledge.json", seed_knowledge or {"application": app_name, "tips": [], "shortcuts": {}})
        self.knowledge = self._load("knowledge.json", {})
        self.ui = self._load("ui_memory.json", {"elements": {}})
        self.successes = self._load("successful_actions.json", [])
        self.failures = self._load("failed_patterns.json", [])

    # ---- io -------------------------------------------------------------------------------------------------
    def _load(self, name: str, default: Any) -> Any:
        p = self.dir / name
        try:
            return json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            return default

    def _save(self, name: str, data: Any) -> None:
        tmp = self.dir / f".{name}.tmp"
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        tmp.replace(self.dir / name)

    def reload_knowledge(self) -> None:
        self.knowledge = self._load("knowledge.json", {})

    # ---- learning -------------------------------------------------------------------------------------------
    def update_profile(self, **fields: Any) -> None:
        with self._lock:
            self.profile.update(fields)
            self._save("profile.json", self.profile)

    def remember_ui(self, obs: Observation) -> None:
        """Store labelled controls with window-relative positions, so a later run can recognize the layout."""
        win = obs.active_window
        if not win or win.geometry[2] <= 0:
            return
        wx, wy, ww, wh = win.geometry
        with self._lock:
            store = self.ui.setdefault("elements", {})
            now = time.time()
            for e in obs.elements:
                if not e.text or e.type in ("text",) and e.source != "accessibility":
                    continue
                key = f"{e.type}:{e.text[:60]}"
                cx, cy = e.center
                item = store.get(key) or {"type": e.type, "text": e.text[:60], "seen": 0, "source": e.source}
                item["rel"] = [round((cx - wx) / ww, 4), round((cy - wy) / wh, 4)]
                item["seen"] = item.get("seen", 0) + 1
                item["last_seen"] = now
                store[key] = item
            if len(store) > MAX_UI:
                for k, _ in sorted(store.items(), key=lambda kv: kv[1].get("last_seen", 0))[: len(store) - MAX_UI]:
                    store.pop(k, None)
            self.ui["window_title"] = win.title
            self._save("ui_memory.json", self.ui)

    def record_success(self, goal: str, actions: list[dict[str, Any]], duration: float) -> None:
        compact = [{k: v for k, v in a.items() if k in ("action", "text", "keys", "target_label", "app", "amount")}
                   for a in actions]
        with self._lock:
            self.successes.append({"goal": goal, "actions": compact, "duration": round(duration, 1), "ts": time.time()})
            self.successes = self.successes[-MAX_SUCCESS:]
            self._save("successful_actions.json", self.successes)
            self.profile["successes"] = self.profile.get("successes", 0) + 1
            self.profile["tasks"] = self.profile.get("tasks", 0) + 1
            self._save("profile.json", self.profile)
            # learn: remember keyboard shortcuts that were part of a successful run
            learned = self.knowledge.setdefault("learned_shortcuts", [])
            for a in compact:
                if a.get("action") == "hotkey" and a.get("keys"):
                    combo = "+".join(a["keys"]).lower()
                    if combo not in learned:
                        learned.append(combo)
            self._save("knowledge.json", self.knowledge)

    def record_failure(self, goal: str, action: dict[str, Any], reason: str) -> None:
        with self._lock:
            self.failures.append({"goal": goal, "action": action, "reason": reason[:300], "ts": time.time()})
            self.failures = self.failures[-MAX_FAIL:]
            self._save("failed_patterns.json", self.failures)

    def record_task_failure(self) -> None:
        with self._lock:
            self.profile["failures"] = self.profile.get("failures", 0) + 1
            self.profile["tasks"] = self.profile.get("tasks", 0) + 1
            self._save("profile.json", self.profile)

    # ---- recall ---------------------------------------------------------------------------------------------
    def hints(self, goal: str, max_workflows: int = 2) -> str:
        """Compact text block injected into the planner prompt."""
        k = self.knowledge
        lines: list[str] = []
        if k.get("description"):
            lines.append(str(k["description"]))
        for tip in k.get("tips", [])[:12]:
            lines.append(f"- {tip}")
        shortcuts = k.get("shortcuts") or {}
        if shortcuts:
            lines.append("Shortcuts: " + "; ".join(f"{v} = {key}" for key, v in list(shortcuts.items())[:20]))
        if k.get("learned_shortcuts"):
            lines.append("Shortcuts that worked before: " + ", ".join(k["learned_shortcuts"][:10]))
        ranked = sorted(self.successes, key=lambda s: _similarity(goal, s["goal"]), reverse=True)
        for s in ranked[:max_workflows]:
            if _similarity(goal, s["goal"]) < 0.25:
                break
            steps = " -> ".join(
                a["action"] + (f" {'+'.join(a['keys'])}" if a.get("keys") else "")
                + (f' "{a["text"][:30]}"' if a.get("text") else "")
                + (f" [{a['target_label']}]" if a.get("target_label") else "")
                for a in s["actions"][:14]
            )
            lines.append(f'A similar goal ("{s["goal"][:80]}") succeeded with: {steps}')
        recent_fail = [f for f in self.failures[-30:] if _similarity(goal, f["goal"]) >= 0.25][-3:]
        for f in recent_fail:
            lines.append(f"Avoid: {f['action'].get('action')} {f['action'].get('target_label', '')} failed before "
                         f"({f['reason'][:80]})")
        return "\n".join(lines)
