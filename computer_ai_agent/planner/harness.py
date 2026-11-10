"""The action planner's harness: editable guidance appended to the fixed SYSTEM prompt.

The directory uses google-research/harness2's layout, so harness2 can improve it from evidence
(../GoogleHrness: `python -m backend.agents improve computer_ai_agent`):

    harness/systemprompt.md                    general guidance
    harness/guardrails.md                      extra rules (skill trigger lines are dropped)
    harness/LongTermMEMORY.md                  lessons kept across tasks
    harness/tool_descriptions/tool_guidance.md how to use the actions well
    harness/skills/<app>/SKILL.md              guidance only for that application

It is append-only on purpose: the action protocol in SYSTEM stays in code, so a harness edit cannot break JSON
parsing. The text is static per app, which keeps Ollama's prefix cache valid from step to step. Files are re-read
when they change.
"""

from __future__ import annotations

import re
import threading
from pathlib import Path

_FRONTMATTER = re.compile(r"\A---\n.*?\n---\n?", re.S)
_TRIGGER = re.compile(r"^\s*-\s.*load the `[^`]+` skill\.?\s*$", re.M)


def _read(p: Path) -> str:
    return p.read_text(encoding="utf-8").strip() if p.is_file() else ""


class PlannerHarness:
    def __init__(self, root: Path | None) -> None:
        self.root = root
        self._sig: tuple = ()
        self._parts: dict[str, str] = {}
        self._skills: dict[str, str] = {}
        self._lock = threading.Lock()

    def _signature(self) -> tuple:
        if self.root is None or not self.root.is_dir():
            return ()
        return tuple(sorted((str(p), p.stat().st_mtime_ns) for p in self.root.rglob("*.md") if "tasks" not in p.parts))

    def _refresh(self) -> None:
        sig = self._signature()
        if sig == self._sig and (sig or not self._parts):
            return
        root = self.root or Path()
        self._parts = {
            "system": _read(root / "systemprompt.md"),
            "rules": _TRIGGER.sub("", _read(root / "guardrails.md")).strip(),
            "tools": _read(root / "tool_descriptions/tool_guidance.md"),
            "memory": _read(root / "LongTermMEMORY.md"),
        }
        self._skills = {
            p.parent.name.lower(): _FRONTMATTER.sub("", p.read_text(encoding="utf-8")).strip()
            for p in sorted(root.glob("skills/*/SKILL.md"))
        } if sig else {}
        self._sig = sig

    def extra(self, app: str) -> str:
        """Text appended to SYSTEM for `app`; empty when there is no harness."""
        with self._lock:
            self._refresh()
            parts, skills = dict(self._parts), dict(self._skills)
        key = re.sub(r"[^a-z0-9_-]+", "-", app.lower()).strip("-")
        out = []
        if parts.get("system"):
            out.append(parts["system"])
        if parts.get("rules"):
            out.append("More rules:\n" + parts["rules"])
        if parts.get("tools"):
            out.append(parts["tools"])
        if skills.get(key):
            out.append(f"Guidance for {app}:\n{skills[key]}")
        if parts.get("memory"):
            out.append("Lessons from earlier tasks:\n" + parts["memory"])
        return "\n\n".join(out)
