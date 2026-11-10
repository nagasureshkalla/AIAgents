"""Agent harnesses: the editable prompt surface each agent overlays on its built-in prompts.

A harness is a directory in google-research/harness2's layout, so harness2 can improve it from evidence
(see ../GoogleHrness, `python -m backend.agents improve multi_ai_agent/<agent>`):

    harness/<agent>/systemprompt.md                  appended to every system prompt of the agent
    harness/<agent>/guardrails.md                    rules, appended after it (skill trigger lines are dropped)
    harness/<agent>/LongTermMEMORY.md                lessons kept across tasks
    harness/<agent>/tool_descriptions/tool_guidance.md
    harness/<agent>/skills/<mode>/SKILL.md           replaces the built-in system prompt for that mode

Every file is optional and a missing directory means "built-in prompts only". Files are re-read when they change,
so an adopted harness takes effect without a restart.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

ALWAYS_ON = ("systemprompt.md", "guardrails.md", "LongTermMEMORY.md", "tool_descriptions/tool_guidance.md")
_FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n?", re.S)
_TRIGGER = re.compile(r"^\s*-\s.*load the `[^`]+` skill\.?\s*$", re.M)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    body: str


@dataclass(frozen=True)
class Harness:
    system: str = ""
    guardrails: str = ""
    memory: str = ""
    tool_guidance: str = ""
    skills: dict[str, Skill] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return not (self.system or self.guardrails or self.memory or self.tool_guidance or self.skills)

    def compose(self, default: str, skill: str | None = None) -> str:
        """The system prompt for one call: the mode's skill (or `default`), then the always-on text."""
        chosen = self.skills.get(skill) if skill else None
        parts = [chosen.body if chosen else default]
        if self.system:
            parts.append(self.system)
        if self.guardrails:
            parts.append("Rules:\n" + self.guardrails)
        if self.tool_guidance:
            parts.append(self.tool_guidance)
        if self.memory:
            parts.append("Lessons from earlier tasks:\n" + self.memory)
        return "\n\n".join(p.strip() for p in parts if p and p.strip())


def parse_skill(name: str, text: str) -> Skill:
    m = _FRONTMATTER.match(text)
    meta, body = (m.group(1), text[m.end():]) if m else ("", text)
    desc = re.search(r"^description:\s*(.+)$", meta, re.M)
    return Skill(name=name, description=desc.group(1).strip() if desc else "", body=body.strip())


def load_harness(d: Path) -> Harness:
    def read(rel: str) -> str:
        p = d / rel
        return p.read_text(encoding="utf-8").strip() if p.is_file() else ""

    skills = {
        p.parent.name: parse_skill(p.parent.name, p.read_text(encoding="utf-8"))
        for p in sorted(d.glob("skills/*/SKILL.md"))
    }
    return Harness(
        system=read("systemprompt.md"),
        guardrails=_TRIGGER.sub("", read("guardrails.md")).strip(),
        memory=read("LongTermMEMORY.md"),
        tool_guidance=read("tool_descriptions/tool_guidance.md"),
        skills=skills,
    )


def _signature(d: Path) -> tuple:
    if not d.is_dir():
        return ()
    return tuple(sorted((str(p), p.stat().st_mtime_ns) for p in d.rglob("*.md")))


class HarnessStore:
    """Per-agent harness cache, reloaded when any file in the agent's directory changes."""

    def __init__(self, root: Path, overrides: dict[str, Path] | None = None) -> None:
        self.root = root
        self.overrides = dict(overrides or {})
        self._cache: dict[str, tuple[tuple, Harness]] = {}
        self._lock = threading.Lock()

    def dir_for(self, agent: str) -> Path:
        return self.overrides.get(agent) or self.root / agent

    def get(self, agent: str) -> Harness:
        d = self.dir_for(agent)
        sig = _signature(d)
        with self._lock:
            hit = self._cache.get(agent)
            if hit and hit[0] == sig:
                return hit[1]
            harness = load_harness(d) if sig else Harness()
            self._cache[agent] = (sig, harness)
            return harness


def seed_tree(agent: str, prompts: dict[str, str], descriptions: dict[str, str] | None = None) -> dict[str, str]:
    """A harness tree that reproduces an agent's built-in prompts exactly (the base harness for harness2)."""
    descriptions = descriptions or {}
    tree = {name: "" for name in ALWAYS_ON}
    triggers = []
    for mode, prompt in prompts.items():
        desc = descriptions.get(mode) or f"the {agent} agent runs in {mode} mode"
        tree[f"skills/{mode}/SKILL.md"] = f"---\nname: {mode}\ndescription: {desc}\n---\n\n{prompt.strip()}\n"
        triggers.append(f"- When {desc}, load the `{mode}` skill.")
    tree["guardrails.md"] = "\n".join(triggers) + "\n" if triggers else ""
    return tree


def write_tree(tree: dict[str, str], d: Path) -> None:
    for rel, text in tree.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
