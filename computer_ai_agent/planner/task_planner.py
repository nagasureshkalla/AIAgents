"""Task planner: natural-language command -> ordered subtasks, each owned by one application's sub-agent."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from models.llm import LocalLLMProvider

logger = logging.getLogger(__name__)


@dataclass
class Subtask:
    id: int
    app: str                       # application name as the user would say it ("chrome", "vs code", "files")
    instruction: str               # self-contained goal for the sub-agent; "" = just open/focus the app
    expected_result: str = ""
    uses_previous: bool = False    # needs output/context from earlier subtasks
    steps: list[str] = field(default_factory=list)  # human-readable checkpoints (for the UI and the sub-agent)
    close: bool = False            # quit the app: done by the window manager and verified, never by clicking

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM = """You are the planning module of a desktop automation agent on Linux.
Split the user's command into ordered subtasks. Each subtask is performed inside exactly ONE desktop application.

Rules:
- "app": the application name as a person would say it (e.g. "chrome", "firefox", "vs code", "terminal", "files",
  "calculator", "text editor", "blender", "spotify"). If the user names no app but the task needs a website, use "chrome".
- "instruction": a complete, self-contained goal for that app, including every concrete value from the command
  (URLs, search terms, file names, numbers, code). Do not describe mouse clicks. Use "" if the app only needs to be opened.
- "steps": 2-8 short observable checkpoints in order (e.g. "Google homepage is shown", "results for X are listed").
- "expected_result": what is visible on screen when the subtask is done.
- "uses_previous": true when the subtask needs something produced by an earlier subtask (a downloaded file, copied text...).
- Keep the number of subtasks minimal: consecutive work in the same app is ONE subtask.
Reply with JSON only:
{"subtasks": [{"app": "...", "instruction": "...", "steps": ["..."], "expected_result": "...", "uses_previous": false}]}"""

OPEN_ONLY = re.compile(r"^\s*(?:please\s+)?(?:open|launch|start|run)\s+(?:the\s+|my\s+)?([\w .+-]+?)(?:\s+app(?:lication)?)?\s*[.!]?\s*$",
                       re.I)
# "Close Calculator", "can you close the calculator app?", "please quit gedit": the whole command is closing one app.
CLOSE_ONLY = re.compile(r"^\s*(?:(?:hey|ok|okay)\b[\s,]*)?(?:(?:can|could|would|will)\s+you\s+|please\s+)*"
                        r"(?:close|quit|exit|shut\s+down)\s+(?:the\s+|my\s+)?([\w .+-]+?)"
                        r"(?:\s+(?:app(?:lication)?|program))?(?:\s+please)?\s*[.!?]?\s*$", re.I)
# A planned subtask whose whole instruction is closing its app ("Close the calculator application.").
CLOSE_INSTRUCTION = re.compile(r"^\s*(?:close|quit|exit)\s+(?:the\s+)?(?:[\w.+-]+\s+){0,3}?"
                               r"(?:app(?:lication)?|program)?\s*[.!]?\s*$", re.I)
# What closing a *part* of an app looks like: never treated as quitting the app.
NOT_THE_APP = re.compile(r"\b(?:tab|tabs|file|document|dialog|popup|pop-up|menu|panel|sidebar|terminal tab|"
                         r"window of|page|folder|project|and|then|in|on|from)\b", re.I)
SPLIT = re.compile(r",?\s*\b(?:and\s+then|then|after\s+that|afterwards)\b\s*,?\s*", re.I)
OPEN_AND = re.compile(r"^\s*(?:please\s+)?(?:open|launch|start)\s+(?:the\s+|my\s+)?((?:[\w.+-]+\s+){0,2}?[\w.+-]+)"
                      r"(?:\s+app(?:lication)?)?\s*(?:,|\band\b)\s*(.+)$", re.I)


class TaskPlanner:
    def __init__(self, llm: LocalLLMProvider | None) -> None:
        self.llm = llm

    @staticmethod
    def quick_plan(command: str) -> list[Subtask] | None:
        """No model needed for "Open X." style commands (a short app name and nothing else to do)."""
        m = OPEN_ONLY.match(command)
        if m and len(m.group(1).split()) <= 4 and not re.search(r"\b(and|then|to|for|with|in)\b|,", m.group(1), re.I):
            return [Subtask(1, m.group(1).strip(), "", f"{m.group(1).strip()} is open and focused")]
        return None

    @staticmethod
    def close_plan(command: str) -> list[Subtask] | None:
        """No model for "close X": the master closes the app's windows itself and checks they are gone."""
        m = CLOSE_ONLY.match(command)
        if not m or len(m.group(1).split()) > 4 or NOT_THE_APP.search(m.group(1)):
            return None
        app = m.group(1).strip()
        return [Subtask(1, app, f"Close {app}", f"{app} has no open window", close=True)]

    @staticmethod
    def single_app_plan(command: str) -> list[Subtask] | None:
        """"Open X and/, <do something>" with one app and no second step: no model call needed (~6-8 s saved).
        The sub-agent gets the whole command as its goal."""
        m = OPEN_AND.match(command)
        if not m:
            return None
        rest = m.group(2)
        if SPLIT.search(rest) or re.search(r"\b(?:open|launch|start|switch to|go back to)\s+(?:the\s+|my\s+)?"
                                           r"(?!(?:the\s+)?(?:first|second|top|new|link|result|file|folder|tab|page|"
                                           r"menu|dialog|settings|website|site|url))\w", rest, re.I):
            return None  # several apps / steps: let the planner split it
        app = m.group(1).strip()
        return [Subtask(1, app, command.strip().rstrip("."), f"the request is done in {app}")]

    @staticmethod
    def rule_plan(command: str) -> list[Subtask]:
        """Fallback when the model is unavailable: split on 'then', read 'open X and ...' per part."""
        parts = [p for p in SPLIT.split(command) if p.strip()]
        out: list[Subtask] = []
        last_app = ""
        for part in parts:
            m = OPEN_AND.match(part)
            if m:
                last_app = m.group(1).strip()
                out.append(Subtask(len(out) + 1, last_app, m.group(2).strip().rstrip(".")))
                continue
            m = OPEN_ONLY.match(part)
            if m:
                last_app = m.group(1).strip()
                out.append(Subtask(len(out) + 1, last_app, ""))
                continue
            if out and out[-1].app == last_app and last_app:
                prev = out[-1]
                prev.instruction = (prev.instruction + ", then " if prev.instruction else "") + part.strip().rstrip(".")
            else:
                out.append(Subtask(len(out) + 1, last_app or "chrome", part.strip().rstrip("."), uses_previous=bool(out)))
        return out or [Subtask(1, "chrome", command)]

    async def plan(self, command: str, installed_apps: list[str] | None = None,
                   previous_workflows: list[dict[str, Any]] | None = None) -> tuple[list[Subtask], str]:
        """Returns (subtasks, source) where source is quick | llm | rules."""
        quick = self.quick_plan(command) or self.close_plan(command)
        if quick:
            return quick, "quick"
        single = self.single_app_plan(command)
        if single:
            return single, "single-app"
        if self.llm is None:
            return self.rule_plan(command), "rules"
        user = f"Command: {command}"
        if installed_apps:
            user += "\nSome installed applications: " + ", ".join(installed_apps[:60])
        for wf in previous_workflows or []:
            subs = "; ".join(f"[{s.get('app')}] {s.get('instruction')}" for s in wf.get("subtasks", []))
            user += f'\nA similar earlier command "{wf.get("user_command")}" was planned as: {subs}'
        try:
            gen = await self.llm.ask(SYSTEM, user, max_tokens=900, temperature=0.0)
            data = gen.json()
        except Exception as exc:  # noqa: BLE001 - planner model down: degrade to rules
            logger.warning("task planner model failed (%s); using rules", exc)
            return self.rule_plan(command), "rules"
        items = data.get("subtasks") if isinstance(data, dict) else data if isinstance(data, list) else None
        subtasks: list[Subtask] = []
        for i, it in enumerate(items or [], start=1):
            if not isinstance(it, dict) or not str(it.get("app", "")).strip():
                continue
            steps = it.get("steps") if isinstance(it.get("steps"), list) else []
            subtasks.append(Subtask(
                id=i,
                app=str(it["app"]).strip(),
                instruction=str(it.get("instruction") or "").strip(),
                expected_result=str(it.get("expected_result") or "").strip(),
                uses_previous=bool(it.get("uses_previous")),
                steps=[str(s) for s in steps][:10],
            ))
            instr = subtasks[-1].instruction
            subtasks[-1].close = bool(CLOSE_INSTRUCTION.match(instr)) and not NOT_THE_APP.search(instr)
        if not subtasks:
            logger.warning("task planner returned nothing usable: %s", str(data)[:200])
            return self.rule_plan(command), "rules"
        return subtasks, "llm"
