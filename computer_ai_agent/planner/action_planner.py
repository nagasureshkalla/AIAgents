"""Action planner: (goal, current observation, history) -> the next action, optionally followed by up to 3
keyboard-only steps that need no new look at the screen (e.g. ctrl+l, then type the URL).

Every action is still verified individually; a chain stops at the first problem.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from core.jsonutil import extract_json
from core.types import Action, Observation
from models.llm import Generation, LocalLLMProvider
from planner.harness import PlannerHarness

logger = logging.getLogger(__name__)

# The system prompt only changes per application, so Ollama can reuse its cached prefix from step to step.
SYSTEM = """You operate the application "{app}" on a Linux desktop to accomplish a goal.
After your actions you get a fresh observation of the screen.

Observation: UI elements as "[id] type "text" at (x,y,width x height) [states] <source>".
Sources: accessibility (exact), ocr (text read from pixels), vision (detected boxes, least reliable).
A screenshot is attached only sometimes; numbered boxes on it match the element ids.

Actions (JSON fields):
- click / double_click / right_click / mouse_move: "element": id   (or "x","y" only if no element fits)
- type_text: "text": "...", optional "element": id to focus first. A final "\\n" presses Enter. Long or multi-line
  text is pasted exactly as given: it is never cut off, so never type the rest of it again.
- key_press: "keys": ["Enter"]  (keys pressed one after another: Enter, Tab, Escape, Down, PageDown, F5, BackSpace...)
- hotkey: "keys": ["ctrl", "l"]  (pressed together)
- scroll: "amount": 5 (positive = down, negative = up), optional "element": id
- select_text: optional "element" (selects its content; without element = select all, like ctrl+a)
- mouse_drag: "element": id and "target": id
- wait: "amount": seconds (for loading)
- open_application / switch_application / close_application: "app": "name"
- done: the goal is achieved AND visible now. Put evidence in "reason" and any answer the user asked for in "text".
- fail: the goal is impossible here; explain in "reason".
- ask_user: genuinely unsure which option the user means: "question" and "choices": [...]

Reply with ONE JSON object only:
{{"reason": "<max 15 words>", "action": "...", <fields>, "expect": "<max 10 words>", "then": [<optional>]}}
"then" may hold up to 3 more keyboard-only actions (hotkey, key_press, type_text without element, wait) that make
sense without looking at the screen again, e.g. {{"action": "hotkey", "keys": ["ctrl","l"], "then": [{{"action":
"type_text", "text": "example.com\\n"}}]}}. Each is verified; the chain stops if one fails.

Rules:
- Use element ids from the list only; prefer keyboard shortcuts and typing over long click sequences.
- Check the result of your previous action first. If something had no effect, do NOT repeat it: use another way
  (keyboard shortcut, typing a path, a different element).
- A text field must be focused before typing (click it, or use the app's focus shortcut).
- File Save/Open dialogs: put the FULL path in the file name field (select_text on it, then type e.g.
  "~/Desktop/notes.json\\n"). Do not click sidebar places like Desktop/Home to change folders.
- Browsers: never navigate away from a tab titled "Agent Control Room" or "Computer Agent" (the user's control
  panel): open a new tab with ctrl+t first.
- Never type passwords or secrets; never bypass logins, CAPTCHAs or security prompts: use ask_user.
- Handle unexpected dialogs/popups first (dismiss them if unrelated to the goal).
- Do not declare done until the observation shows the goal is complete.
{knowledge}"""

POINTER_ACTIONS = {"click", "double_click", "right_click", "mouse_move", "mouse_drag", "drag_and_drop"}
CHAINABLE = {"hotkey", "key_press", "type_text", "wait"}
MAX_CHAIN = 3
# "action": "select_text": "element": 5  ->  "action": "select_text", "element": 5
_BROKEN_PAIR = re.compile(r'("action"\s*:\s*"[a-z_]+")\s*:\s*')


@dataclass
class StepRecord:
    step: int
    action: dict[str, Any]
    ok: bool
    outcome: str                   # what happened (engine message + verification)
    reason: str = ""


@dataclass
class PlannerDecision:
    action: Action | None
    raw: Any
    generation: Generation | None
    error: str = ""
    used_image: bool = False
    followups: list[Action] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


def parse_reply(text: str) -> Any:
    """extract_json plus repairs for mistakes the model makes repeatedly."""
    data = extract_json(text)
    if data is None and text:
        data = extract_json(_BROKEN_PAIR.sub(r"\1, ", text))
    return data


class ActionPlanner:
    def __init__(self, llm: LocalLLMProvider, max_elements: int = 100, harness_dir: Path | None = None) -> None:
        self.llm = llm
        self.max_elements = max_elements
        self.harness = PlannerHarness(harness_dir)

    def system_prompt(self, app: str, hints: str = "") -> str:
        system = SYSTEM.format(app=app, knowledge=f"\nWhat you know about {app}:\n{hints}" if hints else "")
        extra = self.harness.extra(app)
        return f"{system}\n\n{extra}" if extra else system

    def build_prompt(self, app: str, goal: str, obs: Observation, history: list[StepRecord],
                     checkpoints: list[str] | None = None, context: str = "", note: str = "",
                     app_state: str = "") -> str:
        # Stable parts first (goal, checkpoints), then what changes every step.
        parts = [f"GOAL: {goal}"]
        if checkpoints:
            parts.append("Checkpoints to reach in order: " + " -> ".join(checkpoints))
        if context:
            parts.append(f"Context from earlier steps of the overall task: {context}")
        if history:
            parts.append("Your previous actions (oldest first):")
            for h in history[-10:]:
                a = h.action
                desc = a.get("action", "?")
                for k in ("target_label", "text", "keys", "app", "amount"):
                    if a.get(k):
                        v = "+".join(a[k]) if k == "keys" else a[k]
                        desc += f" {k}={str(v)[:60]!r}"
                parts.append(f"  {h.step}. {desc} -> {'OK' if h.ok else 'PROBLEM'}: {h.outcome[:180]}")
        else:
            parts.append("No actions taken yet.")
        if note:
            parts.append(f"NOTE: {note}")
        if app_state:
            parts.append(app_state)
        parts.append("CURRENT OBSERVATION:\n" + obs.summary(self.max_elements, text_limit=1500))
        parts.append("Next action? JSON only.")
        return "\n".join(parts)

    async def next_action(self, app: str, goal: str, obs: Observation, history: list[StepRecord], *,
                          hints: str = "", checkpoints: list[str] | None = None, context: str = "", note: str = "",
                          image: Image.Image | None = None, app_state: str = "") -> PlannerDecision:
        system = self.system_prompt(app, hints)
        user = self.build_prompt(app, goal, obs, history, checkpoints, context, note, app_state)
        imgs = [image] if image is not None else None
        gen: Generation | None = None
        data: Any = None
        for attempt in range(2):  # an empty/garbled reply is retried once right away, without a new observation
            try:
                gen = await self.llm.ask(system, user, imgs, max_tokens=300, temperature=0.1 if attempt else None)
            except Exception as exc:  # noqa: BLE001
                return PlannerDecision(None, None, None, f"model error: {exc}")
            data = parse_reply(gen.text)
            if isinstance(data, list) and data:
                data = {**data[0], "then": data[1:]}  # a bare list = first action + chain
            if isinstance(data, dict):
                break
        if not isinstance(data, dict):
            return PlannerDecision(None, (gen.text if gen else "")[:400], gen, "model reply was not a JSON object",
                                   used_image=image is not None)
        try:
            action = Action.from_dict(data)
        except ValueError as exc:
            return PlannerDecision(None, data, gen, str(exc), used_image=image is not None)
        if action.action in POINTER_ACTIONS and action.element is None and (action.x is None or action.y is None):
            return PlannerDecision(None, data, gen, f'"{action.action}" needs "element": <id from the list> (or x and y)',
                                   used_image=image is not None)
        if action.element is not None and obs.element(action.element) is None and action.action not in (
                "done", "fail", "ask_user", "type_text", "scroll"):
            return PlannerDecision(None, data, gen, f"element {action.element} does not exist in the observation",
                                   used_image=image is not None)
        followups: list[Action] = []
        if action.action not in ("done", "fail", "ask_user"):
            for item in (data.get("then") or [])[:MAX_CHAIN]:
                try:
                    f = Action.from_dict(item) if isinstance(item, dict) else None
                except ValueError:
                    f = None
                # only keyboard steps: element ids would refer to a screen that no longer exists
                if f is None or f.action not in CHAINABLE or f.element is not None or f.x is not None:
                    break
                followups.append(f)
        return PlannerDecision(action, data, gen, used_image=image is not None, followups=followups)
