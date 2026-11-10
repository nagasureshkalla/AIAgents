"""Verifier: did the last action have an effect, and is the goal really achieved?"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from PIL import Image

from core.types import Action, ActionResult, Observation
from models.llm import LocalLLMProvider
from vision.screen_capture import diff_ratio

logger = logging.getLogger(__name__)

# Actions after which *something* on screen should change.
EXPECTS_CHANGE = {"click", "double_click", "right_click", "type_text", "key_press", "hotkey", "scroll",
                  "select_text", "paste", "mouse_drag", "drag_and_drop", "open_application", "switch_application",
                  "close_application"}

GOAL_SYSTEM = """You verify whether a desktop automation goal has been completed, using the current screen.
Be strict: every part of the goal must be satisfied by what is visible NOW. Examples: "open the first search result"
requires the page of the result the agent clicked (its title/URL must match that result, not some other page on the
same site such as a cookie policy or home page); "save the file" requires no unsaved marker; "search for X" requires
results for X. If a popup/dialog covers the page it is still complete when the right page is behind it.
Reply with JSON only (evidence max 30 words): {"complete": true|false, "evidence": "<what on screen proves it, or
what is missing>", "answer": "<the information the user asked for, if any, else empty>"}"""


@dataclass
class Verification:
    status: str          # ok | no_change | failed
    detail: str
    changed: float       # fraction of screen that changed
    data: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class Verifier:
    def __init__(self, llm: LocalLLMProvider | None = None, change_threshold: float = 0.0015) -> None:
        self.llm = llm
        self.change_threshold = change_threshold

    def check_action(self, action: Action, result: ActionResult, before: Observation | None,
                     after: Observation) -> Verification:
        if not result.ok:
            return Verification("failed", result.message, 0.0, {})
        changed = diff_ratio(before.image if before else None, after.image)
        title_before = before.active_window.title if before and before.active_window else ""
        title_after = after.active_window.title if after.active_window else ""
        win_changed = (before.active_window.id if before and before.active_window else None) != (
            after.active_window.id if after.active_window else None)
        focus_before = before.focused.describe() if before and before.focused else ""
        focus_after = after.focused.describe() if after.focused else ""
        signals = []
        if title_before != title_after:
            signals.append(f'title "{title_before[:50]}" -> "{title_after[:50]}"')
        if win_changed:
            signals.append("active window changed")
        if focus_before != focus_after and focus_after:
            signals.append(f"focus moved to {after.focused.type} \"{after.focused.text[:40]}\"")
        if changed >= self.change_threshold:
            signals.append(f"{changed * 100:.1f}% of the screen changed")
        data = {"changed": round(changed, 4), "title": title_after, "signals": signals}

        if action.action == "type_text" and after.focused is not None and after.focused.text:
            typed = action.text.strip().splitlines()[0][:20].lower() if action.text.strip() else ""
            if typed and typed in after.focused.text.lower():
                return Verification("ok", "text is in the focused field", changed, data)
        if action.action in EXPECTS_CHANGE and not signals:
            return Verification("no_change", "no visible change after the action", changed, data)
        return Verification("ok", "; ".join(signals) or result.message, changed, data)

    async def check_goal(self, goal: str, obs: Observation, image: Image.Image | None,
                         claimed_evidence: str = "", actions_taken: str = "") -> tuple[bool, str, str]:
        """Independent second look before reporting success. Returns (complete, evidence, answer)."""
        if self.llm is None:
            return True, claimed_evidence, ""
        user = (f"GOAL: {goal}\nActions the agent took: {actions_taken or '(none)'}\n"
                f"The agent claims it is done because: {claimed_evidence or '(no reason given)'}\n"
                f"CURRENT SCREEN (text fields shown in full):\n{obs.summary(120, text_limit=3000)}")
        try:
            gen = await self.llm.ask(GOAL_SYSTEM, user, [image] if image is not None else None, max_tokens=450,
                                     temperature=0.0)
        except Exception as exc:  # noqa: BLE001 - verification model down: trust the agent but say so
            return True, f"(goal check skipped: {exc})", ""
        from planner.action_planner import parse_reply

        data = parse_reply(gen.text)
        if not isinstance(data, dict):
            return True, "(goal check unparseable; trusting agent)", ""
        return bool(data.get("complete")), str(data.get("evidence", ""))[:400], str(data.get("answer", "") or "")
