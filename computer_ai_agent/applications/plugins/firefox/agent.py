"""Firefox sub-agent."""

from __future__ import annotations

import re

from agents.application_agent import GenericApplicationAgent
from core.types import Action, Observation

CONTROL_UIS = re.compile(r"Agent Control Room|Computer Agent|127\.0\.0\.1:(8000|8765)|^DevTools", re.I)
APP_KEYS = ["firefox", "firefox_firefox", "org.mozilla.firefox", "mozilla firefox"]


class FirefoxSubAgent(GenericApplicationAgent):
    def launch_env(self) -> dict[str, str]:
        return {"GNOME_ACCESSIBILITY": "1"}

    def describe_state(self, obs: Observation) -> str:
        for e in obs.elements:
            if e.type in ("textbox", "combobox") and re.search(r"search with|enter address|url", e.text, re.I):
                return f"Address bar element [{e.id}]: {e.text[:150]}"
        return ""


    def preparation(self, obs: Observation) -> list[Action]:
        """If the tab in front is the user's own control panel (or DevTools), open a new tab first: no model call."""
        title = obs.active_window.title if obs.active_window else ""
        if CONTROL_UIS.search(title):
            return [Action("hotkey", keys=["ctrl", "t"], reason="opened a new tab (keeping your control panel tab)")]
        return []


AGENT = FirefoxSubAgent
