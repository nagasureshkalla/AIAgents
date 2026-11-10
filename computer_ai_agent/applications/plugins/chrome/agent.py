"""Chrome / Chromium sub-agent: launch with accessibility on, read the address bar, browser knowledge."""

from __future__ import annotations

import os
import re

from agents.application_agent import GenericApplicationAgent
from core.types import Action, Observation

CONTROL_UIS = re.compile(r"Agent Control Room|Computer Agent|127\.0\.0\.1:(8000|8765)|^DevTools", re.I)
APP_KEYS = ["google-chrome", "chrome", "chromium", "chromium-browser", "chromium_chromium", "google chrome"]
ADDRESS_BAR = re.compile(r"address and search bar|address bar|search or type", re.I)


class ChromeSubAgent(GenericApplicationAgent):
    def launch_args(self) -> list[str]:
        # Chrome only builds its accessibility tree when asked; without it we fall back to OCR for web pages.
        args = ["--force-renderer-accessibility", "--no-first-run", "--no-default-browser-check"]
        profile = os.getenv("CHROME_USER_DATA_DIR", "")
        session_display = os.getenv("CUA_SESSION_DISPLAY", "")
        target = self.rt.settings.display
        if not profile and session_display and target and target != session_display:
            # Chrome is single-instance per profile: on a different X display (e.g. a nested test screen) a
            # separate profile is required, otherwise the window opens on the user's main screen instead.
            profile = str(self.rt.settings.data_dir / "chrome-profiles" / re.sub(r"\W", "_", target))
        if profile:
            args.append(f"--user-data-dir={profile}")
        return args

    def describe_state(self, obs: Observation) -> str:
        bits = []
        for e in obs.elements:
            if e.type in ("textbox", "combobox") and (ADDRESS_BAR.search(e.text) or e.text.startswith(("http", "www."))):
                bits.append(f"Address bar element [{e.id}] currently shows: {e.text[:150]}")
                break
        if obs.active_window and obs.active_window.title:
            bits.append(f"Page title: {re.sub(r' - (Google Chrome|Chromium)$', '', obs.active_window.title)}")
        return "\n".join(bits)


    def preparation(self, obs: Observation) -> list[Action]:
        """If the tab in front is the user's own control panel (or DevTools), open a new tab first: no model call."""
        title = obs.active_window.title if obs.active_window else ""
        if CONTROL_UIS.search(title):
            return [Action("hotkey", keys=["ctrl", "t"], reason="opened a new tab (keeping your control panel tab)")]
        return []


AGENT = ChromeSubAgent
