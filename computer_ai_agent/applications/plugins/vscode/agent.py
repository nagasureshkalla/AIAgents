"""Visual Studio Code sub-agent (Electron: accessibility must be forced on at launch)."""

from __future__ import annotations

from agents.application_agent import GenericApplicationAgent
from core.types import Observation

APP_KEYS = ["code", "code_code", "com.visualstudio.code", "code-oss", "visual studio code", "vscode"]


class VSCodeSubAgent(GenericApplicationAgent):
    def launch_args(self) -> list[str]:
        return ["--force-renderer-accessibility", "--new-window"]

    def describe_state(self, obs: Observation) -> str:
        if obs.active_window and " - " in obs.active_window.title:
            parts = obs.active_window.title.split(" - ")
            return f"Open editor/workspace (from title): {' | '.join(parts[:-1])}"
        return ""


AGENT = VSCodeSubAgent
