"""Terminal sub-agent: commands are typed; output is read through the accessibility text of the terminal."""

from __future__ import annotations

from agents.application_agent import GenericApplicationAgent
from core.types import Observation

APP_KEYS = ["org.gnome.terminal", "gnome-terminal", "terminal", "org.gnome.console", "kgx", "org.gnome.ptyxis",
            "xterm", "konsole"]


class TerminalSubAgent(GenericApplicationAgent):
    def describe_state(self, obs: Observation) -> str:
        for e in obs.elements:
            if e.type == "terminal" and e.text:
                tail = "\n".join(e.text.strip().splitlines()[-25:])
                return f"Terminal output (last lines):\n{tail}"
        return ""


AGENT = TerminalSubAgent
