"""Calculator sub-agent."""

from __future__ import annotations

from agents.application_agent import GenericApplicationAgent
from core.types import Observation

APP_KEYS = ["org.gnome.calculator", "gnome-calculator", "calculator", "org.kde.kcalc", "kcalc"]


class CalculatorSubAgent(GenericApplicationAgent):
    def describe_state(self, obs: Observation) -> str:
        fields = [e for e in obs.elements if e.type == "textbox" and e.text]
        if fields:
            return "Calculator display: " + " | ".join(f"[{e.id}] {e.text[:60]}" for e in fields[:3])
        return ""


AGENT = CalculatorSubAgent
