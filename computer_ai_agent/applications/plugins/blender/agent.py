"""Blender sub-agent: no accessibility tree, so the agent relies on OCR, the screenshot and shortcuts."""

from __future__ import annotations

from agents.application_agent import GenericApplicationAgent

APP_KEYS = ["blender", "blender_blender", "org.blender.blender"]


class BlenderSubAgent(GenericApplicationAgent):
    pass


AGENT = BlenderSubAgent
