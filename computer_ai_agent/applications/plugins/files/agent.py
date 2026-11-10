"""File manager (GNOME Files / Nautilus, Thunar...) sub-agent."""

from __future__ import annotations

from agents.application_agent import GenericApplicationAgent

APP_KEYS = ["org.gnome.nautilus", "nautilus", "files", "thunar", "org.kde.dolphin", "nemo", "file manager"]


class FilesSubAgent(GenericApplicationAgent):
    pass


AGENT = FilesSubAgent
