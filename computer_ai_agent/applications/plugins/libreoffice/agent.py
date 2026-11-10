"""LibreOffice (Writer, Calc, Impress) sub-agent."""

from __future__ import annotations

from agents.application_agent import GenericApplicationAgent

APP_KEYS = ["libreoffice-startcenter", "libreoffice-writer", "libreoffice-calc", "libreoffice-impress", "libreoffice",
            "soffice"]


class LibreOfficeSubAgent(GenericApplicationAgent):
    pass


AGENT = LibreOfficeSubAgent
