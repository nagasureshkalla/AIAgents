"""Classifies every action before it runs: SAFE, CONFIRM (ask the user first) or BLOCK (never do it)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from core.types import Action, Element, Observation


class Risk(str, Enum):
    SAFE = "safe"
    CONFIRM = "confirm"
    BLOCK = "block"


@dataclass
class Verdict:
    risk: Risk
    reason: str = ""
    category: str = ""


# Controls whose activation sends, deletes, pays, publishes or otherwise can't be taken back.
DESTRUCTIVE_CONTROL = re.compile(
    r"\b(delete|remove|erase|discard|destroy|wipe|format|uninstall|empty trash|move to trash|permanently|"
    r"send|submit|post|publish|tweet|reply all|share now|buy|purchase|pay|checkout|check out|place order|"
    r"order now|subscribe|transfer|withdraw|confirm payment|overwrite|replace all|reset|factory)\b",
    re.I,
)
CATEGORY_WORDS = [
    ("delete", r"delete|remove|erase|discard|destroy|wipe|trash|permanently|overwrite|replace all|reset|factory"),
    ("format", r"format"),
    ("uninstall", r"uninstall"),
    ("send", r"send|reply all|share now"),
    ("publish", r"post|publish|tweet|submit"),
    ("purchase", r"buy|purchase|pay|checkout|check out|order|subscribe"),
    ("money", r"transfer|withdraw|payment"),
]
DANGEROUS_SHELL = re.compile(
    r"(\brm\s+(-[a-z]*\s+)*|\brmdir\b|\bmkfs|\bdd\s+if=|\bshred\b|\bwipefs\b|\bfdisk\b|\bparted\b|"
    r"\bshutdown\b|\breboot\b|\bpoweroff\b|\bhalt\b|\binit\s+[06]\b|\bsudo\b|\bsu\s|\bpkexec\b|"
    r"\bchmod\s+(-R\s+)?[0-7]*7[0-7]*\s+/|\bchown\s+-R\b|>\s*/dev/sd|:\(\)\s*\{|"
    r"\bapt(-get)?\s+(remove|purge|autoremove)|\bsnap\s+remove|\bpip3?\s+uninstall|\bnpm\s+uninstall\s+-g|"
    r"\bgit\s+(push\s+.*--force|push\s+-f|reset\s+--hard|clean\s+-[a-z]*f)|\bkill(all)?\s+-9|\bpkill\b|"
    r"(curl|wget)[^|]*\|\s*(ba|z)?sh|\bmv\s+.*\s+/dev/null|\btruncate\b|\bcrontab\s+-r)",
    re.I,
)
TERMINAL_APPS = re.compile(r"terminal|console|konsole|xterm|ptyxis|alacritty|kitty|tilix|terminator|wezterm", re.I)
FILE_MANAGERS = re.compile(r"nautilus|files|thunar|dolphin|nemo|pcmanfm|file manager", re.I)
SECRET_HINT = re.compile(r"password|passcode|passphrase|\bpin\b|otp|one-time|2fa|secret|api key|token|cvv|card number",
                         re.I)


def _category(text: str) -> str:
    for name, pat in CATEGORY_WORDS:
        if re.search(pat, text, re.I):
            return name
    return "destructive"


class PermissionManager:
    def __init__(self, require_confirmation: bool = True) -> None:
        self.require_confirmation = require_confirmation

    def check(self, action: Action, obs: Observation | None, app_name: str, user_command: str = "") -> Verdict:
        el: Element | None = None
        if obs is not None and action.element is not None:
            el = obs.element(action.element)
        focused = obs.focused if obs else None
        window_title = obs.active_window.title if obs and obs.active_window else ""

        # 1) never handle secrets: don't type into password fields, don't read them (the a11y reader skips them)
        if action.action == "type_text":
            target = el or focused
            if target is not None and (target.type == "password" or SECRET_HINT.search(target.text or "")):
                return Verdict(Risk.BLOCK, "The target is a password/secret field. Please enter it yourself.", "secret")

        # 2) clicking a control labelled with a destructive verb
        if action.action in ("click", "double_click", "key_press") and el is not None:
            label = f"{el.text}"
            if DESTRUCTIVE_CONTROL.search(label):
                return Verdict(Risk.CONFIRM, f'Clicking "{el.text[:60]}" may be irreversible.', _category(label))
        # Enter/space on a focused destructive button
        if action.action == "key_press" and focused is not None and focused.type == "button":
            if any(k.lower() in ("enter", "return", "space") for k in action.keys) and DESTRUCTIVE_CONTROL.search(
                    focused.text):
                return Verdict(Risk.CONFIRM, f'Activating "{focused.text[:60]}" may be irreversible.',
                               _category(focused.text))

        # 3) shell commands in a terminal
        if action.action == "type_text" and (TERMINAL_APPS.search(app_name) or TERMINAL_APPS.search(window_title)):
            m = DANGEROUS_SHELL.search(action.text)
            if m:
                return Verdict(Risk.CONFIRM, f"Terminal command contains `{m.group(0).strip()}`: {action.text[:120]}",
                               "dangerous_command")

        # 4) deleting files in a file manager
        if FILE_MANAGERS.search(app_name) and (
            (action.action in ("key_press", "hotkey") and any(k.lower() in ("delete", "del") for k in action.keys))
        ):
            return Verdict(Risk.CONFIRM, "Deleting the selected file(s).", "delete")

        # 5) closing an application the user did not ask to close (unsaved work could be lost)
        if action.action == "close_application" and not re.search(r"\b(close|quit|exit)\b", user_command, re.I):
            return Verdict(Risk.CONFIRM, f"Closing {action.app or app_name}; unsaved work could be lost.", "close")
        if action.action == "hotkey" and {k.lower() for k in action.keys} in ({"ctrl", "q"}, {"alt", "f4"}) and not \
                re.search(r"\b(close|quit|exit)\b", user_command, re.I):
            return Verdict(Risk.CONFIRM, "Quitting the application; unsaved work could be lost.", "close")

        return Verdict(Risk.SAFE)

    def needs_prompt(self, verdict: Verdict) -> bool:
        return verdict.risk == Risk.CONFIRM and self.require_confirmation
