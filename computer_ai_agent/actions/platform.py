"""Picks the desktop backend for this machine. Everything above this module talks to `Desktop` only.

Adding Windows or macOS = implement WindowManager + ScreenCapture for that OS and add a branch below; Mouse and
Keyboard already work there through pynput.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass

from actions.keyboard import Keyboard, X11Keyboard
from actions.mouse import Mouse
from actions.window_manager import WindowManager
from vision.screen_capture import ScreenCapture

logger = logging.getLogger(__name__)


@dataclass
class Desktop:
    name: str                 # "x11", "wayland", ...
    display: str
    screen: ScreenCapture
    mouse: Mouse
    keyboard: Keyboard
    windows: WindowManager
    accessibility: bool       # AT-SPI usable on this desktop
    clipboard: object | None = None  # X11Clipboard: set_text/get_text (None = always type key by key)


def session_type() -> str:
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform == "darwin":
        return "macos"
    t = os.getenv("XDG_SESSION_TYPE", "").lower()
    if t:
        return t
    return "wayland" if os.getenv("WAYLAND_DISPLAY") else "x11"


def create_desktop(display: str = "", accessibility: bool = True) -> Desktop:
    kind = session_type()
    if display:
        # An explicit X display (e.g. a nested Xephyr test screen) is always driven through X11.
        os.environ["DISPLAY"] = display
        kind = "x11"
    if kind in ("windows", "macos"):
        raise NotImplementedError(
            f"{kind} backend is not implemented yet: add a WindowManager and ScreenCapture in actions/ and vision/"
        )
    if kind == "wayland" and not display:
        return _wayland(accessibility)
    disp = os.environ.get("DISPLAY", "")
    if not disp:
        raise RuntimeError("No X display: set DISPLAY or AGENT_DISPLAY (e.g. :0 or :1)")
    from actions.window_manager import X11WindowManager
    from vision.screen_capture import MssCapture

    return Desktop(
        name="x11",
        display=disp,
        screen=MssCapture(disp),
        mouse=Mouse(),
        keyboard=X11Keyboard(disp),
        windows=X11WindowManager(disp),
        accessibility=accessibility and _atspi_available(),
        clipboard=_clipboard(disp),
    )


def _clipboard(disp: str):  # noqa: ANN202
    try:
        from actions.clipboard import X11Clipboard

        return X11Clipboard(disp)
    except Exception as exc:  # noqa: BLE001 - fall back to typing
        logger.warning("clipboard unavailable (%s); long text will be typed key by key", exc)
        return None


def _atspi_available() -> bool:
    try:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def _wayland(accessibility: bool) -> Desktop:
    """GNOME/KDE Wayland. Input goes through ydotool (uinput); X11 window control only reaches XWayland apps."""
    from actions.window_manager import X11WindowManager
    from vision.screen_capture import CommandCapture

    if not shutil.which("ydotool"):
        raise RuntimeError(
            "Wayland session detected. Synthetic input on Wayland needs ydotool (sudo apt install ydotool and run "
            "ydotoold), or log in with 'Ubuntu on Xorg' for full support."
        )
    logger.warning("Wayland: window listing/activation only sees XWayland windows; prefer an Xorg session")
    disp = os.environ.get("DISPLAY", ":0")
    return Desktop(
        name="wayland",
        display=disp,
        screen=CommandCapture(),
        mouse=_YdotoolMouse(),
        keyboard=_YdotoolKeyboard(),
        windows=X11WindowManager(disp),
        accessibility=accessibility and _atspi_available(),
    )


class _YdotoolMouse(Mouse):
    def __init__(self) -> None:  # noqa: D107 - no pynput on Wayland
        self.move_steps, self.move_duration = 1, 0.0
        self._pos = (0, 0)
        import threading

        self._lock = threading.Lock()

    @property
    def position(self) -> tuple[int, int]:
        return self._pos

    def _run(self, *args: str) -> None:
        subprocess.run(["ydotool", *args], check=True, capture_output=True, timeout=5)

    def move(self, x: int, y: int) -> None:
        self._run("mousemove", "--absolute", "-x", str(x), "-y", str(y))
        self._pos = (x, y)

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> None:
        self.move(x, y)
        code = {"left": "0xC0", "right": "0xC1", "middle": "0xC2"}[button]
        for _ in range(count):
            self._run("click", code)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.4) -> None:
        self.move(x1, y1)
        self._run("click", "0x40")  # left down
        self.move(x2, y2)
        self._run("click", "0x80")  # left up

    def scroll(self, x: int, y: int, clicks: int) -> None:
        self.move(x, y)
        self._run("mousemove", "--wheel", "-x", "0", "-y", str(-clicks))


class _YdotoolKeyboard(Keyboard):
    # Linux input-event key codes for the keys the agent uses most.
    CODES = {
        "ctrl": 29, "shift": 42, "alt": 56, "cmd": 125, "enter": 28, "esc": 1, "tab": 15, "space": 57,
        "backspace": 14, "delete": 111, "up": 103, "down": 108, "left": 105, "right": 106, "home": 102, "end": 107,
        "page_up": 104, "page_down": 109, **{f"f{i}": 58 + i for i in range(1, 11)}, "f11": 87, "f12": 88,
    }
    LETTERS = "qwertyuiop" + "asdfghjkl" + "zxcvbnm"
    LETTER_CODES = dict(zip("qwertyuiop", range(16, 26))) | dict(zip("asdfghjkl", range(30, 39))) | dict(
        zip("zxcvbnm", range(44, 51))
    ) | {str(d): 2 + (d - 1) % 10 for d in range(1, 10)} | {"0": 11}

    def __init__(self) -> None:  # noqa: D107
        import threading

        self.char_delay = 0.0
        self._lock = threading.Lock()

    def _code(self, name: str) -> int:
        from actions.keyboard import normalize_key

        k = normalize_key(name)
        if k in self.CODES:
            return self.CODES[k]
        if k in self.LETTER_CODES:
            return self.LETTER_CODES[k]
        raise ValueError(f"key {name!r} not supported by the ydotool backend")

    def press(self, name: str) -> None:
        c = self._code(name)
        subprocess.run(["ydotool", "key", f"{c}:1", f"{c}:0"], check=True, capture_output=True, timeout=5)

    def hotkey(self, names: list[str]) -> None:
        codes = [self._code(n) for n in names]
        seq = [f"{c}:1" for c in codes] + [f"{c}:0" for c in reversed(codes)]
        subprocess.run(["ydotool", "key", *seq], check=True, capture_output=True, timeout=5)

    def type_text(self, text: str) -> None:
        subprocess.run(["ydotool", "type", "--", text], check=True, capture_output=True, timeout=60)
