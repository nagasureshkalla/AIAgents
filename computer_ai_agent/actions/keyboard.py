"""Keyboard control. Key names from the model (CTRL, Enter, PageDown, F5, super...) are normalized here."""

from __future__ import annotations

import threading
import time
from typing import Any

# Canonical names -> pynput Key attribute names.
KEY_ALIASES: dict[str, str] = {
    "ctrl": "ctrl", "control": "ctrl", "ctl": "ctrl", "lctrl": "ctrl_l", "rctrl": "ctrl_r",
    "alt": "alt", "option": "alt", "lalt": "alt_l", "ralt": "alt_r", "altgr": "alt_gr",
    "shift": "shift", "lshift": "shift_l", "rshift": "shift_r",
    "super": "cmd", "win": "cmd", "windows": "cmd", "meta": "cmd", "cmd": "cmd", "command": "cmd", "logo": "cmd",
    "enter": "enter", "return": "enter", "ret": "enter",
    "esc": "esc", "escape": "esc", "tab": "tab", "space": "space", "spacebar": "space",
    "backspace": "backspace", "bksp": "backspace", "delete": "delete", "del": "delete", "insert": "insert", "ins": "insert",
    "up": "up", "down": "down", "left": "left", "right": "right",
    "arrowup": "up", "arrowdown": "down", "arrowleft": "left", "arrowright": "right",
    "home": "home", "end": "end", "pageup": "page_up", "pgup": "page_up", "pagedown": "page_down", "pgdn": "page_down",
    "capslock": "caps_lock", "menu": "menu", "printscreen": "print_screen", "print": "print_screen",
    **{f"f{i}": f"f{i}" for i in range(1, 21)},
}
MODIFIERS = {"ctrl", "ctrl_l", "ctrl_r", "alt", "alt_l", "alt_r", "alt_gr", "shift", "shift_l", "shift_r", "cmd"}


def normalize_key(name: str) -> str:
    """Return a canonical key name: a pynput Key attribute name, or a single character."""
    raw = str(name).strip()
    if len(raw) == 1:
        return raw.lower() if raw.isalpha() else raw
    k = raw.lower().replace(" ", "").replace("-", "").replace("_", "")
    if k in KEY_ALIASES:
        return KEY_ALIASES[k]
    if k in ("plus",):
        return "+"
    if k in ("minus",):
        return "-"
    raise ValueError(f"unknown key {name!r}")


class Keyboard:
    """Thin wrapper over pynput's keyboard controller (used for Windows/macOS backends).

    On X11 use X11Keyboard instead: pynput types characters with synthetic XSendEvent events, which GTK4 apps
    (e.g. GNOME Calculator 46) silently ignore."""

    def __init__(self, char_delay: float = 0.004) -> None:
        from pynput.keyboard import Controller, Key  # imported late so $DISPLAY is already set

        self._kb = Controller()
        self._Key = Key
        self.char_delay = char_delay
        self._lock = threading.Lock()

    def _key(self, canonical: str) -> Any:
        return getattr(self._Key, canonical) if len(canonical) > 1 else canonical

    def press(self, name: str) -> None:
        key = self._key(normalize_key(name))
        with self._lock:
            self._kb.press(key)
            self._kb.release(key)

    def hotkey(self, names: list[str]) -> None:
        keys = [self._key(normalize_key(n)) for n in names]
        with self._lock:
            pressed = []
            try:
                for k in keys:
                    self._kb.press(k)
                    pressed.append(k)
                    time.sleep(0.02)
            finally:
                for k in reversed(pressed):
                    self._kb.release(k)

    def type_text(self, text: str) -> None:
        with self._lock:
            for ch in text:
                if ch == "\n":
                    self._kb.press(self._Key.enter)
                    self._kb.release(self._Key.enter)
                elif ch == "\t":
                    self._kb.press(self._Key.tab)
                    self._kb.release(self._Key.tab)
                else:
                    self._kb.type(ch)
                if self.char_delay:
                    time.sleep(self.char_delay)


# canonical key name -> X keysym name
X11_KEYSYMS = {
    "ctrl": "Control_L", "ctrl_l": "Control_L", "ctrl_r": "Control_R", "alt": "Alt_L", "alt_l": "Alt_L",
    "alt_r": "Alt_R", "alt_gr": "ISO_Level3_Shift", "shift": "Shift_L", "shift_l": "Shift_L", "shift_r": "Shift_R",
    "cmd": "Super_L", "enter": "Return", "esc": "Escape", "tab": "Tab", "space": "space", "backspace": "BackSpace",
    "delete": "Delete", "insert": "Insert", "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "home": "Home", "end": "End", "page_up": "Prior", "page_down": "Next", "caps_lock": "Caps_Lock", "menu": "Menu",
    "print_screen": "Print", **{f"f{i}": f"F{i}" for i in range(1, 21)},
}


class X11Keyboard(Keyboard):
    """Real XTest key events (what a physical keyboard produces). Characters missing from the current layout are
    typed by temporarily mapping them onto a spare keycode."""

    def __init__(self, display: str | None = None, char_delay: float = 0.004) -> None:  # noqa: D107
        from Xlib import X, XK, display as xdisplay
        from Xlib.ext import xtest

        self.X, self.XK, self._xtest = X, XK, xtest
        self.d = xdisplay.Display(display or None)
        self.char_delay = char_delay
        self._lock = threading.Lock()
        self._shift = self.d.keysym_to_keycode(XK.string_to_keysym("Shift_L"))
        self._altgr = self.d.keysym_to_keycode(XK.string_to_keysym("ISO_Level3_Shift"))
        self._spare = self._find_spare_keycode()

    def _find_spare_keycode(self) -> int | None:
        first, last = self.d.display.info.min_keycode, self.d.display.info.max_keycode
        mapping = self.d.get_keyboard_mapping(first, last - first + 1)
        for offset in range(len(mapping) - 1, -1, -1):
            if not any(mapping[offset]):
                return first + offset
        return None

    def _keysym_for_char(self, ch: str) -> int:
        cp = ord(ch)
        if 0x20 <= cp <= 0x7E or 0xA0 <= cp <= 0xFF:
            return cp  # Latin-1 keysyms equal the code point
        return 0x01000000 + cp  # Unicode keysym

    def _fake(self, keycode: int, down: bool) -> None:
        self._xtest.fake_input(self.d, self.X.KeyPress if down else self.X.KeyRelease, keycode)

    def _tap(self, keycode: int, modifiers: list[int]) -> None:
        for m in modifiers:
            self._fake(m, True)
        self._fake(keycode, True)
        self._fake(keycode, False)
        for m in reversed(modifiers):
            self._fake(m, False)
        self.d.sync()

    def _keycode_for_name(self, canonical: str) -> int:
        if len(canonical) == 1:
            keysym = self._keysym_for_char(canonical)
        else:
            keysym = self.XK.string_to_keysym(X11_KEYSYMS[canonical])
        kc = self.d.keysym_to_keycode(keysym)
        if not kc:
            raise ValueError(f"key {canonical!r} is not on the keyboard map")
        return kc

    def _type_char(self, ch: str) -> None:
        keysym = self._keysym_for_char(ch)
        kc = self.d.keysym_to_keycode(keysym)
        if kc:
            # which shift level produces this keysym on that keycode?
            for index, mods in ((0, []), (1, [self._shift]), (2, [self._altgr]), (3, [self._shift, self._altgr])):
                if self.d.keycode_to_keysym(kc, index) == keysym:
                    self._tap(kc, mods)
                    return
            self._tap(kc, [])
            return
        if self._spare is None:
            raise ValueError(f"cannot type {ch!r}: no spare keycode to remap")
        self.d.change_keyboard_mapping(self._spare, [(keysym, keysym)])
        self.d.sync()
        time.sleep(0.03)  # clients must see the MappingNotify before the key arrives
        self._tap(self._spare, [])
        time.sleep(0.01)
        self.d.change_keyboard_mapping(self._spare, [(0, 0)])
        self.d.sync()

    def press(self, name: str) -> None:
        canonical = normalize_key(name)
        with self._lock:
            if len(canonical) == 1:
                self._type_char(canonical)
            else:
                self._tap(self._keycode_for_name(canonical), [])

    def hotkey(self, names: list[str]) -> None:
        codes = [self._keycode_for_name(normalize_key(n)) for n in names]
        with self._lock:
            pressed = []
            try:
                for kc in codes:
                    self._fake(kc, True)
                    pressed.append(kc)
                    self.d.sync()
                    time.sleep(0.02)
            finally:
                for kc in reversed(pressed):
                    self._fake(kc, False)
                self.d.sync()

    def type_text(self, text: str) -> None:
        enter = self._keycode_for_name("enter")
        tab = self._keycode_for_name("tab")
        with self._lock:
            for ch in text:
                if ch == "\n":
                    self._tap(enter, [])
                elif ch == "\t":
                    self._tap(tab, [])
                else:
                    self._type_char(ch)
                if self.char_delay:
                    time.sleep(self.char_delay)
