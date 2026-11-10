"""Unified Action Engine. The model only ever produces `Action` JSON; this is the one place that touches the OS.

Targets resolve in the order the spec asks for: element id (accessibility/OCR/vision element) -> raw coordinates.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

from core.types import Action, ActionResult, Element, Observation

if TYPE_CHECKING:
    from actions.platform import Desktop

logger = logging.getLogger(__name__)

AppHandler = Callable[[str], Awaitable[ActionResult]]
TERMINALS = re.compile(r"terminal|console|konsole|xterm|ptyxis|alacritty|kitty|tilix|terminator|wezterm", re.I)


class ActionEngine:
    def __init__(self, desktop: Desktop, settle: float = 0.35) -> None:
        self.desktop = desktop
        self.settle = settle
        # Application-level actions are delegated to the ApplicationManager (wired by the agent manager).
        self.open_app: AppHandler | None = None
        self.close_app: AppHandler | None = None
        self.switch_app: AppHandler | None = None

    # ---- target resolution ---------------------------------------------------------------------------------
    def resolve(self, action: Action, obs: Observation | None, which: str = "element") -> tuple[tuple[int, int] | None,
                                                                                               Element | None, str]:
        el_id = action.element if which == "element" else action.target
        if el_id is not None:
            el = obs.element(el_id) if obs else None
            if el is None:
                return None, None, f"element {el_id} is not on the current screen"
            return el.center, el, ""
        x, y = (action.x, action.y) if which == "element" else (action.x2, action.y2)
        if x is not None and y is not None:
            w, h = obs.screen_size if obs else self.desktop.screen.size()
            if not (0 <= x < w and 0 <= y < h):
                return None, None, f"point ({x},{y}) is outside the {w}x{h} screen"
            return (x, y), None, ""
        return None, None, "no target: give an element id or x/y"

    # ---- execution -----------------------------------------------------------------------------------------
    async def execute(self, action: Action, obs: Observation | None) -> ActionResult:
        t = time.perf_counter()
        try:
            res = await self._dispatch(action, obs)
        except Exception as exc:  # noqa: BLE001 - report OS-level failures to the agent so it can recover
            logger.exception("action %s failed", action.action)
            res = ActionResult(False, f"{type(exc).__name__}: {exc}")
        res.duration_ms = (time.perf_counter() - t) * 1000
        if res.ok and action.action not in ("wait", "screenshot") and self.settle:
            await asyncio.sleep(self.settle)
        return res

    @staticmethod
    def should_paste(text: str) -> bool:
        """Paste instead of typing: long text, several lines, or characters editors auto-complete ({[(\"')."""
        body = text[:-1] if text.endswith("\n") else text
        return len(body) > 40 or "\n" in body or any(c in body for c in "{[(\"'<")

    async def _insert_text(self, text: str, obs: Observation | None) -> str:
        """Type short text; paste long/structured text through the clipboard (exact, fast, no auto-closing
        brackets) and restore the user's clipboard afterwards. A trailing newline is sent as Enter."""
        k, clip = self.desktop.keyboard, self.desktop.clipboard
        run = asyncio.to_thread
        if clip is None or not self.should_paste(text):
            await run(k.type_text, text)
            return "typed"
        body, enter = (text[:-1], True) if text.endswith("\n") else (text, False)
        win = obs.active_window if obs else None
        is_terminal = bool(win and TERMINALS.search(f"{win.wm_class} {win.title}"))
        previous = await run(clip.get_text)
        await run(clip.set_text, body)
        await run(k.hotkey, ["ctrl", "shift", "v"] if is_terminal else ["ctrl", "v"])
        await asyncio.sleep(0.35)  # the app fetches the data asynchronously after the shortcut
        if enter:
            await run(k.press, "enter")
        if previous is not None:
            await run(clip.set_text, previous)
        return "pasted"

    async def _dispatch(self, a: Action, obs: Observation | None) -> ActionResult:
        m, k = self.desktop.mouse, self.desktop.keyboard
        run = asyncio.to_thread
        name = a.action

        if name in ("click", "double_click", "right_click", "mouse_move"):
            pt, el, err = self.resolve(a, obs)
            if pt is None:
                return ActionResult(False, err)
            a.target_label = a.target_label or (f'{el.type} "{el.text[:40]}"' if el else f"({pt[0]},{pt[1]})")
            if name == "mouse_move":
                await run(m.move, *pt)
            else:
                button = "right" if name == "right_click" else "left"
                await run(m.click, pt[0], pt[1], button, 2 if name == "double_click" else 1)
            return ActionResult(True, f"{name} at {pt}", point=pt, bbox=el.bbox if el else None)

        if name in ("mouse_drag", "drag_and_drop"):
            p1, el1, err = self.resolve(a, obs, "element")
            if p1 is None:
                return ActionResult(False, err)
            p2, el2, err = self.resolve(a, obs, "target")
            if p2 is None:
                return ActionResult(False, "drag needs a destination (target element or x2/y2): " + err)
            await run(m.drag, p1[0], p1[1], p2[0], p2[1])
            return ActionResult(True, f"dragged {p1} -> {p2}", point=p2, data={"from": p1})

        if name == "scroll":
            pt, el, _ = self.resolve(a, obs)
            if pt is None:  # default: middle of the active window
                win = obs.active_window if obs else None
                if win:
                    x, y, w, h = win.geometry
                    pt = (x + w // 2, y + h // 2)
                else:
                    sw, sh = self.desktop.screen.size()
                    pt = (sw // 2, sh // 2)
            clicks = a.amount or 3
            await run(m.scroll, pt[0], pt[1], clicks)
            return ActionResult(True, f"scrolled {'down' if clicks > 0 else 'up'} {abs(clicks)} at {pt}", point=pt)

        if name == "type_text":
            if not a.text:
                return ActionResult(False, "type_text needs text")
            pt = None
            if a.element is not None or a.x is not None:  # focus the field first
                pt, el, err = self.resolve(a, obs)
                if pt is None:
                    return ActionResult(False, err)
                if not (el and "focused" in el.states):
                    await run(m.click, pt[0], pt[1], "left", 1)
                    await asyncio.sleep(0.15)
            how = await self._insert_text(a.text, obs)
            return ActionResult(True, f"{how} all {len(a.text)} characters (complete, nothing was cut off)", point=pt)

        if name == "key_press":
            keys = a.keys or ([a.text] if a.text else [])
            if not keys:
                return ActionResult(False, "key_press needs a key")
            for key in keys:  # several keys = pressed one after another (e.g. ["Down", "Down", "Enter"])
                await run(k.press, key)
                await asyncio.sleep(0.05)
            return ActionResult(True, f"pressed {' '.join(keys)}")

        if name == "hotkey":
            if not a.keys:
                return ActionResult(False, "hotkey needs keys, e.g. [\"ctrl\", \"s\"]")
            await run(k.hotkey, a.keys)
            return ActionResult(True, f"hotkey {'+'.join(a.keys)}")

        if name == "copy":
            await run(k.hotkey, ["ctrl", "c"])
            return ActionResult(True, "copied selection")
        if name == "paste":
            await run(k.hotkey, ["ctrl", "v"])
            return ActionResult(True, "pasted")
        if name == "select_text":
            if a.element is not None or a.x is not None:
                pt, el, err = self.resolve(a, obs)
                if pt is None:
                    return ActionResult(False, err)
                await run(m.click, pt[0], pt[1], "left", 3)  # triple click selects a line/field
                return ActionResult(True, "selected text in target", point=pt)
            await run(k.hotkey, ["ctrl", "a"])
            return ActionResult(True, "selected all")

        if name == "wait":
            secs = min(max(a.amount or 1, 0), 30)
            await asyncio.sleep(secs)
            return ActionResult(True, f"waited {secs}s")
        if name == "screenshot":
            return ActionResult(True, "screenshot requested (next observation is fresh)")

        if name in ("open_application", "close_application", "switch_application"):
            handler = {"open_application": self.open_app, "close_application": self.close_app,
                       "switch_application": self.switch_app}[name]
            if handler is None:
                return ActionResult(False, f"{name} is not available in this context")
            if not a.app:
                return ActionResult(False, f"{name} needs an app name")
            return await handler(a.app)

        return ActionResult(False, f"action {name!r} is handled by the agent, not the engine")
