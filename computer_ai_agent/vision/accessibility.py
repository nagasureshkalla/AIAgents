"""Reads UI elements from the AT-SPI accessibility tree (GTK, Qt, Chrome/Electron with accessibility on, LibreOffice,
Firefox...). This is the preferred source: real roles, names, states and exact boxes, no pixels involved."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# AT-SPI role name -> our element type. Roles not listed are walked through but not emitted.
ROLE_MAP = {
    "push button": "button", "button": "button", "toggle button": "button", "split button": "button",
    "check box": "checkbox", "radio button": "radio", "check menu item": "menuitem", "radio menu item": "menuitem",
    "menu": "menu", "menu item": "menuitem", "menu bar": "menubar",
    "combo box": "combobox", "entry": "textbox", "text": "textbox", "password text": "password",
    "spin button": "spinbutton", "slider": "slider", "search box": "textbox", "editbar": "textbox",
    "link": "link", "page tab": "tab", "list item": "listitem", "tree item": "treeitem", "table cell": "cell",
    "icon": "icon", "image": "image", "heading": "heading", "label": "text", "static": "text",
    "dialog": "dialog", "alert": "dialog", "notification": "dialog", "file chooser": "dialog",
    "tool bar": "toolbar", "status bar": "statusbar", "scroll bar": "scrollbar", "document web": "document",
    "document frame": "document", "terminal": "terminal", "paragraph": "text", "section": "group",
}
CONTAINER_TYPES = {"menubar", "toolbar", "group", "statusbar", "document", "dialog", "scrollbar"}
INTERESTING_STATES = ("focused", "focusable", "editable", "checked", "selected", "pressed", "expanded", "sensitive")


@dataclass
class RawElement:
    type: str
    text: str
    bbox: tuple[int, int, int, int]
    states: list[str]
    depth: int


class AccessibilityReader:
    def __init__(self, max_nodes: int = 3000, time_budget: float = 1.5) -> None:
        import gi

        gi.require_version("Atspi", "2.0")
        from gi.repository import Atspi

        self.Atspi = Atspi
        self.max_nodes = max_nodes
        self.time_budget = time_budget
        self._lock = threading.Lock()  # libatspi is not thread-safe

    def applications(self) -> list[tuple[str, int]]:
        with self._lock:
            desk = self.Atspi.get_desktop(0)
            out = []
            for i in range(desk.get_child_count()):
                app = desk.get_child_at_index(i)
                if app is None:
                    continue
                try:
                    out.append((app.get_name() or "", app.get_process_id()))
                except Exception:  # noqa: BLE001 - app exited
                    continue
            return out

    def _app_for_pid(self, pid: int):  # noqa: ANN202
        desk = self.Atspi.get_desktop(0)
        for i in range(desk.get_child_count()):
            app = desk.get_child_at_index(i)
            try:
                if app is not None and app.get_process_id() == pid:
                    return app
            except Exception:  # noqa: BLE001
                continue
        return None

    def read(
        self, pid: int, window_title: str = "", region: tuple[int, int, int, int] | None = None
    ) -> tuple[list[RawElement], bool]:
        """Elements of the app owning `pid`, limited to the window whose title matches and to `region`.

        Returns (elements, found_app). found_app=False means the app is not on the accessibility bus at all."""
        with self._lock:
            app = self._app_for_pid(pid)
            if app is None:
                return [], False
            roots = []
            for i in range(app.get_child_count()):
                win = app.get_child_at_index(i)
                if win is None:
                    continue
                try:
                    name = win.get_name() or ""
                    showing = win.get_state_set().contains(self.Atspi.StateType.SHOWING)
                except Exception:  # noqa: BLE001
                    continue
                if not showing:
                    continue
                roots.append((win, name))
            if window_title:
                matched = [r for r in roots if r[1] and (r[1] in window_title or window_title in r[1])]
                roots = matched or roots
            out: list[RawElement] = []
            deadline = time.monotonic() + self.time_budget
            budget = [self.max_nodes]
            for win, _ in roots:
                self._walk(win, 0, out, region, deadline, budget, self._origin(win, region))
            return out, True

    def _origin(self, win, region) -> tuple[int, int] | None:  # noqa: ANN001
        """GTK4 cannot report screen positions: its SCREEN extents are always (0, 0). Detect that and return the
        window's screen origin so WINDOW-relative extents can be converted. None = SCREEN extents are usable."""
        A = self.Atspi
        try:
            scr = win.get_extents(A.CoordType.SCREEN)
        except Exception:  # noqa: BLE001
            return None
        if region and scr.x == 0 and scr.y == 0 and (region[0], region[1]) != (0, 0):
            return region[0], region[1]
        return None

    def _walk(self, node, depth, out, region, deadline, budget, origin=None) -> None:  # noqa: ANN001
        if budget[0] <= 0 or time.monotonic() > deadline or depth > 60:
            return
        budget[0] -= 1
        A = self.Atspi
        try:
            ss = node.get_state_set()
            if not ss.contains(A.StateType.SHOWING) or not ss.contains(A.StateType.VISIBLE):
                return
            role = node.get_role_name() or ""
            name = (node.get_name() or "").strip()
            if origin is None:
                ext = node.get_extents(A.CoordType.SCREEN)
                bbox = (ext.x, ext.y, ext.width, ext.height)
            else:
                rel = node.get_extents(A.CoordType.WINDOW)
                size = node.get_extents(A.CoordType.SCREEN)
                bbox = (origin[0] + rel.x, origin[1] + rel.y, size.width, size.height)
            if bbox[0] <= -100000 or bbox[1] <= -100000:  # G_MININT: not laid out / off screen
                return
            child_count = node.get_child_count()
        except Exception:  # noqa: BLE001 - node disappeared mid-walk
            return
        etype = ROLE_MAP.get(role)
        if etype and bbox[2] > 1 and bbox[3] > 1 and self._in_region(bbox, region):
            text = name
            if not text and etype in ("textbox", "password", "combobox", "spinbutton", "text", "terminal"):
                text = self._text_content(node, etype)
            if not text:
                try:
                    text = (node.get_description() or "").strip()
                except Exception:  # noqa: BLE001
                    text = ""
            states = [s for s in INTERESTING_STATES if ss.contains(getattr(A.StateType, s.upper()))]
            if etype not in CONTAINER_TYPES and (text or etype in ("textbox", "password", "button", "icon", "combobox")):
                out.append(RawElement(etype, text, bbox, states, depth))
            elif etype == "dialog":
                out.append(RawElement(etype, text, bbox, states, depth))
        for i in range(min(child_count, 400)):
            try:
                child = node.get_child_at_index(i)
            except Exception:  # noqa: BLE001
                continue
            if child is not None:
                self._walk(child, depth + 1, out, region, deadline, budget, origin)

    def _text_content(self, node, etype: str) -> str:  # noqa: ANN001
        if etype == "password":
            return ""  # never read secrets
        T = self.Atspi.Text
        try:
            n = T.get_character_count(node)
            if n <= 0:
                return ""
            if etype == "terminal":  # the interesting part of a terminal is its tail
                return (T.get_text(node, max(0, n - 3000), n) or "").strip()
            return (T.get_text(node, 0, min(n, 4000)) or "").strip()
        except Exception:  # noqa: BLE001 - no Text interface on this node
            return ""

    @staticmethod
    def _in_region(bbox, region) -> bool:  # noqa: ANN001
        if region is None:
            return True
        x, y, w, h = bbox
        rx, ry, rw, rh = region
        cx, cy = x + w / 2, y + h / 2
        return rx - 2 <= cx <= rx + rw + 2 and ry - 2 <= cy <= ry + rh + 2
