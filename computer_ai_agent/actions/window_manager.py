"""Window management. WindowManager is the platform contract; X11WindowManager implements it with EWMH.

Windows/macOS backends implement the same five methods (list, active, activate, close, wait_for) and nothing
above this layer changes.
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable

from core.types import WindowInfo

logger = logging.getLogger(__name__)


class WindowManager(ABC):
    @abstractmethod
    def list_windows(self) -> list[WindowInfo]: ...

    @abstractmethod
    def active_window(self) -> WindowInfo | None: ...

    @abstractmethod
    def activate(self, window_id: int) -> bool: ...

    @abstractmethod
    def close(self, window_id: int) -> bool: ...

    def find(self, predicate: Callable[[WindowInfo], bool]) -> list[WindowInfo]:
        return [w for w in self.list_windows() if predicate(w)]

    def wait_for(self, predicate: Callable[[WindowInfo], bool], timeout: float, interval: float = 0.3) -> WindowInfo | None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            found = self.find(predicate)
            if found:
                return found[-1]  # newest last in _NET_CLIENT_LIST stacking order
            time.sleep(interval)
        return None


class X11WindowManager(WindowManager):
    """EWMH via python-xlib (works under GNOME/Mutter, KDE, XFCE...). Falls back to the raw window tree when no
    EWMH window manager is running (e.g. a bare Xephyr/Xvfb test display)."""

    def __init__(self, display: str | None = None) -> None:
        from Xlib import X, Xatom, display as xdisplay
        from Xlib.protocol import event as xevent

        self.X, self.Xatom, self._event = X, Xatom, xevent
        self.d = xdisplay.Display(display or None)
        self.root = self.d.screen().root
        self._lock = threading.Lock()
        self._atoms: dict[str, int] = {}

    def atom(self, name: str) -> int:
        if name not in self._atoms:
            self._atoms[name] = self.d.intern_atom(name)
        return self._atoms[name]

    # ---- property helpers --------------------------------------------------------------------------------
    def _prop(self, win, name: str, kind=None):  # noqa: ANN001
        try:
            p = win.get_full_property(self.atom(name), kind if kind is not None else self.X.AnyPropertyType)
            return p.value if p else None
        except Exception:  # noqa: BLE001 - window vanished between list and query
            return None

    def _title(self, win) -> str:  # noqa: ANN001
        v = self._prop(win, "_NET_WM_NAME", self.atom("UTF8_STRING"))
        if v is None:
            v = self._prop(win, "WM_NAME")
        if isinstance(v, bytes):
            return v.decode("utf-8", "replace")
        return str(v or "")

    def _wm_class(self, win) -> str:  # noqa: ANN001
        try:
            cls = win.get_wm_class()
            return cls[1] if cls else ""
        except Exception:  # noqa: BLE001
            return ""

    def _pid(self, win) -> int | None:  # noqa: ANN001
        v = self._prop(win, "_NET_WM_PID", self.Xatom.CARDINAL)
        return int(v[0]) if v is not None and len(v) else None

    def _geometry(self, win) -> tuple[int, int, int, int]:  # noqa: ANN001
        try:
            g = win.get_geometry()
            t = win.translate_coords(self.root, 0, 0)
            return -t.x, -t.y, g.width, g.height
        except Exception:  # noqa: BLE001
            return 0, 0, 0, 0

    def _info(self, wid: int, active_id: int | None) -> WindowInfo | None:
        win = self.d.create_resource_object("window", wid)
        try:
            attrs = win.get_attributes()
            if attrs.map_state != self.X.IsViewable:
                return None
        except Exception:  # noqa: BLE001
            return None
        desk = self._prop(win, "_NET_WM_DESKTOP", self.Xatom.CARDINAL)
        return WindowInfo(
            id=wid,
            title=self._title(win),
            wm_class=self._wm_class(win),
            pid=self._pid(win),
            geometry=self._geometry(win),
            active=wid == active_id,
            desktop=int(desk[0]) if desk is not None and len(desk) else None,
        )

    def _active_id(self) -> int | None:
        v = self._prop(self.root, "_NET_ACTIVE_WINDOW", self.Xatom.WINDOW)
        if v is not None and len(v) and v[0]:
            return int(v[0])
        try:  # no EWMH: whatever has input focus (walk up to the top-level child of root)
            focus = self.d.get_input_focus().focus
            if isinstance(focus, int):
                return None
            win = focus
            while True:
                parent = win.query_tree().parent
                if parent is None or parent.id == self.root.id:
                    return win.id
                win = parent
        except Exception:  # noqa: BLE001
            return None

    def _client_ids(self) -> list[int]:
        v = self._prop(self.root, "_NET_CLIENT_LIST", self.Xatom.WINDOW)
        if v is not None:
            return [int(x) for x in v]
        ids = []  # no EWMH window manager: top-level children of root that look like app windows
        for child in self.root.query_tree().children:
            if self._wm_class(child) or self._title(child):
                ids.append(child.id)
        return ids

    # ---- public API --------------------------------------------------------------------------------------
    def list_windows(self) -> list[WindowInfo]:
        with self._lock:
            active = self._active_id()
            out = [self._info(wid, active) for wid in self._client_ids()]
            return [w for w in out if w is not None]

    def active_window(self) -> WindowInfo | None:
        with self._lock:
            active = self._active_id()
            return self._info(active, active) if active else None

    def _client_message(self, win, atom_name: str, data: list[int]) -> None:  # noqa: ANN001
        ev = self._event.ClientMessage(window=win, client_type=self.atom(atom_name), data=(32, (data + [0] * 5)[:5]))
        mask = self.X.SubstructureRedirectMask | self.X.SubstructureNotifyMask
        self.root.send_event(ev, event_mask=mask)

    def activate(self, window_id: int) -> bool:
        with self._lock:
            win = self.d.create_resource_object("window", window_id)
            try:
                desk = self._prop(win, "_NET_WM_DESKTOP", self.Xatom.CARDINAL)
                if desk is not None and len(desk):
                    self._client_message(self.root, "_NET_CURRENT_DESKTOP", [int(desk[0]), self.X.CurrentTime])
                # source indication 2 = pager/tool: window managers honour it instead of flashing the taskbar
                self._client_message(win, "_NET_ACTIVE_WINDOW", [2, self.X.CurrentTime, 0])
                win.map()
                win.raise_window()
                self.d.flush()
            except Exception as exc:  # noqa: BLE001
                logger.warning("activate %s failed: %s", window_id, exc)
                return False
        for _ in range(15):
            time.sleep(0.05)
            if self._active_id() == window_id:
                return True
        with self._lock:  # WM ignored the request (or no WM): set focus directly
            try:
                win.set_input_focus(self.X.RevertToParent, self.X.CurrentTime)
                self.d.flush()
            except Exception:  # noqa: BLE001
                return False
        return True

    def close(self, window_id: int) -> bool:
        with self._lock:
            win = self.d.create_resource_object("window", window_id)
            try:
                self._client_message(win, "_NET_CLOSE_WINDOW", [self.X.CurrentTime, 2])
                self.d.flush()
                return True
            except Exception as exc:  # noqa: BLE001
                logger.warning("close %s failed: %s", window_id, exc)
                return False

    def screen_size(self) -> tuple[int, int]:
        s = self.d.screen()
        return s.width_in_pixels, s.height_in_pixels
