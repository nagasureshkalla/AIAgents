"""X11 clipboard via python-xlib (no xclip/xsel needed).

Used to insert long or structured text with one paste instead of typing it key by key: typing is slow and editors
"help" (VS Code auto-closes brackets and quotes, auto-indents), which corrupts JSON and code. The user's previous
clipboard text is restored after the paste.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)


class X11Clipboard:
    def __init__(self, display: str | None = None) -> None:
        from Xlib import X, Xatom, display as xdisplay
        from Xlib.protocol import event as xevent

        self.X, self.Xatom, self._event = X, Xatom, xevent
        self.d = xdisplay.Display(display or None)
        screen = self.d.screen()
        self.win = screen.root.create_window(0, 0, 1, 1, 0, screen.root_depth)
        a = self.d.intern_atom
        self.CLIPBOARD, self.TARGETS, self.UTF8 = a("CLIPBOARD"), a("TARGETS"), a("UTF8_STRING")
        self.TEXT, self.PLAIN = a("TEXT"), a("text/plain;charset=utf-8")
        self.PROP = a("CUA_CLIP")
        self._data = b""
        self._owned = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # ---- reading the current clipboard (to restore it later) ----------------------------------------------
    def get_text(self, timeout: float = 0.5) -> str | None:
        with self._lock:
            owner = self.d.get_selection_owner(self.CLIPBOARD)
            if owner == self.X.NONE or (self._owned.is_set() and owner.id == self.win.id):
                return self._data.decode("utf-8", "replace") if self._owned.is_set() else None
            self.win.convert_selection(self.CLIPBOARD, self.UTF8, self.PROP, self.X.CurrentTime)
            self.d.flush()
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                while self.d.pending_events():
                    ev = self.d.next_event()
                    if ev.type == self.X.SelectionNotify and ev.selection == self.CLIPBOARD:
                        if ev.property == self.X.NONE:
                            return None
                        prop = self.win.get_full_property(self.PROP, self.X.AnyPropertyType)
                        self.win.delete_property(self.PROP)
                        if prop is None:
                            return None
                        val = prop.value
                        return val.decode("utf-8", "replace") if isinstance(val, bytes) else str(val)
                time.sleep(0.01)
            return None

    # ---- owning the clipboard -----------------------------------------------------------------------------
    def set_text(self, text: str) -> None:
        with self._lock:
            self._data = text.encode("utf-8")
            self.win.set_selection_owner(self.CLIPBOARD, self.X.CurrentTime)
            self.d.flush()
            if self.d.get_selection_owner(self.CLIPBOARD).id != self.win.id:
                raise RuntimeError("could not take ownership of the clipboard")
            self._owned.set()
        if self._thread is None or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._serve, name="clipboard-owner", daemon=True)
            self._thread.start()

    def _serve(self) -> None:
        """Answer paste requests while we own the clipboard."""
        while self._owned.is_set():
            with self._lock:
                while self.d.pending_events():
                    ev = self.d.next_event()
                    if ev.type == self.X.SelectionClear:
                        self._owned.clear()
                    elif ev.type == self.X.SelectionRequest:
                        self._answer(ev)
            time.sleep(0.005)

    def _answer(self, req) -> None:  # noqa: ANN001
        prop = req.property if req.property != self.X.NONE else req.target
        ok = True
        if req.target == self.TARGETS:
            req.requestor.change_property(prop, self.Xatom.ATOM, 32,
                                          [self.TARGETS, self.UTF8, self.Xatom.STRING, self.TEXT, self.PLAIN])
        elif req.target in (self.UTF8, self.TEXT, self.PLAIN):
            req.requestor.change_property(prop, req.target, 8, self._data)
        elif req.target == self.Xatom.STRING:
            req.requestor.change_property(prop, self.Xatom.STRING, 8,
                                          self._data.decode("utf-8").encode("latin-1", "replace"))
        else:
            ok = False
        notify = self._event.SelectionNotify(time=req.time, requestor=req.requestor, selection=req.selection,
                                             target=req.target, property=prop if ok else self.X.NONE)
        req.requestor.send_event(notify)
        self.d.flush()
