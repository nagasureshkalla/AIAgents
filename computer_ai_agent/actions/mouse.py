"""Mouse control with smooth-ish movement so apps receive hover/enter events before a click."""

from __future__ import annotations

import threading
import time


class Mouse:
    def __init__(self, move_steps: int = 8, move_duration: float = 0.08) -> None:
        from pynput.mouse import Button, Controller  # imported late so $DISPLAY is already set

        self._m = Controller()
        self._Button = Button
        self.move_steps = move_steps
        self.move_duration = move_duration
        self._lock = threading.Lock()

    @property
    def position(self) -> tuple[int, int]:
        x, y = self._m.position
        return int(x), int(y)

    def _glide(self, x: int, y: int) -> None:
        sx, sy = self.position
        steps = max(1, self.move_steps)
        for i in range(1, steps + 1):
            t = i / steps
            self._m.position = (round(sx + (x - sx) * t), round(sy + (y - sy) * t))
            time.sleep(self.move_duration / steps)

    def move(self, x: int, y: int) -> None:
        with self._lock:
            self._glide(x, y)

    def click(self, x: int, y: int, button: str = "left", count: int = 1) -> None:
        btn = getattr(self._Button, button)
        with self._lock:
            self._glide(x, y)
            time.sleep(0.03)
            for i in range(count):
                self._m.press(btn)
                self._m.release(btn)
                if i + 1 < count:
                    time.sleep(0.06)

    def drag(self, x1: int, y1: int, x2: int, y2: int, duration: float = 0.4) -> None:
        with self._lock:
            self._glide(x1, y1)
            self._m.press(self._Button.left)
            try:
                steps = 20
                for i in range(1, steps + 1):
                    t = i / steps
                    self._m.position = (round(x1 + (x2 - x1) * t), round(y1 + (y2 - y1) * t))
                    time.sleep(duration / steps)
            finally:
                self._m.release(self._Button.left)

    def scroll(self, x: int, y: int, clicks: int) -> None:
        """Positive clicks scroll down, negative scroll up."""
        with self._lock:
            self._glide(x, y)
            self._m.scroll(0, -clicks)
