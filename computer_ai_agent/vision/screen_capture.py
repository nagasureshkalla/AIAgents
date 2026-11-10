"""Screen capture. X11 (and Windows/macOS) via mss; Wayland via grim or gnome-screenshot when available."""

from __future__ import annotations

import hashlib
import io
import shutil
import subprocess
import tempfile
import threading
from abc import ABC, abstractmethod

from PIL import Image


class ScreenCapture(ABC):
    @abstractmethod
    def grab(self) -> Image.Image: ...

    @abstractmethod
    def size(self) -> tuple[int, int]: ...


class MssCapture(ScreenCapture):
    """~10-25 ms per 1080p frame on X11 (XGetImage/XShm)."""

    def __init__(self, display: str | None = None) -> None:
        import mss

        self._mss_mod = mss
        self._display = display
        self._local = threading.local()  # mss handles are per-thread

    def _sct(self):  # noqa: ANN202
        sct = getattr(self._local, "sct", None)
        if sct is None:
            kwargs = {"display": self._display.encode()} if self._display else {}
            factory = getattr(self._mss_mod, "MSS", None) or self._mss_mod.mss  # mss >= 10 renamed the factory
            try:
                sct = factory(**kwargs)
            except TypeError:  # non-Linux mss has no display argument
                sct = factory()
            self._local.sct = sct
        return sct

    def size(self) -> tuple[int, int]:
        mon = self._sct().monitors[0]
        return mon["width"], mon["height"]

    def grab(self) -> Image.Image:
        sct = self._sct()
        shot = sct.grab(sct.monitors[0])  # monitor 0 = the whole virtual screen (all monitors)
        return Image.frombytes("RGB", shot.size, shot.bgra, "raw", "BGRX")


class CommandCapture(ScreenCapture):
    """Wayland fallback: shells out to grim (wlroots) or gnome-screenshot. Slower (~150-400 ms)."""

    def __init__(self) -> None:
        if shutil.which("grim"):
            self.cmd = ["grim", "{out}"]
        elif shutil.which("gnome-screenshot"):
            self.cmd = ["gnome-screenshot", "-f", "{out}"]
        else:
            raise RuntimeError(
                "No Wayland screenshot tool found. Install grim or gnome-screenshot, or log in to 'Ubuntu on Xorg'."
            )
        self._size: tuple[int, int] | None = None

    def grab(self) -> Image.Image:
        with tempfile.NamedTemporaryFile(suffix=".png") as f:
            subprocess.run([c.replace("{out}", f.name) for c in self.cmd], check=True, timeout=10, capture_output=True)
            img = Image.open(f.name).convert("RGB")
            img.load()
        self._size = img.size
        return img

    def size(self) -> tuple[int, int]:
        return self._size or self.grab().size


def image_hash(img: Image.Image) -> str:
    """Cheap perceptual-ish hash: a 64x36 grayscale thumbnail. Equal hashes = visually unchanged screen."""
    thumb = img.convert("L").resize((64, 36))
    return hashlib.sha1(thumb.tobytes()).hexdigest()[:16]


def diff_ratio(a: Image.Image | None, b: Image.Image | None) -> float:
    """Fraction (0..1) of a 160x90 thumbnail that changed noticeably between two screenshots."""
    if a is None or b is None:
        return 1.0
    import numpy as np

    ta = np.asarray(a.convert("L").resize((160, 90)), dtype=np.int16)
    tb = np.asarray(b.convert("L").resize((160, 90)), dtype=np.int16)
    return float((np.abs(ta - tb) > 18).mean())


def downscale(img: Image.Image, width: int) -> tuple[Image.Image, float]:
    """Resize for the model; returns (image, scale) where model_px * scale = screen_px."""
    if img.width <= width:
        return img, 1.0
    scale = img.width / width
    return img.resize((width, round(img.height / scale)), Image.LANCZOS), scale


def to_jpeg(img: Image.Image, width: int = 1280, quality: int = 70) -> bytes:
    small, _ = downscale(img, width)
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def to_png(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
