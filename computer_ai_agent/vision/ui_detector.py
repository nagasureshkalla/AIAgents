"""Builds an Observation of the screen: accessibility first, OCR when accessibility is thin, CV boxes last.

Results are cached by screenshot hash + active window, so an unchanged screen costs one ~20 ms capture."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.types import Element, Observation, WindowInfo
from vision.screen_capture import diff_ratio, image_hash

if TYPE_CHECKING:
    from actions.platform import Desktop
    from config.settings import Settings
    from vision.accessibility import AccessibilityReader
    from vision.ocr import OcrEngine

logger = logging.getLogger(__name__)

INTERACTIVE = {"button", "checkbox", "radio", "menuitem", "menu", "combobox", "textbox", "password", "spinbutton",
               "slider", "link", "tab", "listitem", "treeitem", "icon", "terminal"}
MIN_A11Y_INTERACTIVE = 6  # fewer interactive a11y elements than this => the tree is too thin, add OCR


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int], pad: int = 4) -> bool:
    ox, oy, ow, oh = outer
    ix, iy, iw, ih = inner
    return ox - pad <= ix and oy - pad <= iy and ix + iw <= ox + ow + pad and iy + ih <= oy + oh + pad


def detect_boxes(image: Image.Image, region: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    """Classic CV: rectangular outlines that look like input fields or buttons (used only without accessibility)."""
    import cv2

    x0, y0, w, h = region
    x0, y0 = max(0, x0), max(0, y0)
    crop = np.asarray(image.crop((x0, y0, x0 + w, y0 + h)).convert("L"))
    if crop.size == 0:
        return []
    edges = cv2.Canny(crop, 40, 120)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[tuple[int, int, int, int]] = []
    for c in contours:
        bx, by, bw, bh = cv2.boundingRect(c)
        if bw < 24 or bh < 14 or bh > 90 or bw > w * 0.95:
            continue
        approx = cv2.approxPolyDP(c, 0.02 * cv2.arcLength(c, True), True)
        if len(approx) < 4 or len(approx) > 8:
            continue
        if cv2.contourArea(c) < 0.6 * bw * bh:
            continue
        box = (bx + x0, by + y0, bw, bh)
        if all(_iou(box, b) < 0.6 for b in boxes):
            boxes.append(box)
    return boxes[:60]


class UIDetector:
    def __init__(
        self,
        desktop: Desktop,
        settings: Settings,
        ocr: OcrEngine | None = None,
        accessibility: AccessibilityReader | None = None,
    ) -> None:
        self.desktop = desktop
        self.settings = settings
        self.ocr = ocr
        self.a11y = accessibility
        self._cache_key: tuple[str, int | None, str] | None = None
        self._cache: Observation | None = None
        self.last_timing: dict[str, float] = {}

    def observe(self, force_ocr: bool = False, use_cache: bool = True, target: WindowInfo | None = None) -> Observation:
        """Capture and understand the screen. `target` limits element detection to one window (default: active)."""
        t0 = time.perf_counter()
        image = self.desktop.screen.grab()
        t_grab = time.perf_counter()
        active = self.desktop.windows.active_window()
        win = target or active
        h = image_hash(image)
        key = (h, win.id if win else None, win.title if win else "")
        # A blinking caret or ticking clock changes the exact hash; treat <0.3% changed pixels as "same screen".
        same_screen = (
            self._cache is not None
            and self._cache_key is not None
            and key[1:] == self._cache_key[1:]
            and (key[0] == self._cache_key[0] or diff_ratio(self._cache.image, image) < 0.003)
        )
        if use_cache and not force_ocr and same_screen and self._cache is not None:
            cached = self._cache
            cached.timestamp, cached.active_window = time.time(), active  # keep the reference frame
            self.last_timing = {"grab_ms": (t_grab - t0) * 1000, "cached": 1}
            return cached

        region = win.geometry if win and win.geometry[2] > 0 else (0, 0, image.width, image.height)
        raw: list[Element] = []
        sources: list[str] = []
        focused: Element | None = None

        # 1) accessibility
        t_a = time.perf_counter()
        a11y_interactive = 0
        if self.a11y is not None and self.desktop.accessibility and win and win.pid:
            try:
                items, _found = self.a11y.read(win.pid, win.title, region)
            except Exception as exc:  # noqa: BLE001 - a11y bus hiccup: fall back to pixels
                logger.debug("accessibility read failed: %s", exc)
                items, _found = [], False
            if items:
                sources.append("accessibility")
            for it in items:
                el = Element(0, it.type, it.text, it.bbox, "accessibility", it.states, 1.0, win.title)
                raw.append(el)
                if it.type in INTERACTIVE:
                    a11y_interactive += 1
                if "focused" in it.states and it.type not in ("dialog",):
                    focused = el
        # A button's inner label repeats the button: keep only the control the planner can act on.
        controls = [e for e in raw if e.type in INTERACTIVE]
        raw = [
            e for e in raw
            if e.type != "text"
            or not any(c is not e and _contains(c.bbox, e.bbox) and e.text.lower() in c.text.lower() for c in controls)
        ]
        t_a_end = time.perf_counter()

        # 2) OCR when accessibility is missing or thin (web pages without a11y, canvas apps, games, Electron...)
        words_n = 0
        if self.ocr is not None and self.settings.ocr_enabled and (force_ocr or a11y_interactive < MIN_A11Y_INTERACTIVE):
            try:
                words = self.ocr.read(image, region)
            except Exception as exc:  # noqa: BLE001
                logger.warning("OCR failed: %s", exc)
                words = []
            words_n = len(words)
            if words:
                sources.append("ocr")
            for w in words:
                dup = any(
                    (_iou(w.bbox, e.bbox) > 0.4 or _contains(e.bbox, w.bbox))
                    and (w.text.lower() in e.text.lower() or e.text.lower() in w.text.lower())
                    for e in raw
                )
                if not dup:
                    raw.append(Element(0, "text", w.text, w.bbox, "ocr", [], round(w.score, 2), win.title if win else ""))
        t_ocr = time.perf_counter()

        # 3) CV boxes (likely fields/buttons) only when there is no accessibility at all
        if "accessibility" not in sources:
            try:
                for b in detect_boxes(image, region):
                    if not any(_contains(b, e.bbox) and e.type != "text" for e in raw) and not any(
                        _iou(b, e.bbox) > 0.5 for e in raw
                    ):
                        inner = [e.text for e in raw if e.source == "ocr" and _contains(b, e.bbox)]
                        kind = "button" if inner and b[2] < 260 else "box"
                        raw.append(Element(0, kind, " ".join(inner)[:60], b, "vision", [], 0.4, win.title if win else ""))
                sources.append("vision")
            except Exception as exc:  # noqa: BLE001
                logger.debug("CV box detection failed: %s", exc)

        raw.sort(key=lambda e: (round(e.bbox[1] / 12), e.bbox[0]))
        for i, e in enumerate(raw, start=1):
            e.id = i
        obs = Observation(
            timestamp=time.time(),
            screen_size=image.size,
            active_window=active,
            elements=raw,
            image=image,
            image_hash=h,
            ocr_text=" | ".join(e.text for e in raw if e.source == "ocr")[:4000],
            focused=focused,
            sources=sources,
        )
        self._cache_key, self._cache = key, obs
        self.last_timing = {
            "grab_ms": (t_grab - t0) * 1000,
            "a11y_ms": (t_a_end - t_a) * 1000,
            "ocr_ms": (t_ocr - t_a_end) * 1000 if words_n or "ocr" in sources else 0.0,
            "total_ms": (time.perf_counter() - t0) * 1000,
            "elements": len(raw),
        }
        return obs

    def invalidate(self) -> None:
        self._cache_key = None


_FONT: ImageFont.ImageFont | None = None


def _font() -> ImageFont.ImageFont:
    global _FONT
    if _FONT is None:
        for path in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"):
            try:
                _FONT = ImageFont.truetype(path, 13)
                break
            except OSError:
                continue
        else:
            _FONT = ImageFont.load_default()
    return _FONT


def draw_marks(obs: Observation, width: int, region: tuple[int, int, int, int] | None = None,
               max_marks: int = 150) -> tuple[Image.Image, float, tuple[int, int]]:
    """Set-of-marks image for the vision model: numbered boxes on a (cropped, downscaled) screenshot.

    Returns (image, scale, origin): screen_px = origin + model_px * scale."""
    img = obs.image
    ox, oy = 0, 0
    if region:
        x, y, w, h = region
        x, y = max(0, x), max(0, y)
        img = img.crop((x, y, x + w, y + h))
        ox, oy = x, y
    scale = img.width / width if img.width > width else 1.0
    if scale != 1.0:
        img = img.resize((width, round(img.height / scale)), Image.LANCZOS)
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _font()
    colors = {"accessibility": (37, 99, 235), "ocr": (217, 119, 6), "vision": (124, 58, 237)}
    for e in obs.elements[:max_marks]:
        x, y, w, h = e.bbox
        bx, by = (x - ox) / scale, (y - oy) / scale
        if bx < -5 or by < -5 or bx > img.width or by > img.height:
            continue
        c = colors.get(e.source, (220, 38, 38))
        draw.rectangle([bx, by, bx + w / scale, by + h / scale], outline=c, width=2)
        label = str(e.id)
        tw = draw.textlength(label, font=font)
        draw.rectangle([bx, by - 15, bx + tw + 6, by], fill=c)
        draw.text((bx + 3, by - 15), label, fill=(255, 255, 255), font=font)
    return img, scale, (ox, oy)
