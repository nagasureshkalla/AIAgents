"""OCR with RapidOCR (PP-OCR models on ONNX Runtime): pip-installable, no tesseract binary, CUDA when
onnxruntime-gpu is installed."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class OcrWord:
    text: str
    bbox: tuple[int, int, int, int]  # absolute screen pixels
    score: float


class OcrEngine:
    def __init__(self, min_score: float = 0.5) -> None:
        self.min_score = min_score
        self._engine = None
        self._lock = threading.Lock()
        self.last_ms = 0.0

    def _load(self):  # noqa: ANN202
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR

            self._engine = RapidOCR()
            logger.info("OCR engine loaded (RapidOCR)")
        return self._engine

    def read(self, image: Image.Image, region: tuple[int, int, int, int] | None = None) -> list[OcrWord]:
        """OCR `image` (full screen); only `region` (x, y, w, h) is processed when given."""
        ox, oy = 0, 0
        if region:
            x, y, w, h = region
            x, y = max(0, x), max(0, y)
            w, h = min(w, image.width - x), min(h, image.height - y)
            if w < 8 or h < 8:
                return []
            image = image.crop((x, y, x + w, y + h))
            ox, oy = x, y
        arr = np.asarray(image.convert("RGB"))[:, :, ::-1]  # RapidOCR expects BGR like OpenCV
        t = time.perf_counter()
        with self._lock:
            result, _ = self._load()(arr)
        self.last_ms = (time.perf_counter() - t) * 1000
        words: list[OcrWord] = []
        for box, text, score in result or []:
            score = float(score)
            if score < self.min_score or not str(text).strip():
                continue
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x0, y0 = int(min(xs)), int(min(ys))
            words.append(OcrWord(str(text).strip(), (x0 + ox, y0 + oy, int(max(xs)) - x0, int(max(ys)) - y0), score))
        return words
