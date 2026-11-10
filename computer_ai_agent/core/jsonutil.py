"""Pull a JSON object out of model output that may include <think> blocks, code fences or chatter."""

from __future__ import annotations

import json
import re
from typing import Any

_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S | re.I)


def _balanced(text: str, start: int) -> str | None:
    open_ch = text[start]
    close_ch = "}" if open_ch == "{" else "]"
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        c = text[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def extract_json(text: str) -> Any:
    """Best-effort: returns a dict/list, or None when nothing parseable is found."""
    if not text:
        return None
    text = _THINK.sub("", text).strip()
    candidates = [m.group(1).strip() for m in _FENCE.finditer(text)] + [text]
    for cand in candidates:
        try:
            return json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            pass
        for i, c in enumerate(cand):
            if c in "{[":
                chunk = _balanced(cand, i)
                if chunk:
                    for attempt in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):  # tolerate trailing commas
                        try:
                            return json.loads(attempt)
                        except (json.JSONDecodeError, ValueError):
                            continue
    return None
