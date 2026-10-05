"""Helpers for parsing free-form model output."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_BLOCK = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def extract_json(text: str) -> Any | None:
    """Best-effort extraction of the first JSON object/array in `text`."""
    if not text:
        return None
    candidates = [m.group(1) for m in _JSON_BLOCK.finditer(text)] + [text]
    for candidate in candidates:
        candidate = candidate.strip()
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = candidate.find(opener), candidate.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(candidate[start : end + 1])
                except json.JSONDecodeError:
                    continue
    return None


def split_list(text: str, limit: int = 20) -> list[str]:
    """Parse a comma/newline separated list such as 'cat, sofa, lamp' or bullet points."""
    items: list[str] = []
    seen: set[str] = set()
    for raw in re.split(r"[\n,;]", text or ""):
        item = _BULLET.sub("", raw).strip().strip(".").strip()
        key = item.lower()
        if not item or len(item) > 60 or key in seen:
            continue
        seen.add(key)
        items.append(item)
        if len(items) >= limit:
            break
    return items


def bullet_lines(text: str, limit: int = 10) -> list[str]:
    lines = [_BULLET.sub("", line).strip() for line in (text or "").splitlines()]
    return [line for line in lines if line][:limit]


def split_conclusion(text: str) -> tuple[str, str]:
    """Split '...reasoning... Conclusion: answer' into (reasoning, conclusion)."""
    match = re.search(r"(?im)^\s*\**\s*(?:final\s+)?conclusion\s*\**\s*[:\-]\s*", text or "")
    if not match:
        return text.strip(), ""
    return text[: match.start()].strip(), text[match.end() :].strip()


def max_sentence_repeats(text: str) -> int:
    sentences = [s.strip().lower() for s in re.split(r"(?<=[.!?])\s+|\n+", text or "") if len(s.strip()) > 10]
    if not sentences:
        return 0
    counts: dict[str, int] = {}
    for s in sentences:
        counts[s] = counts.get(s, 0) + 1
    return max(counts.values())


def truncate(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."
