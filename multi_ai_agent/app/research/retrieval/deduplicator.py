"""URL canonicalization and near-duplicate text detection across sources."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from app.research.schemas import Chunk

_TRACKING = re.compile(r"^(utm_\w+|fbclid|gclid|mc_cid|mc_eid|ref|ref_src|igshid|yclid|_hs\w+)$", re.I)


def canonical_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    query = urlencode(sorted((k, v) for k, v in parse_qsl(parts.query) if not _TRACKING.match(k)))
    path = parts.path.rstrip("/") or "/"
    return urlunsplit(("https" if parts.scheme in ("http", "https") else parts.scheme, host, path, query, ""))


def _shingles(text: str, size: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {hash(" ".join(words[i : i + size])) for i in range(max(1, len(words) - size + 1))}


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def dedupe_chunks(chunks: list[Chunk], threshold: float = 0.8) -> list[Chunk]:
    """Drop chunks that are near-copies of an earlier (higher-priority) chunk, e.g. syndicated articles."""
    kept: list[tuple[Chunk, set[int]]] = []
    for chunk in chunks:
        sig = _shingles(chunk.text)
        if any(jaccard(sig, other) >= threshold for _, other in kept):
            continue
        kept.append((chunk, sig))
    return [c for c, _ in kept]
