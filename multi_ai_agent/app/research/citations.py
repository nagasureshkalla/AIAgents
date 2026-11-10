"""Citation bookkeeping. Sources are numbered only after the browser has actually fetched them, and every
[n] marker in model output is checked against that list: the model never supplies URLs."""

from __future__ import annotations

import re

from app.research.retrieval.deduplicator import canonical_url
from app.research.schemas import Document, SourceRef

_MARKER = re.compile(r"\[(\d{1,3}(?:\s*[,–-]\s*\d{1,3})*)\]")
_URL = re.compile(r"https?://\S+")


class CitationManager:
    def __init__(self) -> None:
        self._sources: dict[int, SourceRef] = {}
        self._by_url: dict[str, int] = {}

    def register(self, doc: Document) -> int:
        """Assign (or return) the citation number of a successfully fetched document."""
        if not doc.ok:
            raise ValueError(f"Only fetched, readable pages can be cited (got {doc.status}: {doc.url})")
        key = canonical_url(doc.final_url or doc.url)
        if key in self._by_url:
            return self._by_url[key]
        sid = len(self._sources) + 1
        self._sources[sid] = SourceRef(
            id=sid,
            url=doc.final_url or doc.url,
            title=doc.title or doc.domain,
            domain=doc.domain,
            published_date=doc.published_date,
            author=doc.author,
            retrieved_at=doc.retrieved_at,
        )
        self._by_url[key] = sid
        self._by_url.setdefault(canonical_url(doc.url), sid)
        return sid

    def has(self, sid: int) -> bool:
        return sid in self._sources

    def get(self, sid: int) -> SourceRef:
        return self._sources[sid]

    def id_for_url(self, url: str) -> int | None:
        return self._by_url.get(canonical_url(url))

    @property
    def sources(self) -> list[SourceRef]:
        return [self._sources[k] for k in sorted(self._sources)]

    @staticmethod
    def _expand(group: str) -> list[int]:
        out: list[int] = []
        for part in re.split(r"\s*,\s*", group):
            if re.search(r"[–-]", part):
                lo, hi = (int(x) for x in re.split(r"\s*[–-]\s*", part)[:2])
                out.extend(range(lo, min(hi, lo + 20) + 1))
            elif part.strip().isdigit():
                out.append(int(part))
        return out

    def sanitize(self, text: str) -> tuple[str, list[int], list[int]]:
        """Keep only [n] markers that refer to visited sources and strip model-written URLs.

        Returns (clean text, cited ids in order of first use, invalid ids that were removed).
        """
        cited: list[int] = []
        invalid: list[int] = []

        def repl(match: re.Match[str]) -> str:
            ids = self._expand(match.group(1))
            good = [i for i in ids if i in self._sources]
            invalid.extend(i for i in ids if i not in self._sources)
            for i in good:
                if i not in cited:
                    cited.append(i)
            return "".join(f"[{i}]" for i in dict.fromkeys(good))

        clean = _MARKER.sub(repl, text)
        clean = _URL.sub("", clean)  # URLs only ever come from the Sources list built here
        clean = re.sub(r"[ \t]+([.,;:])", r"\1", clean)
        clean = re.sub(r"[ \t]{2,}", " ", clean)
        for sid in cited:
            self._sources[sid].cited = True
        return clean.strip(), cited, sorted(set(invalid))

    def render_sources(self, ids: list[int] | None = None) -> str:
        chosen = [self._sources[i] for i in (ids or sorted(self._sources)) if i in self._sources]
        lines = []
        for s in chosen:
            meta = " · ".join(x for x in (s.domain, s.published_date and s.published_date[:10]) if x)
            lines.append(f"[{s.id}] {s.title} — {meta}\n    {s.url}")
        return "\n".join(lines)
