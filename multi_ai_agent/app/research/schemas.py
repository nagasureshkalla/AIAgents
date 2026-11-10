"""Data structures shared by the research pipeline."""

from __future__ import annotations

import time
from enum import StrEnum
from urllib.parse import urlsplit

from pydantic import BaseModel, Field


class SearchResult(BaseModel):
    url: str
    title: str = ""
    snippet: str = ""
    provider: str = ""
    query: str = ""
    rank: int = 0

    @property
    def domain(self) -> str:
        return domain_of(self.url)


class FetchStatus(StrEnum):
    OK = "ok"
    BLOCKED = "blocked"  # CAPTCHA, login wall, paywall or robots.txt: never bypassed
    ERROR = "error"
    UNSUPPORTED = "unsupported"


class Document(BaseModel):
    """A web page that the browser actually fetched, converted to clean text."""

    url: str
    final_url: str = ""
    title: str = ""
    domain: str = ""
    published_date: str | None = None
    author: str | None = None
    description: str | None = None
    text: str = ""
    headings: list[str] = Field(default_factory=list)
    retrieved_at: float = Field(default_factory=time.time)
    status: FetchStatus = FetchStatus.OK
    status_code: int | None = None
    error: str | None = None
    redirects: list[str] = Field(default_factory=list)
    links: list[tuple[str, str]] = Field(default_factory=list, exclude=True)  # (absolute url, anchor text)
    renderer: str = "httpx"

    @property
    def ok(self) -> bool:
        return self.status == FetchStatus.OK and bool(self.text.strip())


class Chunk(BaseModel):
    doc_url: str
    index: int
    text: str
    heading: str = ""
    score: float = 0.0


class Evidence(BaseModel):
    claim: str
    source_id: int  # citation number of a visited source
    quote: str = ""
    confidence: float = 0.7


class ClaimStatus(StrEnum):
    SUPPORTED = "SUPPORTED"
    CONFLICTING = "CONFLICTING"
    INSUFFICIENT = "INSUFFICIENT"


class VerifiedClaim(BaseModel):
    claim: str
    status: ClaimStatus
    sources: list[int] = Field(default_factory=list)
    note: str = ""


class SourceRef(BaseModel):
    """Public citation entry. Only ever built from a fetched Document."""

    id: int
    url: str
    title: str
    domain: str
    published_date: str | None = None
    author: str | None = None
    retrieved_at: float
    cited: bool = False


def domain_of(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host
