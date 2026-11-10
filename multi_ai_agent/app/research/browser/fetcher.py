"""Async HTTP page fetching with size caps, manual redirect handling (each hop re-validated) and errors."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from urllib.parse import urljoin

import httpx

from app.research.browser.safety import Resolver, system_resolver, validate_url
from app.research.errors import UnsafeURLError

logger = logging.getLogger(__name__)

_HTML_TYPES = ("text/html", "application/xhtml+xml")
_TEXT_TYPES = ("text/plain", "text/markdown", "application/json")
MAX_REDIRECTS = 5


@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status_code: int | None = None
    content_type: str = ""
    body: str = ""
    redirects: list[str] = field(default_factory=list)
    error: str | None = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None and self.status_code is not None and self.status_code < 400

    @property
    def is_html(self) -> bool:
        return self.content_type.startswith(_HTML_TYPES)

    @property
    def is_text(self) -> bool:
        return self.content_type.startswith(_TEXT_TYPES)


class HttpFetcher:
    def __init__(
        self,
        *,
        timeout: float,
        max_bytes: int,
        user_agent: str,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver = system_resolver,
    ) -> None:
        self.max_bytes = max_bytes
        self.resolver = resolver
        self.client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=min(10.0, timeout)),
            headers={
                "User-Agent": user_agent,
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.9,*/*;q=0.5",
                "Accept-Language": "en-US,en;q=0.8",
            },
            follow_redirects=False,
        )

    async def fetch(self, url: str) -> FetchResult:
        result = FetchResult(url=url)
        current = url
        try:
            for _ in range(MAX_REDIRECTS + 1):
                await validate_url(current, self.resolver)
                async with self.client.stream("GET", current) as resp:
                    if resp.is_redirect and resp.headers.get("location"):
                        nxt = urljoin(current, resp.headers["location"])
                        result.redirects.append(nxt)
                        current = nxt
                        continue
                    result.final_url = str(resp.url)
                    result.status_code = resp.status_code
                    result.content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                    declared = int(resp.headers.get("content-length") or 0)
                    if declared > self.max_bytes:
                        result.error = f"page too large ({declared / 1e6:.2f} MB > {self.max_bytes / 1e6:.2f} MB limit)"
                        return result
                    buf = bytearray()
                    async for chunk in resp.aiter_bytes():
                        buf.extend(chunk)
                        if len(buf) > self.max_bytes:
                            result.truncated = True
                            break
                    encoding = resp.encoding or "utf-8"
                    result.body = bytes(buf[: self.max_bytes]).decode(encoding, errors="replace")
                    return result
            result.error = f"too many redirects (>{MAX_REDIRECTS})"
        except UnsafeURLError as exc:
            result.error = exc.message
        except httpx.TimeoutException:
            result.error = "timed out"
        except httpx.HTTPError as exc:
            result.error = f"{exc.__class__.__name__}: {exc}" if str(exc) else exc.__class__.__name__
        return result

    async def aclose(self) -> None:
        await self.client.aclose()
