"""Browser facade: open a URL -> clean Document. httpx first, Playwright for thin JavaScript pages."""

from __future__ import annotations

import logging
from urllib.parse import urljoin

import httpx

from app.core.config import ResearchConfig
from app.research.browser.extractor import extract_html, extract_plain_text
from app.research.browser.fetcher import FetchResult, HttpFetcher
from app.research.browser.playwright_fetcher import PlaywrightFetcher, playwright_available
from app.research.browser.safety import Resolver, RobotsChecker, detect_restriction, system_resolver
from app.research.errors import WebDisabledError
from app.research.schemas import Document, FetchStatus, domain_of

logger = logging.getLogger(__name__)

THIN_PAGE_CHARS = 400  # below this, a page is probably rendered client-side


class Browser:
    def __init__(
        self,
        cfg: ResearchConfig,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver = system_resolver,
        js_fetcher: PlaywrightFetcher | None = None,
    ) -> None:
        self.cfg = cfg
        self.http = HttpFetcher(
            timeout=cfg.request_timeout,
            max_bytes=int(cfg.max_page_mb * 1024 * 1024),
            user_agent=cfg.user_agent,
            client=client,
            resolver=resolver,
        )
        self.robots = RobotsChecker(self.http.client, cfg.user_agent) if cfg.respect_robots else None
        if js_fetcher is None and cfg.use_playwright != "never" and playwright_available():
            js_fetcher = PlaywrightFetcher(timeout=cfg.request_timeout, user_agent=cfg.user_agent, resolver=resolver)
        self.js = js_fetcher

    async def open(self, url: str) -> Document:
        if not self.cfg.web_enabled:
            raise WebDisabledError("Web access is disabled (LOCAL AI + OFFLINE mode)")
        doc = Document(url=url, domain=domain_of(url))
        if self.robots is not None and not await self.robots.allowed(url):
            doc.status, doc.error = FetchStatus.BLOCKED, "disallowed by robots.txt"
            return doc

        fetched = await (self.js.fetch(url) if self.js and self.cfg.use_playwright == "always" else self.http.fetch(url))
        doc = self._to_document(url, fetched)
        if (
            self.js is not None
            and self.cfg.use_playwright == "auto"
            and fetched.ok
            and fetched.is_html
            and doc.status == FetchStatus.OK
            and len(doc.text) < THIN_PAGE_CHARS
        ):
            rendered = self._to_document(url, await self.js.fetch(url))
            if rendered.ok and len(rendered.text) > len(doc.text):
                rendered.renderer = "playwright"
                rendered.redirects = doc.redirects + rendered.redirects
                doc = rendered
        return doc

    def _to_document(self, url: str, fetched: FetchResult) -> Document:
        doc = Document(
            url=url,
            final_url=fetched.final_url or url,
            domain=domain_of(fetched.final_url or url),
            status_code=fetched.status_code,
            redirects=fetched.redirects,
        )
        if fetched.error:
            doc.status, doc.error = FetchStatus.ERROR, fetched.error
            return doc
        restriction = None
        if fetched.is_html:
            extracted = extract_html(fetched.body, doc.final_url)
            restriction = detect_restriction(fetched.body, fetched.status_code, len(extracted.text))
        elif fetched.is_text:
            extracted = extract_plain_text(fetched.body)
        else:
            doc.status, doc.error = FetchStatus.UNSUPPORTED, f"unsupported content type {fetched.content_type or '?'}"
            return doc
        if restriction is None and fetched.status_code is not None and fetched.status_code >= 400:
            doc.status, doc.error = FetchStatus.ERROR, f"HTTP {fetched.status_code}"
            return doc
        if restriction:
            doc.status, doc.error = FetchStatus.BLOCKED, restriction
            return doc
        doc.title = extracted.title or doc.domain
        doc.text = extracted.text
        doc.headings = extracted.headings
        doc.author = extracted.author
        doc.published_date = extracted.published_date
        doc.description = extracted.description
        doc.links = [(urljoin(doc.final_url, href), text) for href, text in extracted.links]
        if not doc.text.strip():
            doc.status, doc.error = FetchStatus.ERROR, "no readable text"
        return doc

    async def aclose(self) -> None:
        await self.http.aclose()
        if self.js is not None:
            await self.js.aclose()
