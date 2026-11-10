"""Headless Chromium rendering for JavaScript-heavy pages (optional: pip install playwright && playwright install chromium).

Only used to render public pages. It does not solve CAPTCHAs, log in, or get around paywalls.
"""

from __future__ import annotations

import asyncio
import logging

from app.research.browser.fetcher import FetchResult
from app.research.browser.safety import Resolver, system_resolver, validate_url
from app.research.errors import UnsafeURLError

logger = logging.getLogger(__name__)


def playwright_available() -> bool:
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        return False
    return True


class PlaywrightFetcher:
    def __init__(self, *, timeout: float, user_agent: str, resolver: Resolver = system_resolver) -> None:
        self.timeout_ms = int(timeout * 1000)
        self.user_agent = user_agent
        self.resolver = resolver
        self._pw = None
        self._browser = None
        self._lock = asyncio.Lock()

    async def _ensure(self):
        async with self._lock:
            if self._browser is None:
                from playwright.async_api import async_playwright

                self._pw = await async_playwright().start()
                self._browser = await self._pw.chromium.launch(headless=True)
        return self._browser

    async def fetch(self, url: str) -> FetchResult:
        result = FetchResult(url=url)
        try:
            await validate_url(url, self.resolver)
            browser = await self._ensure()
            context = await browser.new_context(user_agent=self.user_agent, java_script_enabled=True)
            try:
                page = await context.new_page()

                async def guard(route):  # re-validate every navigation/sub-request target (no intranet access)
                    try:
                        await validate_url(route.request.url, self.resolver)
                    except UnsafeURLError:
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", guard)
                resp = await page.goto(url, timeout=self.timeout_ms, wait_until="domcontentloaded")
                try:
                    await page.wait_for_load_state("networkidle", timeout=min(8000, self.timeout_ms))
                except Exception:  # noqa: BLE001 - busy pages never go idle; use what rendered
                    pass
                result.status_code = resp.status if resp else None
                result.final_url = page.url
                result.content_type = "text/html"
                result.body = await page.content()
            finally:
                await context.close()
        except UnsafeURLError as exc:
            result.error = exc.message
        except Exception as exc:  # noqa: BLE001 - browser errors become a failed source, not a crash
            result.error = f"browser: {exc.__class__.__name__}: {str(exc)[:200]}"
        return result

    async def aclose(self) -> None:
        if self._browser is not None:
            await self._browser.close()
        if self._pw is not None:
            await self._pw.stop()
        self._browser = self._pw = None
