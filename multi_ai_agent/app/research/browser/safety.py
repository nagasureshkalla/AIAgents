"""URL safety, robots.txt and access-restriction detection.

The agent reads public pages only. It refuses private-network targets (SSRF), honours robots.txt, and
treats CAPTCHAs, login walls and paywalls as "blocked": they are reported, never bypassed.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import re
import socket
from collections.abc import Awaitable, Callable
from urllib import robotparser
from urllib.parse import urlsplit

import httpx

from app.research.errors import UnsafeURLError

logger = logging.getLogger(__name__)

Resolver = Callable[[str], Awaitable[list[str]]]

_ALLOWED_SCHEMES = {"http", "https"}
_ALLOWED_PORTS = {None, 80, 443, 8080, 8443}


async def system_resolver(host: str) -> list[str]:
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    return sorted({info[4][0] for info in infos})


def _is_public_ip(value: str) -> bool:
    ip = ipaddress.ip_address(value.split("%", 1)[0])
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
        or (isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None and not _is_public_ip(str(ip.ipv4_mapped)))
    )


async def validate_url(url: str, resolver: Resolver = system_resolver) -> str:
    """Return the URL if it is a public http(s) address; raise UnsafeURLError otherwise."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise UnsafeURLError(f"Invalid URL: {url!r}") from exc
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Only http(s) URLs can be opened: {url!r}")
    host = (parts.hostname or "").lower()
    if not host:
        raise UnsafeURLError(f"URL has no host: {url!r}")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs with embedded credentials are not opened")
    if port not in _ALLOWED_PORTS:
        raise UnsafeURLError(f"Non-standard port {port} is not allowed")
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal", ".lan")):
        raise UnsafeURLError(f"Local network host {host!r} is not allowed")
    try:
        ipaddress.ip_address(host)
        addresses = [host]
    except ValueError:
        try:
            addresses = await resolver(host)
        except OSError as exc:
            raise UnsafeURLError(f"Could not resolve {host!r}: {exc}") from exc
    if not addresses:
        raise UnsafeURLError(f"Could not resolve {host!r}")
    for addr in addresses:
        if not _is_public_ip(addr):
            raise UnsafeURLError(f"{host!r} resolves to a non-public address ({addr}); refusing to open it")
    return url.strip()


class RobotsChecker:
    """Caches robots.txt per origin. Unreachable robots.txt files are treated as allow-all."""

    def __init__(self, client: httpx.AsyncClient, user_agent: str) -> None:
        self.client = client
        self.user_agent = user_agent
        self._cache: dict[str, robotparser.RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}"
        lock = self._locks.setdefault(origin, asyncio.Lock())
        async with lock:
            if origin not in self._cache:
                self._cache[origin] = await self._load(origin)
        parser = self._cache[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, url)

    async def _load(self, origin: str) -> robotparser.RobotFileParser | None:
        try:
            resp = await self.client.get(f"{origin}/robots.txt", timeout=8, follow_redirects=True)
        except httpx.HTTPError:
            return None
        if resp.status_code >= 400:
            return None
        parser = robotparser.RobotFileParser()
        parser.parse(resp.text.splitlines())
        return parser


_CAPTCHA = re.compile(
    r"g-recaptcha|hcaptcha|cf-turnstile|challenge-platform|cf_chl_|"
    r"<title>\s*(?:just a moment|attention required|access denied|are you a robot)|"
    r"verify (?:that )?you are (?:a )?human|unusual traffic from your computer",
    re.I,
)
_PAYWALL = re.compile(
    r"subscribe (?:now )?to (?:continue|keep) reading|this (?:article|content) is for subscribers|"
    r"you(?:'ve| have) reached your (?:free )?article limit|paywall-(?:overlay|container)",
    re.I,
)
_LOGIN = re.compile(r"<input[^>]+type=[\"']?password", re.I)


def detect_restriction(html: str, status_code: int | None, visible_text_len: int) -> str | None:
    """Return a reason if the page is a CAPTCHA, login wall or paywall (which we must not bypass)."""
    if status_code in (401, 407):
        return "login required"
    if status_code == 403:
        return "access denied (HTTP 403)"
    if status_code == 429:
        return "rate limited (HTTP 429)"
    head = html[:200_000]
    if _CAPTCHA.search(head):
        return "CAPTCHA / bot challenge"
    if _PAYWALL.search(head) and visible_text_len < 2500:
        return "paywall"
    if _LOGIN.search(head) and visible_text_len < 800:
        return "login wall"
    return None
