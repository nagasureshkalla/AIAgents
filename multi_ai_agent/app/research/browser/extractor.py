"""HTML -> clean document text and metadata (BeautifulSoup + lxml).

Removes scripts, navigation, ads, cookie banners and other page chrome; keeps headings, paragraphs,
lists, tables (as Markdown), code and the numbers/dates inside them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

_DROP_TAGS = (
    "script", "style", "noscript", "template", "svg", "canvas", "iframe", "object", "embed",
    "form", "button", "input", "select", "textarea", "nav", "footer", "aside", "dialog", "link", "meta",
)
_DROP_ROLES = {"navigation", "banner", "contentinfo", "complementary", "dialog", "alertdialog", "search", "menu", "menubar"}
_BOILERPLATE = re.compile(
    r"(?:^|[-_\s])(?:cookies?|consent|gdpr|cmp|advert\w*|ads?|ad-?slot|dfp|sponsor(?:ed)?|promo\w*|newsletter|"
    r"subscribe|signup|social|share|sharing|related|recommend\w*|comments?|breadcrumbs?|sidebar|popup|modal|"
    r"overlay|menu|navbar|nav|footer|masthead|toolbar|skip-link|print-only|visually-hidden|sr-only)(?:$|[-_\s])",
    re.I,
)
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_BLOCKS = {"p", "pre", "blockquote", "figcaption", "dt", "dd", "address", "caption", "summary"}
_WS = re.compile(r"[ \t\r\f\v ]+")
_DATE_META = (
    "article:published_time", "og:published_time", "datePublished", "date", "pubdate", "publish-date",
    "dc.date", "dc.date.issued", "dcterms.created", "sailthru.date", "parsely-pub-date",
)
_AUTHOR_META = ("author", "article:author", "dc.creator", "parsely-author", "sailthru.author", "twitter:creator")


@dataclass
class Extracted:
    title: str = ""
    text: str = ""
    headings: list[str] = field(default_factory=list)
    author: str | None = None
    published_date: str | None = None
    description: str | None = None
    links: list[tuple[str, str]] = field(default_factory=list)  # (href, anchor text) for "follow links"


def _clean(text: str) -> str:
    return _WS.sub(" ", text or "").strip()


def _meta(soup: BeautifulSoup, names: tuple[str, ...]) -> str | None:
    for name in names:
        tag = soup.find("meta", attrs={"property": name}) or soup.find("meta", attrs={"name": name}) or soup.find(
            "meta", attrs={"itemprop": name}
        )
        if isinstance(tag, Tag) and tag.get("content"):
            return _clean(str(tag["content"]))
    return None


def _json_ld(soup: BeautifulSoup) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (TypeError, ValueError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data]) if isinstance(data, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            out.setdefault("datePublished", item.get("datePublished") or item.get("dateCreated"))
            out.setdefault("headline", item.get("headline"))
            author = item.get("author")
            if isinstance(author, list) and author:
                author = author[0]
            if isinstance(author, dict):
                author = author.get("name")
            if isinstance(author, str):
                out.setdefault("author", author)
    return {k: v for k, v in out.items() if v}


def _table_markdown(table: Tag, max_rows: int = 40) -> str:
    rows: list[list[str]] = []
    for tr in table.find_all("tr"):
        cells = [_clean(c.get_text(" ")).replace("|", "/") for c in tr.find_all(["th", "td"])]
        if any(cells):
            rows.append(cells)
        if len(rows) >= max_rows:
            break
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    caption = table.find("caption")
    if isinstance(caption, Tag) and _clean(caption.get_text()):
        lines.insert(0, f"Table: {_clean(caption.get_text())}")
    return "\n".join(lines)


def _strip_boilerplate(root: Tag) -> None:
    for tag in root.find_all(_DROP_TAGS):
        tag.decompose()
    total = len(root.get_text(" ", strip=True)) or 1
    for tag in list(root.find_all(True)):
        if getattr(tag, "decomposed", False) or tag.parent is None:
            continue
        if tag.name in ("html", "body", "main", "article"):
            continue
        attrs = tag.attrs or {}
        role = str(attrs.get("role", "")).lower()
        hidden = attrs.get("hidden") is not None or attrs.get("aria-hidden") == "true" or re.search(
            r"display\s*:\s*none", str(attrs.get("style", "")), re.I
        )
        marker = " ".join([*(attrs.get("class") or []), str(attrs.get("id", ""))])
        if role in _DROP_ROLES or hidden or (tag.name == "header" and tag.find("nav")) or (marker and _BOILERPLATE.search(marker)):
            # Never drop a wrapper that holds most of the page (e.g. <div class="content-with-sidebar">).
            if len(tag.get_text(" ", strip=True)) < 0.5 * total:
                tag.decompose()


def _pick_root(soup: BeautifulSoup) -> Tag:
    candidates = soup.find_all("article") or soup.find_all("main") or soup.find_all(attrs={"role": "main"})
    if candidates:
        best = max(candidates, key=lambda t: len(t.get_text(" ", strip=True)))
        if len(best.get_text(" ", strip=True)) > 400:
            return best
    return soup.body or soup


def _emit(node: Tag, out: list[str], headings: list[str]) -> None:
    for child in node.children:
        if isinstance(child, NavigableString):
            text = _clean(str(child))
            if len(text) > 40 and node.name not in ("ul", "ol", "table", "tr"):
                out.append(text)
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name in _HEADINGS:
            text = _clean(child.get_text(" "))
            if text:
                out.append(f"{'#' * min(int(name[1]) + 1, 6)} {text}")
                headings.append(text)
        elif name == "table":
            md = _table_markdown(child)
            if md:
                out.append(md)
        elif name in ("ul", "ol"):
            for i, li in enumerate(child.find_all("li", recursive=False), 1):
                text = _clean(li.get_text(" "))
                if text:
                    out.append(f"{f'{i}.' if name == 'ol' else '-'} {text}")
        elif name in _BLOCKS:
            text = _clean(child.get_text(" "))
            if text:
                out.append(f"> {text}" if name == "blockquote" else text)
        elif name == "br":
            continue
        else:
            _emit(child, out, headings)


def _dedupe(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = re.sub(r"\W+", " ", line.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def extract_html(html: str, base_url: str = "") -> Extracted:
    soup = BeautifulSoup(html, "lxml")
    ld = _json_ld(soup)
    og_title = _meta(soup, ("og:title", "twitter:title"))
    title_tag = soup.title.get_text() if soup.title else ""
    result = Extracted(
        title=_clean(og_title or title_tag or ld.get("headline", "")),
        author=_meta(soup, _AUTHOR_META) or ld.get("author"),
        published_date=_meta(soup, _DATE_META) or ld.get("datePublished"),
        description=_meta(soup, ("description", "og:description")),
    )
    if not result.published_date:
        time_tag = soup.find("time", attrs={"datetime": True})
        if isinstance(time_tag, Tag):
            result.published_date = str(time_tag["datetime"])

    for a in soup.find_all("a", href=True)[:400]:
        text = _clean(a.get_text(" "))
        href = str(a["href"])
        if text and not href.startswith(("#", "javascript:", "mailto:")):
            result.links.append((href, text[:120]))

    root = _pick_root(soup)
    _strip_boilerplate(root)
    lines: list[str] = []
    _emit(root, lines, result.headings)
    result.text = "\n\n".join(_dedupe(lines))
    if not result.title and result.headings:
        result.title = result.headings[0]
    return result


def extract_plain_text(text: str) -> Extracted:
    lines = [_clean(line) for line in text.splitlines()]
    body = "\n".join(_dedupe([line for line in lines if line]))
    return Extracted(title=lines[0][:120] if lines else "", text=body)
