"""Turns a Web Tester run into TestingTree test cases and suites, uploads them and replays them there.

FlowRecorder: while the agent explores, every successful page action is recorded under the scenario the model named
("suite" + "name") as a TestingTree step (TestingTreeServer/Models/TestCaseDefinition.cs StepOperation). agent-browser
refs (@e5) only live inside one session, so each one is resolved from the snapshot it came from into a Playwright
selector TestingTree can replay: role=button[name="Subscribe"s]. Navigation and text the model read become asserts.

TestingTreeClient: finds or creates the site's project, POSTs the test cases (TestCasesController.Create), runs them
with TestingTree's own Playwright runner and reports each one's pass/fail.
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import httpx

# "- button "Subscribe" [ref=e4]", "- textbox "Email" [required, ref=e3]", "- link "Docs" [ref=e2, url=...]"
SNAPSHOT_LINE = re.compile(r'^\s*-\s+([a-z]+)(?:\s+"((?:[^"\\]|\\.)*)")?\s*\[([^\]]*)\]', re.M)
REF = re.compile(r"\bref=(e\d+)\b")
# agent-browser verb -> TestingTree StepOperation (press/scroll/get/is are handled separately)
CHECKS = ("AssertTextContains", "AssertElementVisible", "AssertFieldEquals")  # what the model verified itself
INPUTS = ("Type", "Select", "Check", "Uncheck")
ACTIONS = {"click": "Click", "dblclick": "Click", "fill": "Type", "type": "Type", "select": "Select",
           "check": "Check", "uncheck": "Uncheck", "hover": "Hover"}


def snapshot_refs(snapshot: str) -> dict[str, tuple[str, str]]:
    """ref -> (role, accessible name) for every element line of an agent-browser snapshot."""
    out: dict[str, tuple[str, str]] = {}
    for m in SNAPSHOT_LINE.finditer(snapshot):
        ref = REF.search(m.group(3))
        if ref:
            out[ref.group(1)] = (m.group(1), (m.group(2) or "").replace('\\"', '"'))
    return out


def role_selector(role: str, name: str) -> str:
    """A Playwright role selector with an exact, case-sensitive name match."""
    return f'role={role}[name="{name.replace(chr(34), chr(92) + chr(34))}"s]'


def url_fragment(url: str) -> str:
    """The part of a URL worth asserting on: its path (query strings echo typed data and vary between runs), or
    the site's origin for the root page."""
    p = urlparse(url)
    return p.path if p.path not in ("", "/") else f"{p.scheme}://{p.netloc}/"


def page_key(url: str) -> str:
    p = urlparse(url)
    return f"{p.scheme}://{p.netloc}{p.path or '/'}"


@dataclass
class Scenario:
    suite: str
    name: str
    start_url: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    issue: str | None = None  # an issue the model reported while in this scenario
    last_url: str = ""


class FlowRecorder:
    """Collects TestingTree test cases from the commands the agent ran."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.scenarios: list[Scenario] = []
        self.current: Scenario | None = None

    def begin(self, suite: Any, name: Any, url: str) -> None:
        suite, name = str(suite or "").strip()[:80], str(name or "").strip()[:120]
        if not name or (self.current and (self.current.suite, self.current.name) == (suite or "Exploration", name)):
            return
        carried = self._pending_inputs(url)
        self.current = Scenario(suite=suite or "Exploration", name=name, start_url=url, steps=carried, last_url=url)
        self.scenarios.append(self.current)

    def _pending_inputs(self, url: str) -> list[dict[str, Any]]:
        """Form inputs the previous flow made on this page but never submitted: a new flow named between filling a
        form and submitting it would otherwise replay the submit on an empty form."""
        sc = self.current
        if sc is None or not sc.steps or sc.last_url.split("#")[0] != url.split("#")[0]:
            return []
        tail: list[dict[str, Any]] = []
        for step in reversed(sc.steps):
            if step["operation"] not in ("Type", "Select", "Check", "Uncheck"):
                break
            tail.insert(0, dict(step))
        return tail

    def _ensure(self, url: str) -> Scenario:
        if self.current is None:  # the model did not name a flow: keep its actions as one exploration per page
            self.begin("Exploration", f"Explore {url_fragment(url)}", url)
        assert self.current is not None
        return self.current

    def note_issue(self, title: str) -> None:
        if self.current is not None and self.current.issue is None:
            self.current.issue = title

    def record(self, args: list[str], selector: str | None, label: str | None, url_before: str, url_after: str,
               output: str) -> None:
        """Record one successful command. `selector` is the replayable form of an @ref argument (None if none)."""
        verb = args[0]
        sc = self._ensure(url_before)
        op = ACTIONS.get(verb)
        if verb == "open":
            if sc.steps or url_before.split("#")[0] != args[1].split("#")[0]:
                sc.steps.append({"operation": "Navigate", "value": args[1]})
        elif op and selector:
            value = args[2] if op in ("Type", "Select") and len(args) > 2 else None
            sc.steps.append({"operation": op, "selector": selector, "elementLabel": label, "value": value})
        elif verb == "wait" and len(args) == 2 and args[1].isdigit():
            sc.steps.append({"operation": "Wait", "value": args[1]})
        elif verb == "reload":
            sc.steps.append({"operation": "Reload"})
        elif verb in ("back", "forward") and url_after:
            sc.steps.append({"operation": "Navigate", "value": url_after})
        elif verb == "get" and len(args) >= 2 and args[1] == "text":
            text = output.strip().splitlines()[-1].strip() if output.strip() else ""
            if 0 < len(text) <= 120:
                sc.steps.append({"operation": "AssertTextContains", "value": text})
        elif verb == "is" and len(args) >= 3 and args[1] == "visible" and selector and output.strip().endswith("true"):
            sc.steps.append({"operation": "AssertElementVisible", "selector": selector, "elementLabel": label})
        if verb in ("click", "dblclick", "press") and url_after and page_key(url_after) != page_key(url_before):
            sc.steps.append({"operation": "AssertUrlContains", "value": url_fragment(url_after)})
        sc.last_url = url_after or url_before

    @staticmethod
    def _clean(raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Model-named flow boundaries are approximate: drop what came after the flow's last real check (setup for
        the next flow, e.g. clicking 'Back home') and repeated identical steps."""
        steps = _dedupe_consecutive([dict(s) for s in raw])
        checks = [i for i, s in enumerate(steps) if s["operation"] in CHECKS]
        return steps[: checks[-1] + 1] if checks else steps

    def test_cases(self) -> list[dict[str, Any]]:
        """One test case per flow with at least one action. Every case ends with a check (where the flow ended up,
        if the model checked nothing), exact repeats are dropped and same-named variants get numbered."""
        out: list[dict[str, Any]] = []
        seen: set[str] = set()
        names: dict[tuple[str, str], int] = {}
        for sc in self.scenarios:
            steps = self._clean(sc.steps)
            actions = [s for i, s in enumerate(steps)
                       if not s["operation"].startswith("Assert") and s["operation"] != "Wait"
                       and not (i == 0 and s["operation"] == "Navigate")]
            if not actions:  # waiting, or only opening the start page again, tests nothing
                continue
            if all(s["operation"] in INPUTS for s in actions):
                continue  # inputs never submitted: an abandoned attempt (they were carried into the next flow)
            if not any(s["operation"] in CHECKS for s in steps) and all(
                    s["operation"] == "Click" and (s.get("selector") or "").startswith("role=link[") for s in actions):
                continue  # only followed links: the Navigation suite covers every working link
            if not any(s["operation"].startswith("Assert") for s in steps) and sc.last_url:
                steps.append({"operation": "AssertUrlContains", "value": url_fragment(sc.last_url)})
            key = repr((sc.suite, page_key(sc.start_url), [(s["operation"], s.get("selector"), s.get("value")) for s in steps]))
            if key in seen:
                continue
            seen.add(key)
            n = names[(sc.suite, sc.name)] = names.get((sc.suite, sc.name), 0) + 1
            name = sc.name if n == 1 else f"{sc.name} ({n})"
            if sc.issue:
                name = f"{name} (issue: {sc.issue})"[:200]
            out.append({"name": name, "startUrl": sc.start_url, "suite": sc.suite, "steps": steps})
        return out


def case_key(case: dict[str, Any]) -> str:
    """Identity of a test case for de-duplication across runs: suite, start page and steps (not the name)."""
    steps = [(s.get("operation"), s.get("selector") or None, s.get("value") or None) for s in case.get("steps") or []]
    return repr((case.get("suite") or "", page_key(case.get("startUrl") or ""), steps))


def _dedupe_consecutive(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for s in steps:
        if not out or out[-1] != s:
            out.append(s)
    return out


def navigation_suite(links: dict[str, tuple[str, str]], statuses: dict[str, int | None], limit: int = 15) -> list[dict]:
    """One test per working same-site link: from the page it was found on, click it and land on its URL."""
    out = []
    for url, (text, page) in links.items():
        status = statuses.get(url)
        if not text or status is None or status >= 400 or url.split("#")[0].rstrip("/") == page.split("#")[0].rstrip("/"):
            continue
        out.append({"name": f"Link '{text}' opens {url_fragment(url)}", "startUrl": page, "suite": "Navigation",
                    "steps": [{"operation": "Click", "selector": role_selector("link", text), "elementLabel": text},
                              {"operation": "AssertUrlContains", "value": url_fragment(url)}]})
        if len(out) >= limit:
            break
    return out


def accessibility_suite(pages: list[str], limit: int = 5) -> list[dict]:
    """One axe audit per distinct page (query strings ignored)."""
    unique = list(dict.fromkeys(page_key(p) for p in pages))
    return [{"name": f"Accessibility audit of {url_fragment(p)}", "startUrl": p, "suite": "Accessibility",
             "steps": [{"operation": "CheckAccessibility"}]} for p in unique[:limit]]


class TestingTreeError(Exception):
    __test__ = False  # not a pytest test class


class TestingTreeClient:
    """TestingTreeServer's REST API (projects + test-case library)."""

    __test__ = False  # not a pytest test class

    def __init__(self, api_url: str, ui_url: str, timeout: float = 15) -> None:
        self.api = api_url.rstrip("/")
        self.ui = ui_url.rstrip("/")
        self.timeout = timeout

    async def _req(self, method: str, path: str, **kw: Any) -> Any:
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as c:
                r = await c.request(method, self.api + path, **kw)
        except httpx.HTTPError as exc:
            raise TestingTreeError(f"TestingTree is not reachable at {self.api} ({exc.__class__.__name__})") from exc
        if r.status_code >= 400:
            try:
                detail = r.json().get("error") or r.text
            except ValueError:
                detail = r.text
            raise TestingTreeError(f"TestingTree {method} {path} -> {r.status_code}: {str(detail)[:300]}")
        return r.json() if r.content else None

    async def project_for(self, target: str, goal: str) -> dict[str, Any]:
        """The Web Tester project for this site (same name), created on first use."""
        host = urlparse(target).netloc
        name = f"Web Tester · {host}"
        for p in await self._req("GET", "/api/projects") or []:
            if p.get("name") == name:
                return p
        return await self._req("POST", "/api/projects", json={
            "name": name, "websiteUrl": target, "testInstructions": goal, "checkAccessibility": True})

    async def test_cases(self, project_id: str) -> list[dict]:
        return await self._req("GET", f"/api/projects/{project_id}/testcase-definitions") or []

    async def add_test_cases(self, project_id: str, cases: list[dict], source: str) -> list[dict]:
        return await self._req("POST", f"/api/projects/{project_id}/testcase-definitions",
                               json={"testCases": cases, "source": source})

    async def run_and_wait(self, project_id: str, ids: list[str], timeout: float) -> dict[str, dict]:
        """Run the given test cases in TestingTree and wait until none is Running. Returns id -> definition."""
        await self._req("POST", f"/api/projects/{project_id}/testcase-definitions/run", json={"ids": ids})
        deadline = time.monotonic() + timeout
        while True:
            await asyncio.sleep(2)
            defs = {d["id"]: d for d in await self._req("GET", f"/api/projects/{project_id}/testcase-definitions")
                    if d["id"] in ids}
            if all(d.get("lastRunStatus") not in (None, "Running") for d in defs.values()) or time.monotonic() > deadline:
                return defs

    def project_url(self, project_id: str) -> str:
        return f"{self.ui}/Projects/Details/{project_id}"
