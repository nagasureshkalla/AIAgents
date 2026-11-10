"""Web Tester agent: tests a website with the agent-browser CLI (Chrome over CDP, accessibility snapshots with @eN refs).

OPEN -> AUDIT (axe-core + page errors) -> EXPLORE (snapshot -> the local model picks one command -> run it) -> REPORT

The agent follows its tool skill, harness/web_tester/skills/agent-browser/SKILL.md: it loads the usage guide that
matches the installed CLI (`agent-browser skills get core|dogfood`) instead of a copy that could go stale.
Safety: the browser may only visit the target's domain (--allowed-domains), the model can only pick from a fixed
set of page commands, and every run uses its own named session that is closed at the end.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from app.agents.base_agent import BaseAgent
from app.agents.testing_tree_export import (
    FlowRecorder,
    TestingTreeClient,
    TestingTreeError,
    accessibility_suite,
    case_key,
    navigation_suite,
    role_selector,
    snapshot_refs,
)
from app.core.config import WebTesterConfig
from app.core.exceptions import AgentSystemError
from app.schemas.agent import AgentTask
from app.utils.text import extract_json, truncate

logger = logging.getLogger(__name__)


class WebTesterError(AgentSystemError):
    code = "web_tester_failed"
    error_type = "web_test_failed"


# Commands the model may run. Everything else (eval, cookies/state, network routes, uploads, close...) is refused.
ALLOWED_COMMANDS = {
    "open", "click", "dblclick", "hover", "focus", "fill", "type", "press", "select", "check", "uncheck",
    "scroll", "wait", "get", "is", "back", "forward", "reload",
}
# Sections of `agent-browser skills get <name>` that matter for a test session (the rest: installs, MCP, cloud...).
SKILL_SECTIONS = {
    "core": ["The core loop", "Interacting", "Waiting (read this)", "Dialogs", "Working safely"],
    "dogfood": ["4. Explore", "5. Document Issues (Repro-First)", "Guidance"],
}
SEVERITIES = ("critical", "high", "medium", "low")
# Run by the agent (never the model) after clicks and key presses: the browser's own validation bubbles are not in
# the accessibility snapshot, so without this the model reports "no validation" on forms the browser blocked.
INVALID_FIELDS_JS = (
    "JSON.stringify([...document.querySelectorAll('input,select,textarea')]"
    ".filter(e => e.willValidate && !e.checkValidity())"
    ".map(e => ((e.labels && e.labels[0] && e.labels[0].innerText) || e.name || e.id) + ': ' + e.validationMessage))"
)
LINK_RE = re.compile(r'link "([^"]*)" \[[^\]]*\burl=([^\],\s]+)')
URL_RE = re.compile(r"https?://[^\s<>\"')\]]+", re.I)
BARE_HOST_RE = re.compile(r"\b((?:localhost|(?:\d{1,3}\.){3}\d{1,3}|(?:[a-z0-9-]+\.)+[a-z]{2,})(?::\d+)?(?:/[^\s<>\"')\]]*)?)",
                          re.I)

PROTOCOL = """You are a meticulous QA engineer testing one website through the agent-browser CLI.
Each turn you see the current page (an accessibility snapshot with @eN refs), the last command's output and your
history. Reply with ONE JSON object and nothing else:
{"note": "<what you are checking now, one short sentence>",
 "scenario": {"suite": "<feature area, e.g. Newsletter>", "name": "<user flow, e.g. Subscribe with a valid email>"},
 "command": "<one agent-browser command, without the 'agent-browser' prefix>",
 "issue": null or {"severity": "critical|high|medium|low", "title": "...", "expected": "...", "actual": "..."},
 "done": false}
Your flows are saved as regression test cases: send "scenario" with the FIRST command that belongs to a new flow
(not while you are still navigating to it; omit it while you continue the same flow), test one behaviour per flow,
and end a flow by checking its result with `get text @eN`.
When you have covered the goal (or the step budget is nearly used), reply {"done": true, "summary": "<findings in 2-4 sentences>"}.

Commands you may use: open <url on the same site>, click @eN, dblclick @eN, hover @eN, focus @eN, fill @eN "text",
type @eN "text", press <Key>, select @eN "value", check @eN, uncheck @eN, scroll down|up [px], wait <ms|@eN>,
wait --text "text", get text|value|title|url|attr @eN [name], is visible|enabled|checked @eN, back, forward, reload.
Use refs from the LATEST snapshot only. The page is re-snapshotted for you after every command.

Report an issue as soon as you have evidence (the agent screenshots it and records your steps as the repro):
broken links or 404/500 pages, forms that accept invalid input or reject valid input, missing validation messages,
buttons that do nothing, wrong or missing confirmation text, JavaScript errors, unlabeled controls, dead ends.
After clicks and key presses you are told whether the URL changed and any browser validation messages
(native bubbles such as "Please include an '@'"): those ARE validation feedback, so a blocked submit is correct.
Use realistic test data (qa@example.com, 'QA Tester'). Never use real credentials unless the request gives them.
Do not buy, pay, delete accounts or send messages to real people. Stay on the target site."""

TESTING_APPROACH = """Test like an exploratory tester: first map the main navigation and pages, then exercise every
form (valid input, empty required fields, an invalid email), every button and link you can reach, and check that
each action shows the result a user would expect. Prefer depth on the user's stated goal; skip what it excludes."""


@dataclass
class CommandResult:
    ok: bool
    output: str


@dataclass
class Finding:
    severity: str
    title: str
    expected: str
    actual: str
    url: str
    steps: list[str]
    screenshot: str | None = None
    source: str = "exploration"  # exploration | console | accessibility


@dataclass
class Run:
    target: str
    goal: str
    steps: list[dict[str, Any]] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    a11y: dict[str, Any] = field(default_factory=dict)
    summary: str = ""
    seen_errors: set[str] = field(default_factory=set)       # JS errors already reported (`errors` never clears)
    links: dict[str, tuple[str, str]] = field(default_factory=dict)  # same-site URL -> (link text, page it was on)
    checked: set[str] = field(default_factory=set)            # link URLs already fetched
    visited: set[str] = field(default_factory=set)
    pages: list[str] = field(default_factory=list)             # visited page URLs, in order
    statuses: dict[str, int | None] = field(default_factory=dict)  # link URL -> HTTP status
    recorder: FlowRecorder | None = None
    testing_tree: dict[str, Any] = field(default_factory=dict)


class AgentBrowser:
    """Runs agent-browser commands in one named, domain-restricted session."""

    def __init__(self, cfg: WebTesterConfig, session: str, allowed_host: str) -> None:
        self.cfg = cfg
        self.session = session
        self.allowed_host = allowed_host

    def available(self) -> bool:
        return shutil.which(self.cfg.cli) is not None

    async def run(self, *args: str, timeout: float | None = None) -> CommandResult:
        argv = [self.cfg.cli, "--session", self.session, "--allowed-domains", self.allowed_host]
        if self.cfg.headed:
            argv.append("--headed")
        env = dict(os.environ)
        if self.cfg.browser_path:
            env["AGENT_BROWSER_EXECUTABLE_PATH"] = self.cfg.browser_path
        proc = await asyncio.create_subprocess_exec(
            *argv, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, env=env)
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), timeout or self.cfg.command_timeout)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            return CommandResult(False, f"timed out after {timeout or self.cfg.command_timeout:.0f}s")
        text = out.decode("utf-8", "replace").strip()
        return CommandResult(proc.returncode == 0 and not text.startswith("✗"), text)

    async def skill(self, name: str) -> str:
        """The installed CLI's own guide (`skills get` does not need a browser session)."""
        proc = await asyncio.create_subprocess_exec(
            self.cfg.cli, "skills", "get", name, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 30)
        return out.decode("utf-8", "replace")


def skill_excerpt(text: str, wanted: list[str]) -> str:
    """Keep only the named `##`/`###`/`####` sections (with their sub-sections) of a skill document."""
    keep, out, level = False, [], 0
    for line in text.splitlines():
        m = re.match(r"^(#{2,4}) (.+)$", line)
        if m:
            depth, title = len(m.group(1)), m.group(2).strip()
            if title in wanted:
                keep, level = True, depth
            elif keep and depth <= level:
                keep = False
        if keep:
            out.append(line)
    return "\n".join(out).strip()


def target_from_prompt(prompt: str) -> str | None:
    m = URL_RE.search(prompt)
    if m:
        return m.group(0).rstrip(".,;:")
    m = BARE_HOST_RE.search(prompt)
    if not m:
        return None
    host = m.group(1).rstrip(".,;:")
    local = host.startswith(("localhost", "127.")) or re.match(r"(?:\d{1,3}\.){3}\d{1,3}", host)
    return ("http://" if local else "https://") + host


def parse_command(command: str, allowed_host: str) -> list[str]:
    """Split a model-chosen command and check it is one of the allowed, on-site page commands."""
    try:
        args = shlex.split(command.strip().removeprefix("agent-browser").strip())
    except ValueError as exc:
        raise ValueError(f"could not parse the command ({exc})") from exc
    if not args:
        raise ValueError("empty command")
    verb = args[0].lower()
    if verb not in ALLOWED_COMMANDS:
        raise ValueError(f"'{verb}' is not allowed; use one of: {', '.join(sorted(ALLOWED_COMMANDS))}")
    if any(a.startswith("--") and a not in ("--text", "--url", "--load") for a in args[1:]):
        raise ValueError("global flags are not allowed in commands")
    if verb == "open" and (len(args) != 2 or urlparse(args[1]).hostname != allowed_host):
        raise ValueError(f"open only accepts a full URL on {allowed_host}")
    return [verb, *args[1:]]


class WebTesterAgent(BaseAgent):
    name = "web_tester"
    description = (
        "Web Tester: tests websites in a real Chrome via agent-browser: explores pages, fills forms, clicks every "
        "button and link, runs accessibility and console checks, and writes a QA report with repro steps and screenshots."
    )
    capabilities = ["web_testing", "qa_testing", "browser_automation"]
    default_model = "control_panel"
    prompts = {"explore": TESTING_APPROACH}

    browser_factory: Callable[[WebTesterConfig, str, str], AgentBrowser] = AgentBrowser
    testing_tree_factory: Callable[[str, str], TestingTreeClient] = TestingTreeClient
    _skill_cache: dict[str, str] = {}

    @property
    def wcfg(self) -> WebTesterConfig:
        return self.ctx.settings.web_tester

    # ------------------------------------------------------------------ main loop
    async def run(self, task: AgentTask) -> dict[str, Any]:
        cfg = self.wcfg
        if not cfg.enabled:
            raise WebTesterError("The Web Tester agent is disabled (web_tester.enabled: false)")
        target = task.params.get("url") or target_from_prompt(task.prompt)
        if not target:
            raise WebTesterError("Tell me which website to test, e.g. 'Test the signup form on http://localhost:3000'")
        host = urlparse(target).hostname or ""
        session = f"webtest-{task.task_id[-12:]}"
        browser = self.browser_factory(cfg, session, host)
        if not browser.available():
            raise WebTesterError(f"'{cfg.cli}' is not installed: npm i -g agent-browser (then agent-browser install, "
                                 "or set web_tester.browser_path to an installed Chrome)")
        out_dir = self.ctx.settings.resolve_path(cfg.output_dir) / task.task_id
        (out_dir / "screenshots").mkdir(parents=True, exist_ok=True)
        run = Run(target=target, goal=task.prompt.strip())
        run.recorder = FlowRecorder(target)

        self.progress(task, f"Opening {target}", url=target)
        opened = await browser.run("open", target)
        if not opened.ok:
            raise WebTesterError(f"Could not open {target}: {truncate(opened.output, 300)}")
        try:
            await self._audit(task, browser, run, out_dir)
            system = await self._system_prompt(browser)
            await self._explore(task, browser, run, system, out_dir)
        finally:
            await browser.run("close", timeout=20)

        if cfg.export_to_testing_tree:
            await self._export(task, run)
        report = self._report(run, out_dir)
        (out_dir / "report.md").write_text(report, encoding="utf-8")
        counts = {s: sum(f.severity == s for f in run.findings) for s in SEVERITIES}
        self.progress(task, f"Report: {len(run.findings)} issue(s) in {len(run.steps)} steps", report=str(out_dir / "report.md"))
        return {
            "text": report,
            "target": target,
            "issues": [f.__dict__ for f in run.findings],
            "issue_counts": counts,
            "steps": run.steps,
            "accessibility": run.a11y,
            "testing_tree": run.testing_tree,
            "report_path": str(out_dir / "report.md"),
        }

    async def _system_prompt(self, browser: AgentBrowser) -> str:
        """Fixed command protocol + the tool skill (its CLI-served guides) + the harness' testing approach."""
        parts = [PROTOCOL]
        harness = self.ctx.harness.get(self.name) if self.ctx.harness else None
        tool_skill = harness.skills.get("agent-browser") if harness else None
        if tool_skill is not None:  # the skill says: load the real workflow content from the CLI
            for name in self.wcfg.skills:
                if name not in WebTesterAgent._skill_cache:
                    try:
                        WebTesterAgent._skill_cache[name] = skill_excerpt(await browser.skill(name), SKILL_SECTIONS.get(name, []))
                    except (OSError, TimeoutError) as exc:
                        logger.info("agent-browser skills get %s failed: %s", name, exc)
                        WebTesterAgent._skill_cache[name] = ""
                if WebTesterAgent._skill_cache[name]:
                    parts.append(f"agent-browser guide ({name}), for reference; you only send the command part:\n"
                                 + WebTesterAgent._skill_cache[name])
        parts.append(self.system_prompt("explore"))
        return "\n\n".join(parts)

    async def _audit(self, task: AgentTask, browser: AgentBrowser, run: Run, out_dir: Path) -> None:
        """Deterministic checks on the landing page: axe-core accessibility and JavaScript errors."""
        self.progress(task, "Running the accessibility audit (axe-core)")
        res = await browser.run("a11y", "--json", timeout=90)
        data = extract_json(res.output) if res.ok else None
        if isinstance(data, dict):
            data = data.get("data", data)
            violations = data.get("violations") or []
            run.a11y = {"counts": data.get("counts", {}), "violations": [
                {"id": v.get("id"), "impact": v.get("impact"), "help": v.get("help"), "nodes": len(v.get("nodes") or [])}
                for v in violations]}
            for v in violations:
                if v.get("impact") in ("critical", "serious"):
                    run.findings.append(Finding(
                        severity="high" if v.get("impact") == "critical" else "medium",
                        title=f"Accessibility: {v.get('help') or v.get('id')}",
                        expected="No WCAG violation", actual=f"axe rule '{v.get('id')}' fails on {len(v.get('nodes') or [])} element(s)",
                        url=run.target, steps=[f"open {run.target}", "a11y"], source="accessibility"))
        else:
            run.a11y = {"error": truncate(res.output, 200)}
        await self._page_errors(browser, run, [f"open {run.target}"], out_dir)

    async def _page_errors(self, browser: AgentBrowser, run: Run, steps: list[str], out_dir: Path) -> list[str]:
        """Record JavaScript errors not seen before. Plain `errors` prints only '✗' when there are some, and
        `--clear` does not clear, so the JSON list is read and de-duplicated here."""
        res = await browser.run("errors", "--json")
        data = extract_json(res.output)
        errors = (data.get("data") or {}).get("errors") or [] if isinstance(data, dict) else []
        new = [t for t in dict.fromkeys(str(e.get("text", "")).strip() for e in errors) if t and t not in run.seen_errors]
        if new:
            run.seen_errors.update(new)
            shot = await self._screenshot(browser, out_dir, len(run.findings) + 1)
            url = await self._url(browser)
            for text in new:
                run.findings.append(Finding(
                    severity="high", title=f"JavaScript error: {truncate(text.splitlines()[0], 90)}",
                    expected="No page errors", actual=truncate(text, 600), url=url, steps=list(steps),
                    screenshot=shot, source="console"))
        return new

    async def _check_links(self, task: AgentTask, run: Run, snapshot: str, page_url: str) -> list[str]:
        """Fetch every same-site link seen in a snapshot once; 4xx/5xx answers are broken links."""
        for text, url in LINK_RE.findall(snapshot):
            url = url.split("#")[0]
            if urlparse(url).hostname == urlparse(run.target).hostname and url not in run.links:
                run.links[url] = (text, page_url)
        broken = []
        for url in [u for u in run.links if u not in run.checked][:20]:
            run.checked.add(url)
            status = run.statuses[url] = await self.link_status(url)
            if status is None or status >= 400:
                text, page = run.links[url]
                shown = status or "no response"
                broken.append(f"{text} -> {url} ({shown})")
                run.findings.append(Finding(
                    severity="high" if status is None or status >= 500 or status == 404 else "medium",
                    title=f"Broken link: '{text}' returns {shown}", expected="The page loads (2xx/3xx)",
                    actual=f"GET {url} -> {shown}", url=page, steps=[f"open {page}", f"click link '{text}'"],
                    source="links"))
                self.progress(task, f"⚠ broken link: {text} → {shown}")
        return broken

    @staticmethod
    async def link_status(url: str) -> int | None:
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
                return (await c.get(url)).status_code
        except httpx.HTTPError:
            return None

    async def _explore(self, task: AgentTask, browser: AgentBrowser, run: Run, system: str, out_dir: Path) -> None:
        history: list[str] = [f"open {run.target}"]
        last = "opened the start page"
        failures = 0
        for step in range(1, self.wcfg.max_steps + 1):
            snap = await browser.run("snapshot", "-i", "-c", "-u")
            url = await self._url(browser)
            if url.split("#")[0].rstrip("/") not in run.visited:
                run.pages.append(url.split("#")[0])
            run.visited.add(url.split("#")[0].rstrip("/"))
            broken = await self._check_links(task, run, snap.output, url)
            if broken:
                last += "\nBroken links found (already reported, no need to click them): " + "; ".join(broken)
            unvisited = [u for u in run.links if u.rstrip("/") not in run.visited]
            user = (
                f"Target: {run.target}\nGoal: {run.goal}\nStep {step} of {self.wcfg.max_steps}\n"
                f"Current URL: {url}\nLast command result: {truncate(last, 800)}\n"
                f"History: {' | '.join(history[-10:])}\n"
                f"Same-site pages not visited yet: {', '.join(unvisited[:10]) or 'none'}\n"
                f"Issues reported so far: {'; '.join(f.title for f in run.findings) or 'none'}\n\n"
                f"Page snapshot:\n{truncate(snap.output, 3500)}"
            )
            out = await self.generate(task, user, system=system, max_tokens=400, temperature=0.0)
            reply = extract_json(out.text)
            if not isinstance(reply, dict):
                failures += 1
                last = "Your reply was not a JSON object. Reply with exactly one JSON object."
                if failures >= 3:
                    break
                continue
            scenario = reply.get("scenario")
            if isinstance(scenario, dict) and run.recorder is not None:
                run.recorder.begin(scenario.get("suite"), scenario.get("name"), url)
            if reply.get("issue"):
                await self._record_issue(browser, run, reply["issue"], history, url, out_dir)
                if run.recorder is not None:
                    run.recorder.note_issue(run.findings[-1].title)
                self.progress(task, f"⚠ {run.findings[-1].severity}: {run.findings[-1].title}")
            if reply.get("done"):
                run.summary = str(reply.get("summary") or "")
                break
            command = str(reply.get("command") or "")
            try:
                args = parse_command(command, browser.allowed_host)
                if history[-2:] == [command, command]:
                    raise ValueError("you already ran this command twice in a row and its result will not change; "
                                     "test something you have not covered yet")
            except ValueError as exc:
                failures += 1
                last = f"Command refused: {exc}"
                if failures >= 3:
                    break
                continue
            self.progress(task, f"▶ {truncate(command, 120)}", note=truncate(str(reply.get("note") or ""), 160))
            started = time.perf_counter()
            selector, label = await self._replayable(browser, args, snapshot_refs(snap.output))
            res = await browser.run(*args)
            feedback = await self._after_action(browser, args[0], url) if res.ok else ""
            if res.ok and run.recorder is not None:
                run.recorder.record(args, selector, label, url, await self._url(browser), res.output)
            history.append(command)
            run.steps.append({"step": step, "note": reply.get("note"), "command": command, "ok": res.ok,
                              "output": truncate(res.output, 300), "ms": round((time.perf_counter() - started) * 1000)})
            failures = 0 if res.ok else failures + 1
            last = (res.output or ("ok" if res.ok else "failed")) + feedback
            if failures >= 3:
                break
            new_errors = await self._page_errors(browser, run, history, out_dir)
            if new_errors:
                last += "\nJavaScript error after this command (already reported): " + truncate(new_errors[0], 200)

    @staticmethod
    async def _replayable(browser: AgentBrowser, args: list[str], refs: dict[str, tuple[str, str]]
                          ) -> tuple[str | None, str | None]:
        """A selector TestingTree can replay for the command's @ref (refs only exist inside this session)."""
        ref = next((a[1:] for a in args[1:] if a.startswith("@e")), None)
        if ref is None:
            return None, None
        role, name = refs.get(ref, ("", ""))
        if role and name:
            return role_selector(role, name), name
        for attr, fmt in (("id", "#{}"), ("name", '[name="{}"]')):
            res = await browser.run("get", "attr", f"@{ref}", attr)
            value = res.output.strip().splitlines()[-1].strip() if res.ok and res.output.strip() else ""
            if value and value not in ("null", "undefined") and not value.startswith("✗"):
                return fmt.format(value), name or None
        return None, None

    async def _export(self, task: AgentTask, run: Run) -> None:
        """Save the tested flows (plus navigation and accessibility suites) to TestingTree and replay them there."""
        cfg = self.wcfg
        assert run.recorder is not None
        cases = (run.recorder.test_cases() + navigation_suite(run.links, run.statuses)
                 + accessibility_suite(run.pages or [run.target]))
        if not cases:
            run.testing_tree = {"error": "no replayable flows were recorded"}
            return
        client = self.testing_tree_factory(cfg.testing_tree_api, cfg.testing_tree_ui)
        try:
            self.progress(task, f"Saving {len(cases)} test case(s) to TestingTree")
            project = await client.project_for(run.target, run.goal)
            existing = {case_key(d): d for d in await client.test_cases(project["id"])}
            new = [c for c in cases if case_key(c) not in existing]
            kept = [existing[case_key(c)] for c in cases if case_key(c) in existing]  # saved by an earlier run
            created = await client.add_test_cases(project["id"], new, source=f"web-tester:{task.task_id}") if new else []
            run.testing_tree = {"project_id": project["id"], "project_url": client.project_url(project["id"]),
                                "added": len(created), "already_saved": len(kept),
                                "test_cases": [{"id": d["id"], "suite": d.get("suite"), "name": d["name"],
                                                "steps": len(d.get("steps") or []), "new": d in created}
                                               for d in created + kept]}
            if cfg.run_exported_tests:
                self.progress(task, f"Running {len(created)} test case(s) in TestingTree")
                results = await client.run_and_wait(project["id"], [d["id"] for d in created + kept], cfg.run_timeout)
                for tc in run.testing_tree["test_cases"]:
                    d = results.get(tc["id"], {})
                    tc["status"], tc["error"] = d.get("lastRunStatus"), d.get("lastRunError")
        except TestingTreeError as exc:
            logger.info("TestingTree export failed: %s", exc)
            run.testing_tree = {**run.testing_tree, "error": str(exc)}

    async def _after_action(self, browser: AgentBrowser, verb: str, before_url: str) -> str:
        """What a user would notice after a click or key press: navigation and the browser's validation messages."""
        if verb not in ("click", "dblclick", "press"):
            return ""
        url = await self._url(browser)
        notes = [f"URL changed to {url}" if url and url != before_url else "URL unchanged"]
        res = await browser.run("eval", INVALID_FIELDS_JS)
        invalid = extract_json(res.output) if res.ok else None
        if isinstance(invalid, str):  # eval prints the JSON.stringify result as a JSON string
            invalid = extract_json(invalid)
        if isinstance(invalid, list) and invalid:
            notes.append("browser validation messages: " + "; ".join(map(str, invalid[:5])))
        return "\n" + "; ".join(notes)

    async def _record_issue(self, browser: AgentBrowser, run: Run, issue: Any, history: list[str], url: str,
                            out_dir: Path) -> None:
        issue = issue if isinstance(issue, dict) else {"title": str(issue)}
        severity = str(issue.get("severity", "medium")).lower()
        run.findings.append(Finding(
            severity=severity if severity in SEVERITIES else "medium",
            title=truncate(str(issue.get("title") or "Untitled issue"), 140),
            expected=str(issue.get("expected") or ""), actual=str(issue.get("actual") or ""),
            url=url, steps=list(history), screenshot=await self._screenshot(browser, out_dir, len(run.findings) + 1)))

    @staticmethod
    async def _screenshot(browser: AgentBrowser, out_dir: Path, n: int) -> str | None:
        path = out_dir / "screenshots" / f"issue-{n}.png"
        res = await browser.run("screenshot", str(path))
        return str(path) if res.ok and path.exists() else None

    @staticmethod
    async def _url(browser: AgentBrowser) -> str:
        res = await browser.run("get", "url")
        return res.output.strip().splitlines()[-1] if res.ok and res.output.strip() else ""

    # ------------------------------------------------------------------ report
    @staticmethod
    def _report(run: Run, out_dir: Path) -> str:
        order = {s: i for i, s in enumerate(SEVERITIES)}
        findings = sorted(run.findings, key=lambda f: order.get(f.severity, 9))
        counts = ", ".join(f"{sum(f.severity == s for f in findings)} {s}" for s in SEVERITIES)
        lines = [f"# Website test report: {run.target}", "",
                 f"**Goal:** {run.goal}", f"**Result:** {len(findings)} issue(s) ({counts}) after {len(run.steps)} browser step(s).",
                 f"**Evidence:** `{out_dir}`", ""]
        if run.summary:
            lines += ["## Summary", run.summary, ""]
        lines.append("## Issues")
        if not findings:
            lines.append("No issues found in the explored scope.")
        for i, f in enumerate(findings, 1):
            lines += ["", f"### {i}. [{f.severity}] {f.title}", f"- **Where:** {f.url}  ·  **Found by:** {f.source}"]
            if f.expected:
                lines.append(f"- **Expected:** {f.expected}")
            if f.actual:
                lines.append(f"- **Actual:** {f.actual}")
            lines.append("- **Steps to reproduce:** " + " → ".join(f"`{s}`" for s in f.steps[-8:]))
            if f.screenshot:
                lines.append(f"- **Screenshot:** `{f.screenshot}`")
        tt = run.testing_tree
        if tt.get("test_cases"):
            ran = [t for t in tt["test_cases"] if t.get("status")]
            passed = sum(t.get("status") == "Passed" for t in ran)
            lines += ["", "## TestingTree test cases",
                      f"Saved {tt.get('added', 0)} new test case(s) to [the TestingTree project]({tt['project_url']})"
                      + (f" ({tt['already_saved']} already there from earlier runs)" if tt.get("already_saved") else "")
                      + (f"; replayed there: {passed}/{len(ran)} passed." if ran else "."),
                      "", "| Suite | Test case | Steps | Result |", "|---|---|---|---|"]
            lines += [f"| {t['suite']} | {t['name'].replace('|', '/')} | {t['steps']} | "
                      f"{t.get('status') or 'not run'}{(': ' + str(t['error'])[:80].replace('|', '/')) if t.get('error') else ''} |"
                      for t in tt["test_cases"]]
        if tt.get("error"):
            lines += ["", "## TestingTree test cases", f"Not saved: {tt['error']}"]
        a11y = run.a11y.get("violations")
        if a11y is not None:
            lines += ["", "## Accessibility (axe-core, start page)",
                      "No violations." if not a11y else "\n".join(f"- `{v['id']}` ({v['impact']}): {v['help']}, {v['nodes']} element(s)" for v in a11y)]
        if run.steps:
            lines += ["", "## Steps taken", "| # | Command | Result |", "|---|---|---|"]
            lines += [f"| {s['step']} | `{s['command']}` | {'ok' if s['ok'] else 'failed: ' + s['output'].replace('|', '/')[:80]} |"
                      for s in run.steps]
        return "\n".join(lines)
