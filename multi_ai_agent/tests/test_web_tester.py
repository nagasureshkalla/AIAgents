"""Web Tester agent: routing, command safety, skill loading and a full scripted test run (fake browser + model)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from app.agents import web_tester_agent as wt
from app.agents.manager_agent import ManagerAgent
from app.agents.planner_agent import PLAN_TEMPLATES
from app.agents.testing_tree_export import TestingTreeClient, TestingTreeError
from app.agents.web_tester_agent import (
    AgentBrowser,
    CommandResult,
    WebTesterAgent,
    parse_command,
    skill_excerpt,
    target_from_prompt,
)
from app.core.config import Settings
from app.models.base import GenerationOutput
from app.schemas.common import TaskStatus
from app.schemas.task import TaskRequest
from tests.fakes import FakeFactory, build_container, make_settings


# ---------------------------------------------------------------- routing
@pytest.mark.parametrize("prompt", [
    "Test the website http://localhost:3000",
    "QA https://example.com signup flow",
    "dogfood vercel.com",
    "find bugs on my site localhost:5000",
    "check the forms on http://127.0.0.1:8931",
    "smoke test the landing page",
    "Can you test the signup flow on the staging site?",
])
def test_website_testing_requests_go_to_the_web_tester(prompt):
    assert ManagerAgent.classify_intent(prompt, False) == "web_test"


@pytest.mark.parametrize(("prompt", "intent"), [
    ("Open Calculator and calculate 25 * 89", "computer_control"),
    ("what is the latest Python release", "research"),
    ("Check the price of an iPhone on amazon.com", "research"),
    ("https://example.com what is this company", "research"),
    ("Summarize the test results on page 3", "summarize"),
    ("Explain how to test a web app", "explain"),
    ("Write a test plan for my app", "generate"),
    ("Generate unit tests for this page component", "generate"),
])
def test_other_requests_keep_their_routes(prompt, intent):
    assert ManagerAgent.classify_intent(prompt, False) == intent


def test_web_test_plan_is_one_web_testing_step():
    (step,) = PLAN_TEMPLATES["web_test"]
    assert step.capability == "web_testing" and step.mode == "explore"


# ---------------------------------------------------------------- helpers
@pytest.mark.parametrize(("prompt", "url"), [
    ("Test http://localhost:3000/signup, please.", "http://localhost:3000/signup"),
    ("dogfood vercel.com", "https://vercel.com"),
    ("find bugs on localhost:5000", "http://localhost:5000"),
    ("QA 127.0.0.1:8931/app", "http://127.0.0.1:8931/app"),
    ("smoke test the landing page", None),
])
def test_target_from_prompt(prompt, url):
    assert target_from_prompt(prompt) == url


def test_parse_command_allows_page_commands_only():
    assert parse_command('fill @e3 "qa@example.com"', "localhost") == ["fill", "@e3", "qa@example.com"]
    assert parse_command("agent-browser click @e2", "localhost") == ["click", "@e2"]
    assert parse_command('wait --text "Thanks"', "localhost") == ["wait", "--text", "Thanks"]
    assert parse_command("open http://localhost/pricing", "localhost") == ["open", "http://localhost/pricing"]
    for bad in ["eval 'document.cookie'", "cookies get", "close", "open https://evil.example/",
                "click @e1 --allowed-domains evil.example", "state save /tmp/x.json", ""]:
        with pytest.raises(ValueError):
            parse_command(bad, "localhost")


def test_skill_excerpt_keeps_the_named_sections_with_their_subsections():
    doc = "# t\n## Intro\nskip\n## Interacting\nclick\n### Sub\nkept\n## Install\nskip too\n## Waiting (read this)\nwait\n"
    out = skill_excerpt(doc, ["Interacting", "Waiting (read this)"])
    assert "click" in out and "kept" in out and "wait" in out
    assert "skip" not in out


# ---------------------------------------------------------------- full run with a fake browser and a scripted model
class FakeBrowser(AgentBrowser):
    """'/' has a newsletter form, a Promo button that throws, and links to /pricing (ok) and /docs (404);
    the form leads to '/thanks'. Like the real CLI, `errors --json` keeps returning every error (never cleared)."""

    instances: list[FakeBrowser] = []

    def __init__(self, cfg, session, allowed_host) -> None:
        super().__init__(cfg, session, allowed_host)
        self.commands: list[list[str]] = []
        self.url = ""
        self.page_errors: list[str] = []
        FakeBrowser.instances.append(self)

    def available(self) -> bool:
        return True

    async def skill(self, name: str) -> str:
        return f"## The core loop\nsnapshot then act ({name})\n## Install\nnot needed\n## 4. Explore\nexplore {name}\n"

    async def run(self, *args: str, timeout: float | None = None) -> CommandResult:
        self.commands.append(list(args))
        verb = args[0]
        if verb == "open":
            self.url = args[1]
            return CommandResult(True, f"✓ Acme\n  {self.url}")
        if verb == "snapshot":
            if self.url.endswith("/thanks"):
                return CommandResult(True, '- heading "Thanks!" [ref=e1]')
            return CommandResult(True, '- link "Pricing" [ref=e1, url=http://localhost:3000/pricing]\n'
                                       '- link "Docs" [ref=e2, url=http://localhost:3000/docs]\n'
                                       '- link "Elsewhere" [ref=e5, url=https://other.example/]\n'
                                       '- textbox "Email" [required, ref=e3]\n- button "Subscribe" [ref=e4]\n'
                                       '- button "Promo" [ref=e9]')
        if verb == "get" and args[1] == "url":
            return CommandResult(True, self.url)
        if verb == "a11y":
            return CommandResult(True, json.dumps({"success": True, "data": {"counts": {"violations": 2}, "violations": [
                {"id": "image-alt", "impact": "critical", "help": "Images must have alternate text", "nodes": [{}]},
                {"id": "region", "impact": "moderate", "help": "Content should be in landmarks", "nodes": [{}, {}]}]}}))
        if verb == "errors":
            return CommandResult(True, json.dumps({"success": True, "data": {"errors": [{"text": e} for e in self.page_errors]}}))
        if verb == "fill":
            return CommandResult(True, "✓ Done")
        if verb == "click" and args[1] == "@e4":
            self.url = self.url.rstrip("/") + "/thanks"
            return CommandResult(True, "✓ Done")
        if verb == "click" and args[1] == "@e9":
            self.page_errors.append("TypeError: promo is not defined")
            return CommandResult(True, "✓ Done")
        if verb == "screenshot":
            Path(args[1]).write_bytes(b"png")
            return CommandResult(True, f"✓ Screenshot saved to {args[1]}")
        if verb == "close":
            return CommandResult(True, "✓ Browser closed")
        return CommandResult(False, f"✗ Unknown command {verb}")


class FakeTestingTree(TestingTreeClient):
    """Records what the agent uploads; every replayed test passes except ones whose name mentions an issue."""

    instances: list[FakeTestingTree] = []
    unreachable = False
    saved: list[dict] = []  # test cases already in the project (from earlier runs)

    def __init__(self, api_url: str, ui_url: str) -> None:
        super().__init__(api_url, ui_url)
        self.uploaded: list[dict] = []
        self.source = ""
        FakeTestingTree.instances.append(self)

    async def test_cases(self, project_id: str) -> list[dict]:
        return list(FakeTestingTree.saved)

    async def project_for(self, target: str, goal: str) -> dict:
        if FakeTestingTree.unreachable:
            raise TestingTreeError("TestingTree is not reachable at http://127.0.0.1:5001 (ConnectError)")
        return {"id": "p1", "name": "Web Tester · localhost:3000"}

    async def add_test_cases(self, project_id: str, cases: list[dict], source: str) -> list[dict]:
        self.uploaded, self.source = cases, source
        created = [{**c, "id": f"tc{len(FakeTestingTree.saved) + i}"} for i, c in enumerate(cases)]
        FakeTestingTree.saved.extend(created)
        return created

    async def run_and_wait(self, project_id: str, ids: list[str], timeout: float) -> dict:
        return {c["id"]: {"lastRunStatus": "Failed" if "issue:" in c["name"] else "Passed",
                          "lastRunError": "Step 3 (AssertTextContains): expected email" if "issue:" in c["name"] else None}
                for c in FakeTestingTree.saved if c["id"] in ids}


SCRIPT = [
    {"note": "Try the promo button", "command": "click @e9", "issue": None, "done": False,
     "scenario": {"suite": "Promotions", "name": "Show the promo"}},
    {"note": "Again", "command": "click @e9", "done": False},
    {"note": "And again (refused: same command twice in a row already)", "command": "click @e9", "done": False},
    {"note": "Submit with an email", "command": 'fill @e3 "qa@example.com"', "issue": None, "done": False,
     "scenario": {"suite": "Newsletter", "name": "Subscribe with a valid email"}},
    {"note": "eval is not allowed", "command": "eval 'alert(1)'", "done": False},
    {"note": "Subscribe", "command": "click @e4", "done": False},
    {"note": "Confirmation lacks the email", "command": "get title",
     "issue": {"severity": "low", "title": "Thank-you page does not repeat the email",
               "expected": "Shows qa@example.com", "actual": "Only 'Thanks!'"}, "done": False},
    {"done": True, "summary": "Newsletter works; promo button throws; confirmation is terse."},
]


@pytest.fixture
def web_settings(tmp_path) -> Settings:
    data = make_settings(tmp_path).model_dump()
    data["agents"]["web_tester"] = {"model": "fast_text"}
    data["web_tester"] = {"output_dir": str(tmp_path / "web_tests"), "max_steps": 10, "skills": ["core", "dogfood"]}
    return Settings.model_validate(data)


@pytest.fixture
async def web_container(web_settings, monkeypatch):
    FakeBrowser.instances = []
    WebTesterAgent._skill_cache = {}
    replies = iter(SCRIPT)
    prompts: list[tuple[str, str]] = []

    async def scripted(self, task, user, *, system=None, **kw: Any) -> GenerationOutput:
        prompts.append((system or "", user))
        return GenerationOutput(text=json.dumps(next(replies)), model="fast_text")

    async def link_status(url: str) -> int | None:
        fetched.append(url)
        return 404 if url.endswith("/docs") else 200

    fetched: list[str] = []
    monkeypatch.setattr(WebTesterAgent, "link_status", staticmethod(link_status))
    monkeypatch.setattr(WebTesterAgent, "browser_factory", FakeBrowser)
    FakeTestingTree.instances, FakeTestingTree.saved = [], []
    monkeypatch.setattr(WebTesterAgent, "testing_tree_factory", FakeTestingTree)
    monkeypatch.setattr(WebTesterAgent, "generate", scripted)
    c = build_container(web_settings, factory=FakeFactory())
    c.prompts = prompts
    c.fetched = fetched
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


async def test_full_run_reports_issues_with_evidence(web_container, web_settings):
    record = await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    assert record.status == TaskStatus.COMPLETED, record.error
    result = record.agent_results["webtest"].result
    browser = FakeBrowser.instances[-1]

    # every browser call ran in one named session restricted to the target host, which is closed at the end
    assert browser.session.startswith("webtest-") and browser.allowed_host == "localhost"
    assert browser.commands[0] == ["open", "http://localhost:3000/"] and browser.commands[-1] == ["close"]
    assert ["eval", "'alert(1)'"] not in browser.commands  # refused before it reached the browser

    titles = [i["title"] for i in result["issues"]]
    assert any(t.startswith("Accessibility: Images must have alternate text") for t in titles)  # critical axe rule
    assert not any("landmarks" in t for t in titles)  # moderate axe rules stay in the a11y section only
    js = [i for i in result["issues"] if i["source"] == "console"]
    assert len(js) == 1  # reported once although the CLI keeps listing it after every later command
    assert "promo is not defined" in js[0]["actual"] and js[0]["steps"][-1] == "click @e9" and js[0]["screenshot"]
    assert browser.commands.count(["click", "@e9"]) == 2  # the third identical click was refused
    assert any("JavaScript error after this command" in user for _, user in web_container.prompts)

    # links: every same-site link fetched once, the 404 reported with where it was found; other sites never fetched
    assert sorted(web_container.fetched) == ["http://localhost:3000/docs", "http://localhost:3000/pricing"]
    broken = [i for i in result["issues"] if i["source"] == "links"]
    assert len(broken) == 1 and broken[0]["title"] == "Broken link: 'Docs' returns 404"
    assert broken[0]["url"] == "http://localhost:3000/" and broken[0]["severity"] == "high"
    assert "Same-site pages not visited yet: http://localhost:3000/pricing" in web_container.prompts[0][1]
    low = next(i for i in result["issues"] if i["severity"] == "low")
    assert low["url"].endswith("/thanks") and 'fill @e3 "qa@example.com"' in low["steps"]

    report = Path(result["report_path"]).read_text()
    assert "# Website test report: http://localhost:3000/" in report and "promo is not defined" in report
    assert "Newsletter works" in report and "`image-alt` (critical)" in report
    assert result["issue_counts"]["high"] == 3 and result["issue_counts"]["low"] == 1


async def test_tool_skill_loads_the_cli_guides(web_container, web_settings):
    skill_dir = Path(web_settings.harness.root) / "web_tester" / "skills" / "agent-browser"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("---\nname: agent-browser\ndescription: browser CLI\n---\nRun skills get core.\n")
    await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    system = web_container.prompts[0][0]
    assert "snapshot then act (core)" in system and "explore dogfood" in system
    assert "not needed" not in system  # install sections are left out


async def test_without_a_target_it_asks_for_one(web_container):
    record = await web_container.run_and_wait(
        TaskRequest(prompt="smoke test the landing page", params={}), timeout=20)
    assert record.status == TaskStatus.FAILED and "which website" in (record.error or "")
    assert record.agent_results["webtest"].attempts == 1  # not retried


async def test_missing_cli_is_reported(web_container, monkeypatch):
    monkeypatch.setattr(FakeBrowser, "available", lambda self: False)
    record = await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    assert record.status == TaskStatus.FAILED and "npm i -g agent-browser" in (record.error or "")


def test_agent_is_registered_with_its_capability(web_settings):
    c = build_container(web_settings, factory=FakeFactory())
    agent = c.agents.first_for("web_testing")
    assert agent is not None and agent.name == "web_tester" and wt.ALLOWED_COMMANDS.isdisjoint({"eval", "close"})


async def test_blocked_submit_reports_the_browsers_validation_message(web_settings):
    class Blocked(FakeBrowser):
        async def run(self, *args: str, timeout: float | None = None) -> CommandResult:
            if args[0] == "eval":
                return CommandResult(True, json.dumps(json.dumps(["Email: Please include an '@' in the email address."])))
            if args[0] == "get":
                return CommandResult(True, "http://localhost:3000/")
            return CommandResult(True, "✓ Done")

    agent = WebTesterAgent(build_container(web_settings, factory=FakeFactory()).agents.get("web_tester").ctx)
    browser = Blocked(web_settings.web_tester, "s", "localhost")
    note = await agent._after_action(browser, "click", "http://localhost:3000/")
    assert "URL unchanged" in note and "Please include an '@'" in note
    assert await agent._after_action(browser, "fill", "http://localhost:3000/") == ""  # only clicks / key presses
# ---------------------------------------------------------------- TestingTree export
from app.agents.testing_tree_export import (
    FlowRecorder,
    accessibility_suite,
    navigation_suite,
    role_selector,
    snapshot_refs,
)


def test_snapshot_refs_and_role_selectors():
    snap = ('- textbox "Email" [required, ref=e3]\n- button "Say \\"hi\\"" [ref=e4]\n'
            '- link "Docs" [ref=e2, url=http://x/docs]\n- generic [ref=e7]')
    assert snapshot_refs(snap) == {"e3": ("textbox", "Email"), "e4": ("button", 'Say "hi"'), "e2": ("link", "Docs"),
                                   "e7": ("generic", "")}
    assert role_selector("button", 'Say "hi"') == 'role=button[name="Say \\"hi\\""s]'


def test_recorder_turns_commands_into_named_test_cases_with_asserts():
    rec = FlowRecorder("http://s/")
    rec.begin("Newsletter", "Subscribe", "http://s/")
    rec.record(["fill", "@e3", "qa@example.com"], 'role=textbox[name="Email"s]', "Email", "http://s/", "http://s/", "✓ Done")
    rec.record(["click", "@e4"], 'role=button[name="Subscribe"s]', "Subscribe", "http://s/", "http://s/thanks", "✓ Done")
    rec.record(["get", "text", "@e1"], 'role=heading[name="Thanks!"s]', "Thanks!", "http://s/thanks", "http://s/thanks", "Thanks!")
    rec.record(["scroll", "down"], None, None, "http://s/thanks", "http://s/thanks", "✓ Done")  # not replayable: skipped
    rec.begin("Newsletter", "Subscribe", "http://s/thanks")  # same flow named again: continues it
    rec.begin("Pricing", "Choose Pro", "http://s/pricing")
    rec.note_issue("Plan not shown")
    rec.record(["click", "@e9"], 'role=button[name="Choose Pro"s]', "Choose Pro", "http://s/pricing", "http://s/pricing", "✓")
    rec.begin("Empty", "Only waits", "http://s/")
    rec.record(["wait", "500"], None, None, "http://s/", "http://s/", "✓")

    first, second = rec.test_cases()  # the wait-only flow is dropped
    assert (first["suite"], first["name"], first["startUrl"]) == ("Newsletter", "Subscribe", "http://s/")
    assert [s["operation"] for s in first["steps"]] == ["Type", "Click", "AssertUrlContains", "AssertTextContains"]
    assert first["steps"][0]["value"] == "qa@example.com" and first["steps"][2]["value"] == "/thanks"
    assert first["steps"][3]["value"] == "Thanks!"
    assert second["name"] == "Choose Pro (issue: Plan not shown)" and second["startUrl"] == "http://s/pricing"


def test_unnamed_actions_become_one_exploration_flow_per_page():
    rec = FlowRecorder("http://s/")
    rec.record(["click", "@e1"], 'role=button[name="Show promo"s]', "Show promo", "http://s/", "http://s/", "✓")
    (case,) = rec.test_cases()
    assert case["suite"] == "Exploration" and case["name"] == "Explore http://s/"


def test_navigation_suite_skips_broken_and_self_links():
    links = {"http://s/pricing": ("Pricing", "http://s/"), "http://s/docs": ("Docs", "http://s/"),
             "http://s/": ("Home", "http://s/")}
    (case,) = navigation_suite(links, {"http://s/pricing": 200, "http://s/docs": 404, "http://s/": 200})
    assert case["suite"] == "Navigation" and case["steps"][0]["selector"] == 'role=link[name="Pricing"s]'
    assert case["steps"][1] == {"operation": "AssertUrlContains", "value": "/pricing"}


async def test_full_run_saves_suites_to_testing_tree_and_reports_replay_results(web_container):
    record = await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    result = record.agent_results["webtest"].result
    tt = FakeTestingTree.instances[-1]
    assert tt.source.startswith("web-tester:task_")
    suites = {c["suite"] for c in tt.uploaded}
    assert {"Promotions", "Newsletter", "Navigation", "Accessibility"} <= suites
    news = next(c for c in tt.uploaded if c["suite"] == "Newsletter")
    ops = [s["operation"] for s in news["steps"]]
    assert ops[:3] == ["Type", "Click", "AssertUrlContains"]  # fill + subscribe + landed on /thanks
    assert news["steps"][0]["selector"] == 'role=textbox[name="Email"s]'
    assert news["name"].startswith("Subscribe with a valid email (issue: Thank-you page")  # the issue found in it
    nav = [c for c in tt.uploaded if c["suite"] == "Navigation"]
    assert [c["name"] for c in nav] == ["Link 'Pricing' opens /pricing"]  # /docs is broken, other.example off-site

    assert result["testing_tree"]["project_url"] == "http://localhost:5000/Projects/Details/p1"
    statuses = {t["name"]: t["status"] for t in result["testing_tree"]["test_cases"]}
    assert statuses["Link 'Pricing' opens /pricing"] == "Passed" and statuses[news["name"]] == "Failed"
    report = Path(result["report_path"]).read_text()
    assert "## TestingTree test cases" in report and "replayed there:" in report and "| Newsletter |" in report


async def test_unreachable_testing_tree_does_not_fail_the_test_run(web_container, monkeypatch):
    monkeypatch.setattr(FakeTestingTree, "unreachable", True)
    record = await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    assert record.status == TaskStatus.COMPLETED
    result = record.agent_results["webtest"].result
    assert "not reachable" in result["testing_tree"]["error"]
    assert "Not saved: TestingTree is not reachable" in Path(result["report_path"]).read_text()


def test_recorder_dedupes_numbers_variants_and_always_ends_with_a_check():
    rec = FlowRecorder("http://s/")
    sub = ("click", 'role=button[name="Subscribe"s]')
    for _ in range(2):  # the model announces the same flow twice and repeats it exactly: one test case
        rec.begin("Newsletter", "Subscribe", "http://s/")
        rec.record(["fill", "@e3", "qa@example.com"], 'role=textbox[name="Email"s]', "Email", "http://s/", "http://s/", "✓")
        rec.record([sub[0], "@e4"], sub[1], "Subscribe", "http://s/", "http://s/thanks?email=qa%40example.com", "✓")
        rec.begin("Other", "Break", "http://s/thanks")
    rec.begin("Newsletter", "Subscribe", "http://s/")  # same name, different steps: numbered
    rec.record([sub[0], "@e4"], sub[1], "Subscribe", "http://s/", "http://s/", "✓")  # blocked submit: URL unchanged

    cases = rec.test_cases()
    names = [c["name"] for c in cases if c["suite"] == "Newsletter"]
    assert names == ["Subscribe", "Subscribe (2)"]
    first, second = (c for c in cases if c["suite"] == "Newsletter")
    assert first["steps"][-1] == {"operation": "AssertUrlContains", "value": "/thanks"}  # path only, no echoed query
    assert second["steps"][-1] == {"operation": "AssertUrlContains", "value": "http://s/"}  # added: it stayed put


def test_inputs_made_before_a_new_flow_is_named_are_carried_into_it():
    rec = FlowRecorder("http://s/")
    rec.begin("Newsletter", "Fill the form", "http://s/")
    rec.record(["fill", "@e3", "qa@example.com"], 'role=textbox[name="Email"s]', "Email", "http://s/", "http://s/", "✓")
    rec.begin("Newsletter", "Submit it", "http://s/")  # named between filling and submitting
    rec.record(["click", "@e4"], 'role=button[name="Subscribe"s]', "Subscribe", "http://s/", "http://s/thanks", "✓")
    submit = next(c for c in rec.test_cases() if c["name"] == "Submit it")
    assert [s["operation"] for s in submit["steps"]] == ["Type", "Click", "AssertUrlContains"]


def test_accessibility_suite_audits_each_page_once():
    cases = accessibility_suite(["http://s/", "http://s/thanks?email=a", "http://s/thanks", "http://s/pricing"])
    assert [c["name"] for c in cases] == ["Accessibility audit of http://s/", "Accessibility audit of /thanks",
                                          "Accessibility audit of /pricing"]


async def test_a_second_run_adds_only_new_test_cases_and_replays_all(web_container, monkeypatch):
    await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    first = len(FakeTestingTree.saved)
    replies = iter(SCRIPT)  # replay the same script: the same flows again

    async def again(self, task, user, *, system=None, **kw):
        return GenerationOutput(text=json.dumps(next(replies)), model="fast_text")

    monkeypatch.setattr(WebTesterAgent, "generate", again)
    record = await web_container.run_and_wait(TaskRequest(prompt="Test the website http://localhost:3000/"), timeout=20)
    tt = record.agent_results["webtest"].result["testing_tree"]
    assert len(FakeTestingTree.saved) == first and tt["added"] == 0 and tt["already_saved"] == first
    assert all(t["status"] for t in tt["test_cases"])  # the existing ones were replayed


def test_flows_are_trimmed_and_noise_is_dropped():
    rec = FlowRecorder("http://s/")
    email, sub, home = 'role=textbox[name="Email"s]', 'role=button[name="Subscribe"s]', 'role=link[name="Back home"s]'
    rec.begin("Newsletter", "Subscribe", "http://s/")
    rec.record(["fill", "@e3", "qa@example.com"], email, "Email", "http://s/", "http://s/", "✓")
    rec.record(["click", "@e4"], sub, "Subscribe", "http://s/", "http://s/thanks", "✓")
    for _ in range(2):  # read twice: one assertion
        rec.record(["get", "text", "@e1"], None, None, "http://s/thanks", "http://s/thanks", "Thanks!")
    rec.record(["click", "@e9"], home, "Back home", "http://s/thanks", "http://s/", "✓")  # setup for the next flow
    rec.begin("Newsletter", "Typed only", "http://s/")
    rec.record(["fill", "@e3", "x@example.com"], email, "Email", "http://s/", "http://s/", "✓")
    rec.begin("Navigation", "Go home", "http://s/thanks")
    rec.record(["click", "@e9"], home, "Back home", "http://s/thanks", "http://s/", "✓")

    (case,) = rec.test_cases()  # "Typed only" never submitted, "Go home" only followed a link
    assert [s["operation"] for s in case["steps"]] == ["Type", "Click", "AssertUrlContains", "AssertTextContains"]
