"""Web research agent: extraction, retrieval, safety, citations, the research loop and its API.

No network: search uses a static provider, pages come from httpx.MockTransport, DNS from a fake resolver.
"""

from __future__ import annotations

import json
import time

import httpx
import pytest
from fastapi.testclient import TestClient

from app.agents.manager_agent import ManagerAgent
from app.agents.research_agent import NO_GPU_MESSAGE, OFFLINE_NOTICE
from app.core.events import EventType
from app.main import create_app
from app.research.browser.browser import Browser
from app.research.browser.extractor import extract_html
from app.research.browser.safety import detect_restriction, validate_url
from app.research.citations import CitationManager
from app.research.errors import SearchError, UnsafeURLError
from app.research.retrieval.chunker import chunk_document
from app.research.retrieval.deduplicator import canonical_url, dedupe_chunks
from app.research.retrieval.ranker import retrieve_relevant_chunks
from app.research.schemas import Chunk, Document, FetchStatus, SearchResult
from app.research.search.base import SearchProvider
from app.research.search.manager import SearchManager
from app.research.search.providers.custom import StaticSearchProvider
from app.schemas.task import TaskOptions, TaskRequest
from tests.fakes import CPU_DEVICE, CUDA_DEVICE, FakeFactory, build_container, make_settings

# ---------------------------------------------------------------- fixtures: a tiny fake web
PAGES = {
    "https://alpha.example/a": (
        "<html><head><title>Alpha report</title><meta name='author' content='Ann'>"
        "<meta property='article:published_time' content='2026-09-01'></head><body>"
        "<nav>Home | Products | Login</nav><div class='cookie-consent'>We use cookies to track you.</div>"
        "<article><h1>Alpha report</h1><p>The fact value is 42 according to the Alpha measurement team.</p>"
        "<p>Further details about the fact value and its history are described here at length.</p></article>"
        "<footer>(c) Alpha</footer><script>tracker()</script></body></html>"
    ),
    "https://beta.example/b": (
        "<html><head><title>Beta notes</title></head><body><main><h2>Beta</h2>"
        "<p>Beta also measured the fact value as 42 in a separate study last month.</p>"
        "<p>Additional context on the fact measurement methodology.</p></main></body></html>"
    ),
    "https://gamma.example/c": (
        "<html><head><title>Gamma</title></head><body><main>"
        "<p>CONFLICT: Gamma reports the fact value as 41, not 42.</p></main></body></html>"
    ),
    "https://captcha.example/x": (
        "<html><head><title>Just a moment...</title></head><body><div class='cf-turnstile'></div></body></html>"
    ),
}


def handler(request: httpx.Request) -> httpx.Response:
    url = str(request.url)
    if url.endswith("/robots.txt"):
        if request.url.host == "robots.example":
            return httpx.Response(200, text="User-agent: *\nDisallow: /")
        return httpx.Response(404)
    if request.url.host == "slow.example":
        raise httpx.ReadTimeout("timed out", request=request)
    if request.url.host == "redirect.example":
        return httpx.Response(302, headers={"location": "http://10.0.0.5/admin"})
    if request.url.host == "pdf.example":
        return httpx.Response(200, content=b"%PDF-1.4", headers={"content-type": "application/pdf"})
    if request.url.host == "big.example":
        return httpx.Response(200, content=b"<p>" + b"x" * 5000 + b"</p>", headers={"content-type": "text/html"})
    if url in PAGES:
        status = 403 if "captcha" in url else 200
        return httpx.Response(status, text=PAGES[url], headers={"content-type": "text/html; charset=utf-8"})
    return httpx.Response(404, text="<html><body>Not found</body></html>", headers={"content-type": "text/html"})


async def fake_resolver(host: str) -> list[str]:
    return ["10.1.2.3"] if host == "intranet.example" else ["93.184.216.34"]


def make_browser(cfg, **overrides) -> Browser:
    cfg = cfg.model_copy(update={"use_playwright": "never", **overrides})
    return Browser(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), resolver=fake_resolver)


def result(url: str, title: str = "", rank: int = 1) -> SearchResult:
    return SearchResult(url=url, title=title or url, snippet="fact value measurement", rank=rank)


QUESTION = "What is the latest fact value?"


def research_settings(tmp_path, **research):
    settings = make_settings(tmp_path)
    settings.research = settings.research.model_copy(update={"reasoning_model": "base", **research})
    return settings


def wire(container, provider: SearchProvider) -> None:
    agent = container.agents.get("research")
    agent.search_factory = lambda cfg: SearchManager(provider, cfg, retries=0)
    agent.browser_factory = lambda cfg: make_browser(cfg)


def research_request(depth: str = "normal") -> TaskRequest:
    return TaskRequest(
        prompt=QUESTION, options=TaskOptions(agents=["research"], skip_validation=True, params={"depth": depth})
    )


@pytest.fixture
async def gpu_container(tmp_path):
    c = build_container(research_settings(tmp_path), factory=FakeFactory(), device=CUDA_DEVICE)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


def payload_of(record) -> dict:
    return next(r.result for r in record.agent_results.values() if r.agent == "research")


# ---------------------------------------------------------------- extraction & retrieval
def test_extractor_removes_page_chrome_and_keeps_content():
    html = (
        "<html><head><title>Spec</title><script type='application/ld+json'>"
        '{"@type": "Article", "datePublished": "2025-01-06", "author": {"name": "Jane"}}</script></head><body>'
        "<header><nav>Menu</nav></header><div id='gdpr-banner'>Accept cookies</div><main><h2>Memory</h2>"
        "<p>The card has 32 GB of GDDR7 memory.</p><table><tr><th>Spec</th><th>Value</th></tr>"
        "<tr><td>TDP</td><td>575 W</td></tr></table><ul><li>Boost 2.41 GHz</li></ul>"
        "<p>The card has 32 GB of GDDR7 memory.</p><div class='ad-slot'>Buy now</div></main>"
        "<footer>Footer</footer><script>x()</script></body></html>"
    )
    doc = extract_html(html)
    assert doc.published_date == "2025-01-06" and doc.author == "Jane"
    assert "## Memory" in doc.text and "| TDP | 575 W |" in doc.text and "- Boost 2.41 GHz" in doc.text
    assert doc.text.count("32 GB of GDDR7") == 1  # duplicated paragraph removed
    for junk in ("Menu", "Accept cookies", "Footer", "x()", "Buy now"):
        assert junk not in doc.text


def test_chunking_and_relevance_ranking():
    text = "\n\n".join(
        [f"## Section {i}\n\nFiller paragraph number {i} about unrelated gardening topics." for i in range(8)]
        + ["## Memory\n\nThe RTX 5090 has 32 GB of GDDR7 memory."]
    )
    doc = Document(url="https://x.example", text=text)
    assert len(chunk_document(doc, max_chars=200)) >= 5
    top = retrieve_relevant_chunks(doc, "How much memory does the RTX 5090 have?", max_chunks=1)
    assert len(top) == 1 and "32 GB" in top[0].text


def test_dedup_urls_and_chunks():
    assert canonical_url("http://www.Example.com/a/?utm_source=x&b=2") == canonical_url("https://example.com/a?b=2")
    a = Chunk(doc_url="1", index=0, text="the quick brown fox jumps over the lazy dog near the river bank today")
    b = Chunk(doc_url="2", index=0, text="The quick brown fox jumps over the lazy dog near the river bank today!")
    c = Chunk(doc_url="3", index=0, text="completely different content about graphics card memory sizes")
    assert [x.doc_url for x in dedupe_chunks([a, b, c])] == ["1", "3"]


# ---------------------------------------------------------------- citations
def test_citations_only_for_visited_pages_and_sanitized():
    cm = CitationManager()
    ok = Document(url="https://a.example/x", title="A", domain="a.example", text="body")
    assert cm.register(ok) == 1
    assert cm.register(ok.model_copy(update={"url": "https://www.a.example/x/"})) == 1  # same page
    with pytest.raises(ValueError):
        cm.register(Document(url="https://b.example", status=FetchStatus.ERROR, error="404"))
    text, cited, invalid = cm.sanitize("Claim [1]. Fake [4]. Range [1-2]. Link https://evil.example/p.")
    assert cited == [1] and invalid == [2, 4]
    assert "[4]" not in text and "evil.example" not in text and "Claim [1]" in text


# ---------------------------------------------------------------- browser safety
@pytest.mark.parametrize(
    "url",
    ["ftp://a.example/x", "http://localhost/x", "http://127.0.0.1/", "http://10.0.0.1/", "http://[::1]/",
     "http://user:pw@a.example/", "http://intranet.example/", "http://a.example:22/", "not a url"],
)
async def test_unsafe_urls_rejected(url):
    with pytest.raises(UnsafeURLError):
        await validate_url(url, fake_resolver)


async def test_public_url_allowed():
    assert await validate_url("https://alpha.example/a", fake_resolver) == "https://alpha.example/a"


def test_restriction_detection():
    assert detect_restriction("<div class='g-recaptcha'></div>", 200, 10) == "CAPTCHA / bot challenge"
    assert detect_restriction("<p>Subscribe to continue reading</p>", 200, 100) == "paywall"
    assert detect_restriction("<form><input type='password'></form>", 200, 50) == "login wall"
    assert detect_restriction("<p>normal</p>", 200, 5000) is None


@pytest.mark.parametrize(
    ("url", "status", "error_part"),
    [
        ("https://alpha.example/a", FetchStatus.OK, None),
        ("https://nowhere.example/missing", FetchStatus.ERROR, "HTTP 404"),
        ("https://slow.example/", FetchStatus.ERROR, "timed out"),
        ("https://redirect.example/", FetchStatus.ERROR, "non-public"),
        ("https://captcha.example/x", FetchStatus.BLOCKED, "403"),
        ("https://robots.example/page", FetchStatus.BLOCKED, "robots.txt"),
        ("https://pdf.example/doc.pdf", FetchStatus.UNSUPPORTED, "application/pdf"),
        ("http://intranet.example/", FetchStatus.ERROR, "non-public"),
    ],
)
async def test_browser_outcomes(tmp_path, url, status, error_part):
    browser = make_browser(make_settings(tmp_path).research)
    try:
        doc = await browser.open(url)
    finally:
        await browser.aclose()
    assert doc.status == status, doc.error
    if error_part:
        assert error_part in (doc.error or "")
    else:
        assert doc.title == "Alpha report" and "42" in doc.text and "cookies" not in doc.text
        assert doc.published_date == "2026-09-01" and doc.author == "Ann"


async def test_browser_caps_page_size(tmp_path):
    browser = make_browser(make_settings(tmp_path).research, max_page_mb=0.001)
    try:
        doc = await browser.open("https://big.example/")
    finally:
        await browser.aclose()
    assert doc.status == FetchStatus.ERROR and "too large" in doc.error  # declared Content-Length over the cap


# ---------------------------------------------------------------- search manager
class FailingProvider(SearchProvider):
    name = "failing"

    async def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        raise SearchError("provider down")


async def test_search_manager_merges_dedupes_and_reports_failures(tmp_path):
    cfg = make_settings(tmp_path).research
    provider = StaticSearchProvider({
        "a": [result("https://alpha.example/a"), result("https://beta.example/b", rank=2)],
        "b": [result("https://www.alpha.example/a/"), result("https://gamma.example/c", rank=2)],
    })
    merged, errors = await SearchManager(provider, cfg).search_many(["a", "b"])
    assert [r.url for r in merged] == ["https://alpha.example/a", "https://beta.example/b", "https://gamma.example/c"]
    assert errors == []
    with pytest.raises(SearchError):
        await SearchManager(FailingProvider(), cfg, retries=0).search_many(["a"])


# ---------------------------------------------------------------- routing
def test_manager_routes_fresh_questions_to_research():
    assert ManagerAgent.classify_intent("What is the latest stable Python release?", False) == "research"
    assert ManagerAgent.classify_intent("Search the web for RTX 5090 specs", False) == "research"
    assert ManagerAgent.classify_intent("Explain why the sky is blue", False) == "explain"
    # lookups on people / platforms / links need the web even without "search the web"
    for prompt in [
        "search for linkedin posts of username",
        "find linkedin posts by satya nadella",
        "what did elon musk tweet yesterday",
        "show me the github repos of torvalds",
        "can you google the best 4k monitors",
        "find me articles about local LLMs",
        "summarize https://example.com/post",
        "who is the current CEO of OpenAI",
        "linkedin profile of jane smith",
    ]:
        assert ManagerAgent.classify_intent(prompt, False) == "research", prompt
    # writing for a platform, or "search" as a programming topic, stays local
    assert ManagerAgent.classify_intent("Write a LinkedIn post about my promotion", False) == "generate"
    assert ManagerAgent.classify_intent("Implement binary search in Python", False) == "generate"
    assert ManagerAgent.classify_intent("Write a function to search for a key in a BST", False) == "generate"
    assert ManagerAgent.classify_intent("Find the bug in this code: x = 1/0", False) == "generate"
    long_text = "Summarize this article: " + "today the latest news " * 60
    assert ManagerAgent.classify_intent(long_text, False) == "summarize"


async def test_research_intent_runs_research_agent(gpu_container):
    wire(gpu_container, StaticSearchProvider([result("https://alpha.example/a"), result("https://beta.example/b", rank=2)]))
    record = await gpu_container.run_and_wait(TaskRequest(prompt=QUESTION), timeout=20)
    assert record.status == "completed", record.error
    assert record.plan.intent == "research" and [s.agent for s in record.plan.steps] == ["research"]


# ---------------------------------------------------------------- the research loop
async def test_research_answer_has_only_real_citations(gpu_container):
    provider = StaticSearchProvider([
        result("https://alpha.example/a", rank=1),
        result("https://beta.example/b", rank=2),
        result("https://nowhere.example/missing", rank=3),
        result("https://captcha.example/x", rank=4),
    ])
    wire(gpu_container, provider)
    record = await gpu_container.run_and_wait(research_request("quick"), timeout=20)
    assert record.status == "completed", record.error
    payload = payload_of(record)
    answer = record.result["answer"]

    visited = {s["url"] for s in payload["sources"]}
    assert visited == {"https://alpha.example/a", "https://beta.example/b"}
    assert {f["url"] for f in payload["failed_sources"]} == {"https://nowhere.example/missing", "https://captcha.example/x"}
    # Model-invented markers/URLs are removed; the Sources list holds only visited pages.
    assert "[99]" not in answer and "[7]" not in answer and "evil.example" not in answer and "made.up" not in answer
    assert "[1]" in answer and payload["invalid_citations"] == [7, 99]
    sources_section = answer.split("## Sources", 1)[1]
    assert sources_section.count("https://") == len(payload["cited"]) and all(u in visited for u in
        [part.split(")")[0] for part in sources_section.split("](")[1:]])
    for heading in ("## Answer", "## Key Findings", "## Confidence", "## Sources"):
        assert heading in answer
    # Untraceable claims were dropped; every evidence item points at a visited source.
    assert all("cheese" not in e["claim"] for e in payload["evidence"])
    assert {e["source_id"] for e in payload["evidence"]} <= {s["id"] for s in payload["sources"]}
    assert all(sid in {s["id"] for s in payload["sources"]} for c in payload["claims"] for sid in c["sources"])

    events = [e.event for e in gpu_container.events.history(record.task_id)]
    assert EventType.AGENT_PROGRESS in events and EventType.RESEARCH_SOURCE in events
    messages = [e.data.get("message", "") for e in gpu_container.events.history(record.task_id)]
    assert any(m.startswith("Reading source 1/") for m in messages)
    assert any(m.startswith("Comparing") for m in messages) and "Verifying claims across sources" in messages


async def test_research_searches_again_when_information_is_missing(gpu_container):
    provider = StaticSearchProvider({
        QUESTION: [result("https://alpha.example/a")],
        "q1": [result("https://alpha.example/a")],
        "follow up": [result("https://beta.example/b")],
    })
    wire(gpu_container, provider)
    record = await gpu_container.run_and_wait(research_request("normal"), timeout=20)
    payload = payload_of(record)
    assert record.status == "completed", record.error
    assert "follow up" in provider.queries and payload["iterations"] == 2
    assert len(payload["sources"]) == 2


async def test_conflicting_sources_are_reported(gpu_container):
    wire(gpu_container, StaticSearchProvider([result("https://alpha.example/a"), result("https://gamma.example/c", rank=2)]))
    record = await gpu_container.run_and_wait(research_request("quick"), timeout=20)
    payload = payload_of(record)
    assert [c["status"] for c in payload["claims"]] == ["CONFLICTING"]
    assert "## Disagreements Between Sources" in record.result["answer"]
    assert payload["confidence"] != "High"


@pytest.mark.parametrize("provider", [StaticSearchProvider([]), FailingProvider()], ids=["empty", "search-failure"])
async def test_no_evidence_means_no_invented_answer(gpu_container, provider):
    wire(gpu_container, provider)
    record = await gpu_container.run_and_wait(research_request("quick"), timeout=20)
    assert record.status == "completed", record.error
    payload = payload_of(record)
    assert payload["confidence"] == "Low" and payload["sources"] == []
    assert "could not find enough reliable information" in record.result["answer"]


async def test_offline_mode_never_touches_the_web(tmp_path):
    settings = research_settings(tmp_path, web_enabled=False)
    c = build_container(settings, factory=FakeFactory(), device=CUDA_DEVICE)
    await c.start()
    try:
        agent = c.agents.get("research")
        agent.search_factory = agent.browser_factory = lambda cfg: pytest.fail("web used in offline mode")
        record = await c.run_and_wait(research_request(), timeout=20)
    finally:
        await c.stop()
    answer = record.result["answer"]
    assert record.status == "completed" and answer.startswith(OFFLINE_NOTICE)
    assert "[3]" not in answer and "made.up" not in answer and payload_of(record)["mode"] == "offline"


async def test_research_requires_cuda_gpu(tmp_path, factory):
    c = build_container(research_settings(tmp_path), factory=factory, device=CPU_DEVICE)
    await c.start()
    try:
        wire(c, StaticSearchProvider([result("https://alpha.example/a")]))
        record = await c.run_and_wait(research_request(), timeout=20)
    finally:
        await c.stop()
    assert record.status == "failed" and NO_GPU_MESSAGE in record.error
    assert record.agent_results["research"].attempts == 1  # not retried
    assert factory.calls["base"] == 0  # no inference was attempted


async def test_cpu_allowed_only_when_explicitly_enabled(tmp_path):
    c = build_container(research_settings(tmp_path, allow_cpu=True), factory=FakeFactory(), device=CPU_DEVICE)
    await c.start()
    try:
        wire(c, StaticSearchProvider([result("https://alpha.example/a")]))
        record = await c.run_and_wait(research_request("quick"), timeout=20)
    finally:
        await c.stop()
    assert record.status == "completed", record.error


# ---------------------------------------------------------------- API + SSE
@pytest.fixture
def api(tmp_path):
    settings = research_settings(tmp_path)
    holder = {}

    def factory(s):
        c = build_container(s, factory=FakeFactory(), device=CUDA_DEVICE)
        wire(c, StaticSearchProvider([result("https://alpha.example/a"), result("https://beta.example/b", rank=2)]))
        holder["c"] = c
        return c

    with TestClient(create_app(settings, container_factory=factory)) as client:
        yield client


def wait_research(client, rid, timeout=10) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        body = client.get(f"/api/research/{rid}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.05)
    raise AssertionError("research did not finish")


def test_research_api_and_sse_stream(api):
    res = api.post("/api/research", json={"query": QUESTION, "depth": "quick"})
    assert res.status_code == 202
    rid = res.json()["research_id"]
    assert res.json()["stream"] == f"/api/stream/{rid}"

    frames = []
    with api.stream("GET", f"/api/stream/{rid}") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        for line in stream.iter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))
                if frames[-1]["type"] in ("answer", "error"):
                    break
    types = [f["type"] for f in frames]
    assert "status" in types and "source" in types and types[-1] == "answer"
    assert "## Sources" in frames[-1]["content"]

    body = wait_research(api, rid)
    assert body["status"] == "completed" and body["confidence"] in ("High", "Medium", "Low")
    assert {s["url"] for s in body["sources"]} == {"https://alpha.example/a", "https://beta.example/b"}
    sources = api.get(f"/api/research/{rid}/sources").json()
    assert len(sources["sources"]) == 2 and sources["evidence"]
    assert any(r["research_id"] == rid for r in api.get("/api/research").json())


def test_research_api_validation(api):
    assert api.post("/api/research", json={"query": ""}).status_code == 422
    assert api.post("/api/research", json={"query": "x", "depth": "extreme"}).status_code == 422
    assert api.get("/api/research/nope").status_code == 404
    assert api.get("/api/stream/nope").status_code == 404


def test_status_gpu_and_model_endpoints(api):
    status = api.get("/api/status").json()
    assert status["ai_inference"] == "local" and status["external_ai_apis"] == "disabled"
    assert status["mode"] == "LOCAL AI + WEB" and status["cuda_available"] is True
    assert status["research_model"]["resolved"] == "base"
    gpu = api.get("/api/gpu").json()
    assert gpu["cuda_available"] is True and gpu["device"] == "cuda:0" and gpu["gpu_mode"] == "single"

    loaded = api.post("/api/models/load", json={"name": "fast_text"})
    assert loaded.status_code == 200 and loaded.json()["state"] == "LOADED"
    assert api.post("/api/models/unload", json={"name": "fast_text"}).json()["unloaded"] is True
    assert api.post("/api/models/download", json={"hf_id": "not a repo"}).status_code == 400
    models = {m["name"]: m for m in api.get("/api/models").json()}
    assert models["base"]["engine"] == "transformers" and models["base"]["format"] == "hf"
