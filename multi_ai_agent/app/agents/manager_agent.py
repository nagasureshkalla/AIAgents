"""Manager agent: understands the request and decides which specialist agents are needed."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import AgentNotFoundError, InvalidRequestError
from app.schemas.agent import AgentTask
from app.utils.text import extract_json

logger = logging.getLogger(__name__)

# Ordered: first match wins.
IMAGE_INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("screenshot_debug", re.compile(r"\b(screenshot|wrong|error|bug|broken|issue|crash|warning|ui|layout)\b", re.I)),
    (
        "image_analysis",
        re.compile(r"\b(analy[sz]e|describe|explain|caption|what(?:'s| is) happening|scene|overview)\b", re.I),
    ),
    ("visual_qa", re.compile(r"\?|\b(what|which|how many|where|who|is there|are there|does|do you see|count)\b", re.I)),
]
# Requests that need fresh, external facts. Checked first, but only for short prompts (long prompts usually
# carry their own text to summarize/extract, which must not trigger a web search).
WEB_SITES = (
    r"linked\s?in|twitter|tweets?|x\.com|reddit|subreddit|github|gitlab|youtube|instagram|facebook|tiktok|"
    r"threads|mastodon|bluesky|medium|substack|wikipedia|stack\s?overflow|hacker\s?news|imdb|yelp|glassdoor|"
    r"crunchbase|product\s?hunt|google\s(?:news|scholar|maps)|arxiv|pypi|npm"
)
LOOKUP_VERBS = (
    r"search|find|look(?:ing)?\s(?:up|for|at)|lookup|show|get|fetch|check|browse|pull|scrape|read|see|"
    r"what\s(?:did|does|has|have|is)|recent|latest|new"
)
RESEARCH_PATTERNS: list[re.Pattern[str]] = [
    # explicit "go to the web" wording and freshness cues
    re.compile(
        r"\b(search (?:the )?(?:web|internet|online|google)|on the (?:web|internet)|look (?:it |this |that )?up|"
        r"lookup|browse|research|latest|newest|most recent|recent news|news|today|tonight|yesterday|right now|"
        r"currently|this (?:week|month|year)|as of|up[- ]to[- ]date|price of|stock price|exchange rate|weather|"
        r"release date|with (?:sources|citations)|cite (?:your )?sources)\b",
        re.I,
    ),
    # a request that opens with a search verb: "search for ...", "can you google ...", "find me articles on ..."
    # ("implement binary search" / "search for a key in a BST" don't start the sentence with the verb)
    re.compile(
        r"(?:^|[.!?\n]\s*)(?:please\s+|pls\s+)?(?:(?:can|could|would) you\s+|(?:i )?(?:want|need) you to\s+)?"
        r"(?:search(?:\s+(?:for|up|on|about))?|google|look\s+up|look\s+for|lookup)\b",
        re.I,
    ),
    re.compile(
        r"\bfind (?:me )?(?:\w+ ){0,2}(?:info|information|details|articles?|posts?|news|sources|reviews?|"
        r"profiles?|links?|websites?|pages?|papers?|jobs?|videos?|tweets?)\b",
        re.I,
    ),
    re.compile(r"\b(?:find|search|look|check)\b[^.?!\n]{0,60}\bonline\b", re.I),
    # an online platform plus a lookup verb ("search linkedin posts of X", "what did X tweet") - but not
    # "write a LinkedIn post", which has no lookup verb
    re.compile(rf"\b(?:{LOOKUP_VERBS})\b[^.?!\n]{{0,80}}\b(?:{WEB_SITES})\b", re.I),
    re.compile(rf"\b(?:{WEB_SITES})\b[^.?!\n]{{0,40}}\b(?:posts?|profile|account|page|tweets|videos|repos?|"
               rf"activity|followers)\s+(?:of|by|from|for)\b", re.I),
    # who currently holds a role ("who is the CEO of OpenAI") changes over time
    re.compile(
        r"\bwho (?:is|are|was|runs|owns|leads) (?:the )?(?:current |new )?(?:ceo|cto|cfo|president|founder|owner|"
        r"head|leader|prime minister|chair(?:man|woman)?|coach|maintainer)s?\b",
        re.I,
    ),
    # a link to read
    re.compile(r"https?://|\bwww\.\S+", re.I),
]
RESEARCH_RULE = ("research", RESEARCH_PATTERNS)
# Readable summary of RESEARCH_PATTERNS for the /api/routing help view.
RESEARCH_PHRASES = [
    "search for ...", "google ...", "look up", "find articles / posts / profiles / news ...", "... online",
    "LinkedIn / X / Reddit / GitHub ... + search, find, show, what did", "posts / profile of ...", "a URL",
    "who is the CEO of ...", "latest", "news", "today", "currently", "price of", "weather", "with sources",
]
RESEARCH_MAX_PROMPT_CHARS = 600

# Requests to operate applications on this computer (Control Panel agent). Checked before research: "Open Chrome and
# search Google for X" is something to DO on the desktop, not a question to research.
DESKTOP_APPS = (
    r"chrome|chromium|firefox|browser|vs ?code|visual studio code|terminal|console|file manager|files app|nautilus|"
    r"finder|calculator|text editor|gedit|libreoffice|writer|spreadsheet|blender|gimp|steam|spotify|settings|"
    r"system monitor|thunderbird|slack|discord|obs|vlc"
)
CONTROL_PATTERNS: list[re.Pattern[str]] = [
    # "open / launch / close / switch to <app>" at the start of a sentence
    re.compile(r"(?:^|[.!?\n]\s*)(?:please\s+)?(?:(?:can|could|would) you\s+)?(?:open|launch|close|quit|exit|"
               r"switch to|bring up|minimi[sz]e|maximi[sz]e)\s+(?!(?:(?:the|my|a|an|new)\s+)*(?:source\b|-|ai\b|"
               r"ended\b|questions?\b|letter|issues?\b|discussion|debate|conversation|mind\b|account|position|door|"
               r"heart|up\b))\w", re.I),
    re.compile(rf"^(?:please\s+)?in\s+(?:the\s+|my\s+)?(?:{DESKTOP_APPS})\s*,?\s+(?:go|open|search|click|type|"
               rf"find|create|navigate|play|show|close|download|save)\b", re.I),
    re.compile(rf"\b(?:start|run|use)\s+(?:the\s+)?(?:{DESKTOP_APPS}|app|application|program)\b", re.I),
    # explicit GUI operations
    re.compile(r"\b(?:click|double[- ]click|right[- ]click|drag and drop|scroll (?:down|up))\b", re.I),
    re.compile(r"\b(?:on|control|use) (?:my|this|the) (?:computer|desktop|screen|pc|laptop|mouse|keyboard)\b", re.I),
    # an imperative that names an app: "in Chrome, go to ...", "create main.py in VS Code"
    re.compile(rf"^(?:please\s+)?(?:go to|navigate|search|find|create|type|play|download|install|show|take|save|"
               rf"rename|move|copy)\b[^?!\n]{{0,120}}\b(?:in|using|with|on|from)\s+(?:the\s+|my\s+)?(?:{DESKTOP_APPS})\b",
               re.I),
]
CONTROL_RULE = ("computer_control", CONTROL_PATTERNS)

# Requests to test a website (Web Tester agent). Checked first: "test http://..." names a URL (research) and
# "open the site and check the form" starts like a desktop command, but both ask for QA of a web app.
WEB_HOST = (r"(?:https?://\S+|www\.\S+|localhost(?::\d+)?\S*|(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?\S*|"
            r"[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|org|net|io|dev|app|ai|co|edu|gov|us|uk|in|de|fr|xyz|site|online)\b\S*)")
OWN_SITE = (r"(?:the|my|this|our|your)\s+(?:[\w-]+\s+){0,2}?(?:web\s?site|site|web\s?app|web\s?page|landing page|"
            r"home\s?page|login page|sign\s?up (?:page|flow)|checkout|staging site)")
TEST_VERB = (r"(?:test|qa|dogfood|bug[- ]?hunt|smoke[- ]test|e2e[- ]test|regression[- ]test|exploratory[- ]test|"
             r"(?:check|verify|validate)\s+(?:that\s+)?(?:the\s+|all\s+)?(?:forms?|links?|buttons?|login|sign\s?up|checkout|"
             r"navigation|pages?))")
IMPERATIVE = r"(?:^|[.!?\n]\s*)(?:please\s+)?(?:(?:can|could|would) you\s+)?"
WEB_TEST_PATTERNS: list[re.Pattern[str]] = [
    # "test http://localhost:3000", "QA example.com signup", "please smoke test the landing page"
    re.compile(rf"{IMPERATIVE}{TEST_VERB}\b[^.?!\n]{{0,80}}?(?:\b{OWN_SITE}\b|{WEB_HOST})", re.I),
    # "find bugs / broken links on my site", "look for issues at localhost:5000"
    re.compile(rf"{IMPERATIVE}(?:find|look for|check for|hunt for)\s+(?:bugs|issues|broken links|errors|problems)\b"
               rf"[^.?!\n]{{0,60}}?\b(?:on|in|at|of)\s+(?:{OWN_SITE}|{WEB_HOST})", re.I),
]
WEB_TEST_RULE = ("web_test", WEB_TEST_PATTERNS)
WEB_TEST_PHRASES = ["test / QA / dogfood <website or URL>", "smoke test / e2e test <site>",
                    "find bugs / broken links on <site>", "check the forms / login / links on <site>"]
CONTROL_PHRASES = ["open / launch / close <app>", "start or use <app>", "click / scroll / drag", "on my computer",
                   "<do something> in Chrome / VS Code / Terminal / Files ..."]

# Job hunting (Job Agent, backed by Auto-Apply AI). Checked after desktop rules ("open Chrome and search jobs" is a
# desktop task) and before research, whose "find ... jobs" pattern would otherwise send it to a generic web search.
JOB_ROLE = r"(?:jobs?|roles?|positions?|openings?|vacanc(?:y|ies)|internships?|opportunit(?:y|ies))"
JOB_PATTERNS: list[re.Pattern[str]] = [
    re.compile(rf"\b(?:find|search(?:\s+for)?|look\s+for|hunt\s+for|get|show|list|match)\s+(?:me\s+)?(?:some\s+|new\s+|the\s+best\s+)?"
               rf"(?:[\w+#./-]+\s+){{0,5}}?{JOB_ROLE}\b", re.I),
    re.compile(rf"\bapply\s+(?:me\s+)?(?:to|for)\s+(?:[\w+#./-]+\s+){{0,6}}?{JOB_ROLE}\b", re.I),
    re.compile(r"\b(?:job\s+(?:search|hunt(?:ing)?|agent|applications?)|auto[- ]?apply|cover\s+letters?\s+for\b)", re.I),
    re.compile(rf"\b{JOB_ROLE}\b[^.?!\n]{{0,60}}\b(?:match(?:ing)?|fit|suit)(?:es|s)?\s+my\s+(?:resume|cv|profile|skills)\b", re.I),
    # "set up my job profile from ~/cv.pdf", "create my account from my resume"
    re.compile(r"\b(?:set\s*up|create|build|update|import)\b[^?!\n]{0,40}\b(?:job\s+)?(?:profile|account)\b[^?!\n]{0,40}"
               r"\b(?:resume|cv)\b", re.I),
]
JOB_RULE = ("job_search", JOB_PATTERNS)
JOB_PHRASES = ["find <role> jobs / internships", "apply to <role> jobs", "job search / job hunt / auto-apply",
               "jobs that match my resume", "set up my job profile from ~/path/resume.pdf"]

# The AirLLM agent (a 120B model streamed layer by layer: minutes per answer) only runs when asked for by name.
BIG_MODEL_RE = re.compile(r"\b(?:gpt[- ]?oss(?:[- ]?120b)?|air ?llm|120\s?b(?:illion)?(?:[- ]parameters?)?\s+model|"
                          r"(?:ask|use|using|with|via|let)\s+(?:the|your|my)\s+(?:big|biggest|largest|huge)\s+"
                          r"(?:local\s+)?model)\b", re.I)
BIG_MODEL_PHRASES = ["ask gpt-oss / gpt-oss-120b ...", "use airllm to ...", "ask the 120b model / the big model ..."]

TEXT_INTENT_RULES: list[tuple[str, re.Pattern[str] | list[re.Pattern[str]]]] = [
    ("big_model", BIG_MODEL_RE),
    WEB_TEST_RULE,
    CONTROL_RULE,
    JOB_RULE,
    RESEARCH_RULE,
    ("summarize", re.compile(r"\b(summari[sz]e|summary|tl;?dr|condense|key takeaways)\b", re.I)),
    ("extract", re.compile(r"\b(extract|pull out|list all|find all|parse)\b", re.I)),
    ("classify", re.compile(r"\b(classify|categori[sz]e|sentiment|label|which category)\b", re.I)),
    ("format", re.compile(r"\b(format|convert (?:it |this )?to|rewrite as|as json|as a table|markdown table)\b", re.I)),
    (
        "explain",
        re.compile(r"\b(explain|why|analy[sz]e|reason|compare|evaluate|pros and cons|how does|trade-?offs?)\b", re.I),
    ),
]
TEXT_INTENTS = ["big_model", "web_test", "computer_control", "job_search", "research", "summarize", "extract", "classify", "format", "explain", "generate"]

INTENT_HELP: dict[str, str] = {
    "big_model": "the user explicitly asks for gpt-oss-120b / AirLLM / the big model to answer",
    "web_test": "test or QA a website / web app in a browser: explore pages, try forms, buttons and links, find bugs",
    "computer_control": "operate applications on this computer: open/close apps, click, type, browse in Chrome, "
    "use VS Code, Terminal, Files, Calculator...",
    "job_search": "find job / internship openings, score them against the user's resume, draft or send applications",
    "research": "needs facts from the internet: searching the web, people/companies, social media posts or "
    "profiles (LinkedIn, X, Reddit, GitHub...), links, news, prices, anything recent or changing",
    "summarize": "shorten text the user provided",
    "extract": "pull specific fields or items out of text the user provided",
    "classify": "put the user's text into a category or sentiment",
    "format": "convert or reformat content the user provided (JSON, table, markdown...)",
    "explain": "reason about or explain a concept, comparison or trade-off from general knowledge",
    "generate": "write new content (emails, posts, stories, code) from general knowledge",
}

INTENT_CAPABILITIES: dict[str, list[str]] = {
    "big_model": ["big_model_generation"],  # one slow, strong answer: no extra model calls around it
    "web_test": ["web_testing"],  # the Web Tester checks every step in the browser and reports with evidence
    "computer_control": ["computer_control"],  # the Control Panel agent verifies every step on screen itself
    "job_search": ["job_search"],  # the Job Agent scores against the resume and asks before every application
    "research": ["web_research"],  # the research agent verifies its own claims against sources
    "image_analysis": ["image_analysis", "request_interpretation", "reasoning", "validation"],
    "visual_qa": ["visual_question_answering", "image_description", "reasoning", "validation"],
    "screenshot_debug": ["screenshot_analysis", "image_description", "reasoning", "validation"],
    "summarize": ["summarization", "reasoning", "validation"],
    "extract": ["extraction", "validation"],
    "classify": ["classification", "reasoning", "validation"],
    "format": ["formatting", "validation"],
    "explain": ["reasoning", "text_generation", "validation"],
    "generate": ["text_generation", "validation"],
}


class ManagerAgent(BaseAgent):
    name = "manager"
    role = "control"
    description = "Understands the user's request and decides which specialist agents are required."
    capabilities = ["request_analysis", "agent_selection", "routing"]
    default_model = "fast_text"

    @staticmethod
    def classify_intent(prompt: str, has_image: bool) -> str:
        rules = IMAGE_INTENT_RULES if has_image else TEXT_INTENT_RULES
        for intent, pattern in rules:
            if intent == "research":
                if ManagerAgent.needs_web(prompt):
                    return intent
            elif intent == "computer_control":
                if ManagerAgent.wants_desktop(prompt):
                    return intent
            elif intent == "web_test":
                if ManagerAgent.wants_web_test(prompt):
                    return intent
            elif intent == "job_search":
                if ManagerAgent.wants_jobs(prompt):
                    return intent
            elif pattern.search(prompt):
                return intent
        return "image_analysis" if has_image else "generate"

    @staticmethod
    def wants_desktop(prompt: str) -> bool:
        """True when a (short) prompt asks to operate applications on this computer."""
        if len(prompt) > RESEARCH_MAX_PROMPT_CHARS:
            return False
        return any(p.search(prompt) for p in CONTROL_PATTERNS)

    @staticmethod
    def wants_web_test(prompt: str) -> bool:
        """True when a (short) prompt asks to test a website."""
        if len(prompt) > RESEARCH_MAX_PROMPT_CHARS:
            return False
        return any(p.search(prompt) for p in WEB_TEST_PATTERNS)

    @staticmethod
    def wants_jobs(prompt: str) -> bool:
        """True when a (short) prompt asks to find or apply to jobs."""
        if len(prompt) > RESEARCH_MAX_PROMPT_CHARS:
            return False
        return any(p.search(prompt) for p in JOB_PATTERNS)

    @staticmethod
    def needs_web(prompt: str) -> bool:
        """True when a (short) prompt asks for information that has to come from the web."""
        if len(prompt) > RESEARCH_MAX_PROMPT_CHARS:
            return False
        return any(p.search(prompt) for p in RESEARCH_PATTERNS)

    async def _llm_intent(self, task: AgentTask) -> str | None:
        system = (
            "You route requests to specialist agents. Classify the user's request into exactly one intent:\n"
            + "\n".join(f"- {name}: {INTENT_HELP[name]}" for name in TEXT_INTENTS)
            + '\nReply with JSON only, e.g. {"intent": "summarize"}.'
        )
        try:
            out = await self.generate(task, task.prompt[:2000], system=system, max_tokens=24, temperature=0.0)
        except Exception as exc:
            logger.info("LLM routing unavailable (%s); using rules", exc)
            return None
        data = extract_json(out.text)
        intent = data.get("intent") if isinstance(data, dict) else None
        return intent if intent in TEXT_INTENTS else None

    async def _prepare_models(self, task: AgentTask, agent_names: list[str]) -> dict[str, Any]:
        """Load what the chosen agents need and unload idle models they don't (see ModelManager.prepare)."""
        registry = self.ctx.registry
        assert registry is not None
        required: list[str] = []
        for name in agent_names:
            if registry.has(name):
                required += [m for m in registry.get(name).required_models() if m not in required]
        if not required:
            return {"required": [], "loaded": [], "unloaded": [], "resident": [], "skipped": {}}

        def note(kind: str, model: str) -> None:
            self.progress(task, f"{'Unloading' if kind == 'unloading' else 'Loading'} model {model}",
                          model=model, model_action=kind)

        try:
            return await self.ctx.models.prepare(required, on_event=note)
        except Exception as exc:
            logger.warning("model preparation failed: %s", exc)
            return {"required": required, "loaded": [], "unloaded": [], "resident": [], "skipped": {"*": str(exc)}}

    def _keyword_agents(self, prompt: str, chosen: list[str]) -> list[str]:
        """Custom agents whose declared keywords appear in the prompt (enables drop-in agents)."""
        registry = self.ctx.registry
        assert registry is not None
        extra: list[str] = []
        for agent in registry.workers():
            if agent.name in chosen or not agent.keywords:
                continue
            if any(re.search(rf"\b{re.escape(k)}\b", prompt, re.I) for k in agent.keywords):
                extra.append(agent.name)
        return extra

    async def run(self, task: AgentTask) -> dict[str, Any]:
        registry = self.ctx.registry
        assert registry is not None
        has_image = task.image_path is not None
        forced: list[str] | None = task.params.get("forced_agents")

        intent = self.classify_intent(task.prompt, has_image)
        routing = "rules"
        # Web cues are high-precision: never let the small routing model talk a web request out of research.
        if (not has_image and not forced and intent not in ("research", "computer_control", "web_test", "big_model")
                and intent != "job_search" and self.ctx.settings.runtime.use_llm_routing):
            llm_intent = await self._llm_intent(task)
            if llm_intent:
                intent, routing = llm_intent, "llm"

        if intent == "research" and registry.first_for("web_research") is None:
            intent = "generate"  # research agent disabled: answer from the local model instead
        if intent == "computer_control" and registry.first_for("computer_control") is None:
            intent = "generate"  # Control Panel disabled: explain instead of acting
        if intent == "web_test" and registry.first_for("web_testing") is None:
            intent = "generate"  # Web Tester disabled: answer from the local model instead
        if intent == "big_model" and registry.first_for("big_model_generation") is None:
            intent = "generate"  # AirLLM agent disabled: answer with the regular text agents
        if intent == "job_search" and registry.first_for("job_search") is None:
            intent = "research" if self.needs_web(task.prompt) and registry.first_for("web_research") else "generate"
        capabilities = list(INTENT_CAPABILITIES.get(intent, INTENT_CAPABILITIES["generate"]))
        if task.params.get("skip_validation"):
            capabilities = [c for c in capabilities if c != "validation"]

        agents: list[str] = []
        missing: list[str] = []
        for cap in capabilities:
            agent = registry.first_for(cap)
            if agent is None:
                missing.append(cap)
            elif agent.name not in agents:
                agents.append(agent.name)

        if forced:
            for name in forced:
                try:
                    agent = registry.get(name)
                except AgentNotFoundError as exc:
                    raise InvalidRequestError(str(exc)) from exc
                if agent.role != "worker":
                    raise InvalidRequestError(f"Agent '{name}' is a control agent and cannot be forced")
            agents, capabilities, missing, routing = list(forced), [], [], "forced"

        extra = [] if forced else self._keyword_agents(task.prompt, agents)
        input_type = "text"
        if has_image:
            input_type = "multimodal" if intent in ("visual_qa", "screenshot_debug") else "image"

        models = await self._prepare_models(task, agents + extra)

        summary = f"intent={intent}; agents={', '.join(agents + extra)}"
        if models.get("required"):
            summary += f"; models={', '.join(models['required'])}"
        if models.get("unloaded"):
            summary += f"; unloaded={', '.join(models['unloaded'])}"
        if missing:
            summary += f"; unavailable capabilities={', '.join(missing)}"
        return {
            "models": models,
            "intent": intent,
            "input_type": input_type,
            "capabilities": capabilities,
            "agents": agents,
            "extra_agents": extra,
            "forced_agents": forced or [],
            "missing_capabilities": missing,
            "routing": routing,
            "text": summary,
        }
