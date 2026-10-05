"""Validation agent: rule-based quality checks plus a short model review of the candidate answer."""

from __future__ import annotations

import logging
import re
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import ModelNotFoundError
from app.schemas.agent import AgentTask
from app.utils.text import extract_json, max_sentence_repeats, truncate

logger = logging.getLogger(__name__)

REVIEW_SYSTEM = (
    "You are a strict reviewer. Decide whether the CANDIDATE ANSWER addresses the REQUEST and is consistent with "
    'the EVIDENCE. Reply with JSON only: {"valid": true or false, "issues": ["short issue", ...]}. '
    "Use an empty issues list when the answer is fine."
)
_STOPWORDS = {"a", "an", "the", "of", "and", "with", "in", "on", "person", "people", "object"}


class ValidationAgent(BaseAgent):
    name = "validation"
    description = "Validates candidate answers: completeness, consistency with evidence, and failed upstream steps."
    capabilities = ["validation", "consistency_check", "quality_check", "fact_checking"]
    default_model = "fast_text"

    def rule_checks(self, task: AgentTask) -> tuple[list[dict[str, Any]], str, str]:
        upstream = task.upstream()
        candidate = ""
        if upstream:
            text = upstream[0].payload.get("text")
            candidate = text if isinstance(text, str) else ""
        evidence = "\n\n".join(
            str(m.payload.get("text", "")) for m in upstream[1:] if isinstance(m.payload.get("text"), str)
        )
        issues: list[dict[str, Any]] = []

        if not candidate.strip():
            issues.append({"check": "has_output", "severity": 0.6, "message": "Candidate answer is empty"})
        elif len(candidate.strip()) < 10:
            issues.append({"check": "length", "severity": 0.2, "message": "Candidate answer is very short"})
        if max_sentence_repeats(candidate) >= 3:
            issues.append({"check": "repetition", "severity": 0.3, "message": "Answer repeats itself"})
        for failed in task.params.get("failed_steps", []):
            issues.append(
                {
                    "check": "upstream_failure",
                    "severity": 0.15,
                    "message": f"Step '{failed.get('step_id')}' ({failed.get('agent')}) failed: {failed.get('error')}",
                }
            )

        objects = [o for m in upstream for o in m.payload.get("objects", []) if isinstance(o, str)]
        if candidate and len(objects) >= 2:
            words = {w for o in objects for w in re.findall(r"[a-z]{3,}", o.lower()) if w not in _STOPWORDS}
            if words and not any(w in candidate.lower() for w in words):
                issues.append(
                    {"check": "grounding", "severity": 0.1, "message": "Answer does not mention any detected objects"}
                )
        return issues, candidate, evidence

    async def run(self, task: AgentTask) -> dict[str, Any]:
        issues, candidate, evidence = self.rule_checks(task)
        llm_review: dict[str, Any] | None = None

        if candidate.strip():
            user = (
                f"REQUEST:\n{truncate(task.prompt, 2000)}\n\n"
                f"CANDIDATE ANSWER:\n{truncate(candidate, 3000)}\n\n"
                f"EVIDENCE:\n{truncate(evidence, 3000) or '(none)'}"
            )
            try:
                out = await self.generate(task, user, system=REVIEW_SYSTEM, max_tokens=160, temperature=0.0)
                parsed = extract_json(out.text)
                if isinstance(parsed, dict) and "valid" in parsed:
                    llm_review = {
                        "valid": bool(parsed.get("valid")),
                        "issues": [str(i) for i in parsed.get("issues", []) if i][:5],
                    }
                    for msg in llm_review["issues"]:
                        issues.append({"check": "model_review", "severity": 0.1, "message": msg})
                    if not llm_review["valid"]:
                        issues.append({"check": "model_review", "severity": 0.25, "message": "Reviewer judged the answer invalid"})
                else:
                    llm_review = {"unparsed": truncate(out.text, 300)}
            except ModelNotFoundError as exc:
                logger.info("validation model unavailable, rule checks only: %s", exc)

        score = round(max(0.0, 1.0 - sum(i["severity"] for i in issues)), 3)
        valid = score >= 0.6 and not any(i["check"] == "has_output" for i in issues)
        summary = "valid" if valid else "needs attention"
        if issues:
            summary += ": " + "; ".join(i["message"] for i in issues[:4])
        return {
            "valid": valid,
            "score": score,
            "issues": issues,
            "llm_review": llm_review,
            "text": summary,
        }
