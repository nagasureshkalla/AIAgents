"""Combine agent outputs into the final answer, synthesizing with the base model when needed."""

from __future__ import annotations

import logging
from typing import Any

from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentMessage, AgentResult, AgentTask, MessageType
from app.schemas.task import Plan
from app.utils.text import truncate

logger = logging.getLogger(__name__)

SYNTHESIS_SYSTEM = (
    "You are the final response writer of a team of AI specialists. Combine their findings into one coherent, "
    "accurate answer to the user's request. Resolve contradictions in favour of direct evidence, address any "
    "reviewer issues, do not repeat yourself, and do not mention the specialists or the process."
)


class ResultAggregator(BaseAgent):
    """Not routed to by the planner; invoked by the orchestrator after the plan completes.

    It subclasses BaseAgent only to reuse the scheduler-backed `generate` helper.
    """

    name = "aggregator"
    role = "aggregator"
    description = "Synthesizes the outputs of all agents into the final answer."
    capabilities = ["synthesis"]
    default_model = "base"
    prompts = {"synthesize": SYNTHESIS_SYSTEM}

    async def run(self, task: AgentTask) -> dict[str, Any]:  # pragma: no cover - use aggregate()
        raise NotImplementedError("Use aggregate()")

    @staticmethod
    def primary_step(plan: Plan, results: dict[str, AgentResult]) -> str | None:
        """The last completed non-validation step that nothing else (except validation) depends on."""
        content_steps = [s for s in plan.steps if s.capability != "validation"]
        depended = {d for s in content_steps for d in s.depends_on}
        sinks = [s.id for s in content_steps if s.id not in depended and results.get(s.id, None) and results[s.id].ok]
        if sinks:
            return sinks[0]
        completed = [s.id for s in content_steps if s.id in results and results[s.id].ok]
        return completed[-1] if completed else None

    async def aggregate(
        self,
        task: AgentTask,
        plan: Plan,
        results: dict[str, AgentResult],
    ) -> dict[str, Any]:
        primary = self.primary_step(plan, results)
        validation = next(
            (results[s.id].result for s in plan.steps if s.capability == "validation" and s.id in results and results[s.id].ok),
            None,
        )
        outputs: dict[str, dict[str, Any]] = {}
        for step in plan.steps:
            res = results.get(step.id)
            if res is None:
                continue
            outputs[step.id] = {
                "agent": res.agent,
                "status": res.status.value,
                "text": res.text,
                "execution_time": res.execution_time,
                "error": res.error,
            }

        content = [
            (sid, results[sid])
            for sid in [s.id for s in plan.steps if s.capability != "validation"]
            if sid in results and results[sid].ok and results[sid].text.strip()
        ]
        if primary is not None:  # primary first
            content.sort(key=lambda item: item[0] != primary)

        vision = next((r.result for _, r in content if r.agent == "vision"), None)
        base: dict[str, Any] = {
            "intent": plan.intent,
            "primary_step": primary,
            "agents_used": sorted({r.agent for r in results.values() if r.ok}),
            "validation": validation,
            "outputs": outputs,
        }
        if vision:
            base["vision"] = {k: vision.get(k) for k in ("description", "objects", "confidence")}

        if not content:
            base.update(answer="No agent produced an answer.", synthesized=False)
            return base

        issues = validation.get("issues", []) if validation else []
        if len(content) == 1 and not issues:
            base.update(answer=content[0][1].text, synthesized=False)
            return base

        try:
            answer = await self._synthesize(task, content, issues)
            base.update(answer=answer, synthesized=True)
        except Exception as exc:  # noqa: BLE001 - fall back to the best single output
            logger.warning("synthesis failed (%s); returning primary output", exc)
            base.update(answer=content[0][1].text, synthesized=False, synthesis_error=str(exc))
        return base

    async def _synthesize(self, task: AgentTask, content: list[tuple[str, AgentResult]], issues: list[dict]) -> str:
        messages = [
            AgentMessage(
                task_id=task.task_id,
                sender=res.agent,
                receiver=self.name,
                message_type=MessageType.RESULT,
                payload={"step_id": sid, **res.result},
            )
            for sid, res in content
        ]
        synth_task = task.model_copy(update={"messages": messages})
        findings = self.render_upstream(synth_task, limit=7000)
        user = f"User request:\n{task.prompt}\n\nSpecialist findings (most important first):\n{findings}"
        if issues:
            user += "\n\nReviewer issues to address:\n" + "\n".join(f"- {i.get('message')}" for i in issues[:6])
        out = await self.generate(synth_task, truncate(user, 12000), system=self.system_prompt("synthesize"))
        if not out.text.strip():
            raise ValueError("empty synthesis")
        return out.text
