"""Planner agent: turns the manager's decision into an executable dependency graph of steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.agents.base_agent import BaseAgent
from app.core.exceptions import InvalidRequestError
from app.schemas.agent import AgentTask
from app.schemas.task import Plan, PlanStep


@dataclass(frozen=True)
class StepTemplate:
    id: str
    capability: str
    task: str
    deps: tuple[str, ...] = ()
    optional: bool = False
    mode: str | None = None


V = "validation"
# Steps without a dependency between them run concurrently. Validation lists the candidate answer first.
PLAN_TEMPLATES: dict[str, list[StepTemplate]] = {
    "image_analysis": [
        StepTemplate("see", "image_analysis", "Analyze the image: describe the scene and list visible objects", mode="describe"),
        StepTemplate("intent", "request_interpretation", "Interpret what the user wants to know", mode="interpret_request", optional=True),
        StepTemplate("reason", "reasoning", "Interpret the visual information to answer the request", ("see", "intent"), mode="interpret_visual"),
        StepTemplate("validate", V, "Check the answer against the visual evidence", ("reason", "see"), optional=True),
    ],
    "visual_qa": [
        StepTemplate("answer", "visual_question_answering", "Answer the question about the image", mode="vqa"),
        StepTemplate("see", "image_description", "Describe the image for supporting evidence", optional=True, mode="describe"),
        StepTemplate("reason", "reasoning", "Reconcile the visual answer with the scene description", ("answer", "see"), mode="interpret_visual"),
        StepTemplate("validate", V, "Check the answer against the visual evidence", ("reason", "answer", "see"), optional=True),
    ],
    "screenshot_debug": [
        StepTemplate("inspect", "screenshot_analysis", "Inspect the screenshot for errors or problems", mode="screenshot"),
        StepTemplate("see", "image_description", "Describe the screenshot contents", optional=True, mode="describe"),
        StepTemplate("reason", "reasoning", "Diagnose what is wrong and suggest fixes", ("inspect", "see"), mode="diagnose"),
        StepTemplate("validate", V, "Check the diagnosis against the screenshot evidence", ("reason", "inspect"), optional=True),
    ],
    "summarize": [
        StepTemplate("summary", "summarization", "Summarize the text", mode="summarize"),
        StepTemplate("analysis", "reasoning", "Identify the key claims and their logical structure", optional=True, mode="key_points"),
        StepTemplate("validate", V, "Check the summary is faithful and complete", ("summary", "analysis"), optional=True),
    ],
    "extract": [
        StepTemplate("extract", "extraction", "Extract the requested information", mode="extract"),
        StepTemplate("validate", V, "Check the extraction is complete and well-formed", ("extract",), optional=True),
    ],
    "classify": [
        StepTemplate("classify", "classification", "Classify the input", mode="classify"),
        StepTemplate("rationale", "reasoning", "Reason about which category fits best", optional=True, mode="rationale"),
        StepTemplate("validate", V, "Check the label is consistent with the reasoning", ("classify", "rationale"), optional=True),
    ],
    "format": [
        StepTemplate("format", "formatting", "Reformat the content as requested", mode="format"),
        StepTemplate("validate", V, "Check no information was lost", ("format",), optional=True),
    ],
    "explain": [
        StepTemplate("reason", "reasoning", "Analyze the question step by step", mode="explain"),
        StepTemplate("write", "text_generation", "Write a clear final explanation from the analysis", ("reason",), mode="compose"),
        StepTemplate("validate", V, "Check the explanation is correct and consistent", ("write", "reason"), optional=True),
    ],
    "generate": [
        StepTemplate("answer", "text_generation", "Answer the user's request", mode="generate"),
        StepTemplate("validate", V, "Check the answer addresses the request", ("answer",), optional=True),
    ],
}


class PlannerAgent(BaseAgent):
    name = "planner"
    role = "control"
    description = "Converts the manager's decision into executable steps and identifies parallelizable work."
    capabilities = ["task_planning", "dependency_analysis", "parallelization"]
    default_model = None

    async def run(self, task: AgentTask) -> dict[str, Any]:
        decisions = task.upstream_by_agent("manager")
        if not decisions:
            raise InvalidRequestError("Planner requires the manager's decision")
        decision = decisions[0]
        steps = self.build_steps(decision, skip_validation=bool(task.params.get("skip_validation")))
        plan = Plan(task_id=task.task_id, intent=decision.get("intent", "generate"), steps=steps)
        plan.parallel_groups = plan.compute_parallel_groups()
        flow = " -> ".join(" || ".join(group) for group in plan.parallel_groups)
        return {"plan": plan.model_dump(), "text": f"{plan.intent}: {flow}"}

    def build_steps(self, decision: dict[str, Any], skip_validation: bool = False) -> list[PlanStep]:
        registry = self.ctx.registry
        assert registry is not None

        forced = decision.get("forced_agents") or []
        if forced:
            return [
                PlanStep(id=name, agent=name, task=f"Handle the request with the {name} agent", params={"mode": "generate"})
                for name in forced
            ]

        template = PLAN_TEMPLATES.get(decision.get("intent", ""), PLAN_TEMPLATES["generate"])
        steps: list[PlanStep] = []
        dropped: set[str] = set()
        for t in template:
            if t.capability == V and skip_validation:
                dropped.add(t.id)
                continue
            agent = registry.first_for(t.capability)
            if agent is None:
                if t.optional:
                    dropped.add(t.id)
                    continue
                raise InvalidRequestError(
                    f"No agent provides the required capability '{t.capability}' for intent '{decision.get('intent')}'"
                )
            steps.append(
                PlanStep(
                    id=t.id,
                    agent=agent.name,
                    task=t.task,
                    capability=t.capability,
                    depends_on=[d for d in t.deps if d not in dropped],
                    optional=t.optional,
                    params={"mode": t.mode} if t.mode else {},
                )
            )

        # Drop-in agents selected by keyword run in parallel with the first wave; validation also reviews them.
        used = {s.agent for s in steps}
        for name in decision.get("extra_agents", []):
            if name in used or not registry.has(name):
                continue
            step = PlanStep(
                id=f"{name}_extra",
                agent=name,
                task=f"Contribute {name} expertise to the request",
                capability=(registry.get(name).capabilities or [None])[0],
                optional=True,
            )
            validation = next((s for s in steps if s.capability == V), None)
            if validation is not None:
                validation.depends_on.append(step.id)
                steps.insert(steps.index(validation), step)
            else:
                steps.append(step)
        return steps
