from __future__ import annotations

from typing import Any

from app.agents.base_agent import BaseAgent
from app.agents.manager_agent import ManagerAgent
from app.schemas.agent import AgentStatus, AgentTask
from app.schemas.common import InputType
from app.schemas.message import AgentMessage, MessageType
from tests.fakes import make_image


def task_for(agent: str, prompt: str, *, image: str | None = None, mode: str | None = None, messages=None) -> AgentTask:
    return AgentTask(
        task_id="task_test",
        step_id=agent,
        agent=agent,
        instruction="test",
        prompt=prompt,
        input_type=InputType.IMAGE if image else InputType.TEXT,
        image_path=image,
        params={"mode": mode} if mode else {},
        messages=messages or [],
    )


def result_msg(sender: str, payload: dict[str, Any], receiver: str = "x") -> AgentMessage:
    return AgentMessage(
        task_id="task_test", sender=sender, receiver=receiver, message_type=MessageType.RESULT, payload=payload
    )


# ---------------------------------------------------------------- manager
def test_manager_intent_rules():
    assert ManagerAgent.classify_intent("Summarize this document.", has_image=False) == "summarize"
    assert ManagerAgent.classify_intent("Explain why the sky is blue", has_image=False) == "explain"
    assert ManagerAgent.classify_intent("Write a poem", has_image=False) == "generate"
    assert (
        ManagerAgent.classify_intent("Analyze this image and explain what is happening.", has_image=True)
        == "image_analysis"
    )
    assert ManagerAgent.classify_intent("What is wrong with this screenshot?", has_image=True) == "screenshot_debug"
    assert ManagerAgent.classify_intent("How many dogs are there?", has_image=True) == "visual_qa"


async def test_manager_selects_agents_for_text(container):
    res = await container.agents.get("manager").execute(task_for("manager", "Summarize this document: ..."))
    assert res.status == AgentStatus.COMPLETED
    assert res.result["intent"] == "summarize"
    assert res.result["agents"] == ["text", "reasoning", "validation"]


async def test_manager_selects_agents_for_image(container, tmp_path):
    img = make_image(container.settings.input_dir)
    res = await container.agents.get("manager").execute(
        task_for("manager", "Analyze this image and explain what is happening.", image=str(img))
    )
    assert res.result["intent"] == "image_analysis"
    assert res.result["agents"] == ["vision", "text", "reasoning", "validation"]
    assert res.result["input_type"] == "image"


async def test_manager_routes_screenshot_to_multimodal(container):
    img = make_image(container.settings.input_dir)
    res = await container.agents.get("manager").execute(
        task_for("manager", "What is wrong with this screenshot?", image=str(img))
    )
    assert res.result["intent"] == "screenshot_debug"
    assert res.result["agents"][0] == "multimodal"
    assert res.result["input_type"] == "multimodal"


# ---------------------------------------------------------------- planner
async def test_planner_builds_parallel_groups(container):
    planner = container.agents.get("planner")
    decision = {"intent": "image_analysis", "agents": [], "extra_agents": []}
    res = await planner.execute(task_for("planner", "x", messages=[result_msg("manager", decision)]))
    assert res.ok, res.error
    plan = res.result["plan"]
    assert plan["parallel_groups"] == [["intent", "see"], ["reason"], ["validate"]]
    by_id = {s["id"]: s for s in plan["steps"]}
    assert by_id["see"]["agent"] == "vision"
    assert by_id["reason"]["depends_on"] == ["see", "intent"]
    assert by_id["validate"]["optional"] is True


async def test_planner_skip_validation(container):
    steps = container.agents.get("planner").build_steps({"intent": "generate"}, skip_validation=True)
    assert [s.id for s in steps] == ["answer"]


async def test_planner_requires_manager_decision(container):
    res = await container.agents.get("planner").execute(task_for("planner", "x"))
    assert res.status == AgentStatus.FAILED
    assert res.error_type == "invalid_input"


# ---------------------------------------------------------------- workers
async def test_text_agent_standard_result(container):
    res = await container.agents.get("text").execute(task_for("text", "Summarize: a b c", mode="summarize"))
    dumped = res.model_dump()
    for key in ("agent", "status", "result", "execution_time", "error"):
        assert key in dumped
    assert res.agent == "text"
    assert res.status == AgentStatus.COMPLETED
    assert res.result["text"].startswith("Answer:")
    assert res.models == ["fast_text"]


async def test_vision_agent_describes_image(container):
    img = make_image(container.settings.input_dir)
    res = await container.agents.get("vision").execute(task_for("vision", "describe", image=str(img), mode="describe"))
    assert res.ok, res.error
    assert res.result["objects"] == ["cat", "sofa", "lamp"]
    assert "cat" in res.result["description"]
    assert 0.0 <= res.result["confidence"] <= 1.0


async def test_vision_agent_invalid_image(container):
    bad = container.settings.input_dir / "bad.png"
    bad.write_bytes(b"not an image")
    res = await container.agents.get("vision").execute(task_for("vision", "describe", image=str(bad)))
    assert res.status == AgentStatus.FAILED
    assert res.error_type == "invalid_input"


async def test_reasoning_agent_uses_upstream_evidence(container):
    evidence = result_msg("vision", {"step_id": "see", "text": "A cat on a sofa", "objects": ["cat", "sofa"]})
    res = await container.agents.get("reasoning").execute(
        task_for("reasoning", "What is happening?", mode="interpret_visual", messages=[evidence])
    )
    assert res.ok
    assert res.result["conclusion"] == "A cat is resting on a sofa."
    assert res.models == ["base"]


async def test_validation_agent_accepts_good_answer(container):
    candidate = result_msg("reasoning", {"step_id": "reason", "text": "A cat is resting on a sofa near a lamp."})
    evidence = result_msg("vision", {"step_id": "see", "text": "cat on sofa", "objects": ["cat", "sofa"]})
    res = await container.agents.get("validation").execute(
        task_for("validation", "What is happening?", messages=[candidate, evidence])
    )
    assert res.ok
    assert res.result["valid"] is True
    assert res.result["score"] == 1.0


async def test_validation_agent_flags_empty_answer(container):
    res = await container.agents.get("validation").execute(
        task_for("validation", "What is happening?", messages=[result_msg("text", {"text": ""})])
    )
    assert res.result["valid"] is False
    assert any(i["check"] == "has_output" for i in res.result["issues"])


async def test_missing_model_is_reported_not_raised(tmp_path):
    from tests.fakes import build_container, make_settings

    c = build_container(make_settings(tmp_path, install=("fast_text",)))
    img = make_image(c.settings.input_dir)
    res = await c.agents.get("vision").execute(task_for("vision", "describe", image=str(img)))
    assert res.status == AgentStatus.FAILED
    assert res.error_type == "missing_model"


# ---------------------------------------------------------------- extensibility
class CodeAgent(BaseAgent):
    name = "code"
    description = "Writes code"
    capabilities = ["code_generation"]
    keywords = ["python", "code"]
    default_model = "fast_text"

    async def run(self, task: AgentTask) -> dict[str, Any]:
        out = await self.generate(task, task.prompt)
        return {"text": out.text, "language": "python"}


async def test_new_agent_is_routed_by_keywords(container):
    container.agents.register(CodeAgent)
    assert "code_generation" in container.agents.all_capabilities()

    decision = await container.agents.get("manager").execute(task_for("manager", "Write python code to sort a list"))
    assert "code" in decision.result["extra_agents"]

    steps = container.agents.get("planner").build_steps(decision.result)
    code_step = next(s for s in steps if s.agent == "code")
    validate = next(s for s in steps if s.capability == "validation")
    assert code_step.id in validate.depends_on
