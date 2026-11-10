"""AirLLM agent: routing by name, freeing the GPU, the first-load wait and streamed generation (fake service)."""

from __future__ import annotations

from typing import Any

import pytest

from app.agents.airllm_agent import AirLLMAgent, AirLLMClient, AirLLMError, question_of
from app.agents.manager_agent import ManagerAgent
from app.schemas.common import TaskStatus
from app.schemas.task import TaskRequest
from tests.fakes import CUDA_DEVICE, FakeFactory, build_container, make_settings


@pytest.mark.parametrize("prompt", ["ask gpt-oss-120b: why is the sky blue?", "use airllm to write a haiku",
                                    "Ask the 120B model what 2+2 is", "let the big model explain RAG"])
def test_routed_only_when_named(prompt):
    assert ManagerAgent.classify_intent(prompt, False) == "big_model"


@pytest.mark.parametrize("prompt", ["explain recursion", "write a haiku", "summarize: a big model is slow",
                                    "send a cc to the team"])
def test_other_requests_are_not(prompt):
    assert ManagerAgent.classify_intent(prompt, False) != "big_model"


def test_routing_words_are_removed_from_the_question():
    assert question_of("ask gpt-oss-120b: why is the sky blue?") == "why is the sky blue?"
    assert question_of("Using the big model, explain RAG") == "explain RAG"
    assert question_of("gpt-oss") == "gpt-oss"


def test_service_must_be_local(tmp_path):
    s = make_settings(tmp_path)
    with pytest.raises(AirLLMError, match="this computer"):
        AirLLMClient(s.airllm.model_copy(update={"url": "http://8.8.8.8:8770"}), s)


class FakeService:
    calls: list[Any] = []
    polls = 0

    def __init__(self, cfg, settings) -> None:
        pass

    async def ensure_running(self) -> dict[str, Any]:
        FakeService.calls.append("start")
        return {"model": "gpt-oss-120b", "installed": True, "loaded": False, "layers_split": False}

    async def load(self) -> dict[str, Any]:
        FakeService.calls.append("load")
        return {"model": "gpt-oss-120b", "loaded": False, "loading": True}

    async def health(self) -> dict[str, Any]:
        FakeService.polls += 1
        return {"model": "gpt-oss-120b", "loaded": FakeService.polls >= 2, "loading": FakeService.polls < 2}

    async def generate(self, messages, max_new_tokens, reasoning):
        FakeService.calls.append(("generate", messages, reasoning))
        for t in ["The", " sky", " scatters", " blue", " light."]:
            yield {"type": "token", "text": t, "channel": "final"}
        yield {"type": "done", "answer": "The sky scatters blue light.", "reasoning": "Rayleigh.", "tokens": 5,
               "seconds": 10.0}


@pytest.fixture
async def container(tmp_path, monkeypatch):
    monkeypatch.setattr(AirLLMAgent, "client_factory", FakeService)
    monkeypatch.setattr("app.agents.airllm_agent.asyncio.sleep", _no_sleep)
    FakeService.calls, FakeService.polls = [], 0
    settings = make_settings(tmp_path)
    settings.agents["airllm"] = settings.agent_config("airllm").model_copy(update={"timeout_seconds": 30})
    c = build_container(settings, factory=FakeFactory(), device=CUDA_DEVICE)
    await c.start()
    try:
        yield c
    finally:
        await c.stop()


async def _no_sleep(seconds):
    return None


async def test_answers_with_the_big_model(container):
    await container.models.load_model("fast_text")  # idle model: freed before the layers stream through the GPU
    record = await container.run_and_wait(TaskRequest(prompt="ask gpt-oss-120b: why is the sky blue?"), timeout=20)
    assert record.status == TaskStatus.COMPLETED, record.error
    assert [s.agent for s in record.plan.steps] == ["airllm"]
    assert not container.models.is_loaded("fast_text")
    assert FakeService.calls[:2] == ["start", "load"]
    gen = next(c for c in FakeService.calls if isinstance(c, tuple))
    assert gen[1] == [{"role": "user", "content": "why is the sky blue?"}] and gen[2] == "low"
    answer = record.result["answer"]
    assert answer.startswith("The sky scatters blue light.") and "5 tokens in 10s" in answer
    progress = [e.data.get("message") for e in container.events.history(record.task_id) if e.event.value == "agent_progress"]
    assert any("first run: splitting" in (m or "") for m in progress)


async def test_uses_ollama_when_gpt_oss_is_pulled(tmp_path, monkeypatch):
    from app.core.config import Settings
    from app.models.engines import ollama_engine

    monkeypatch.setattr(ollama_engine, "ollama_installed", lambda url, tag: True)
    monkeypatch.setattr(AirLLMAgent, "client_factory", FakeService)
    FakeService.calls = []
    data = make_settings(tmp_path).model_dump()
    data["models"]["gpt_oss"] = {"engine": "ollama", "ollama_model": "gpt-oss:120b", "type": "text", "est_size_gb": 22,
                                 "max_tokens": 2048, "model_kwargs": {"num_ctx": 8192, "think": "low"}}
    data["agents"]["airllm"] = {"model": "gpt_oss", "timeout_seconds": 30}
    c = build_container(Settings.model_validate(data), factory=FakeFactory(), device=CUDA_DEVICE)
    await c.start()
    try:
        record = await c.run_and_wait(TaskRequest(prompt="ask gpt-oss-120b: why is the sky blue?"), timeout=20)
    finally:
        await c.stop()
    assert record.status == TaskStatus.COMPLETED, record.error
    assert record.decision["models"]["required"] == ["gpt_oss"]  # the ModelManager loads it like any model
    result = record.agent_results["bigmodel"].result
    assert result["engine"] == "ollama" and "via Ollama" in result["text"]
    assert not FakeService.calls  # the AirLLM service is not used
