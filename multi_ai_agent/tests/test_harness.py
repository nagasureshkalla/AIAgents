from __future__ import annotations

import os

from app.agents.text_agent import SYSTEM_PROMPTS
from app.core.harness import HarnessStore, load_harness, seed_tree, write_tree
from tests.test_agents import task_for


def test_seeded_harness_reproduces_builtin_prompts(tmp_path):
    write_tree(seed_tree("text", SYSTEM_PROMPTS), tmp_path / "text")
    harness = load_harness(tmp_path / "text")
    assert harness.guardrails == ""  # skill trigger lines are harness2 bookkeeping, not prompt text
    for mode, prompt in SYSTEM_PROMPTS.items():
        assert harness.compose(prompt, mode) == prompt


def test_missing_harness_is_builtin_only(tmp_path):
    store = HarnessStore(tmp_path)
    assert store.get("text").empty
    assert store.get("text").compose("Built in.", "summarize") == "Built in."


def test_skill_replaces_mode_prompt_and_always_on_text_is_appended(tmp_path):
    d = tmp_path / "text"
    tree = seed_tree("text", {"extract": "Old extract prompt."})
    tree["skills/extract/SKILL.md"] = "---\nname: extract\ndescription: x\n---\n\nReturn JSON only."
    tree["guardrails.md"] += "- Never guess a date that the text does not state.\n"
    tree["LongTermMEMORY.md"] = "- 'net 30 from X' means due 30 days after X.\n"
    write_tree(tree, d)
    out = load_harness(d).compose("Old extract prompt.", "extract")
    assert out.startswith("Return JSON only.")
    assert "Old extract prompt." not in out
    assert "Rules:\n- Never guess a date" in out
    assert "load the `extract` skill" not in out
    assert out.endswith("Lessons from earlier tasks:\n- 'net 30 from X' means due 30 days after X.")


def test_store_reloads_changed_files(tmp_path):
    d = tmp_path / "text"
    write_tree({"systemprompt.md": "First."}, d)
    store = HarnessStore(tmp_path)
    assert store.get("text").system == "First."
    p = d / "systemprompt.md"
    p.write_text("Second.")
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert store.get("text").system == "Second."


def test_override_dir_wins_over_root(tmp_path):
    write_tree({"systemprompt.md": "Root."}, tmp_path / "root" / "text")
    write_tree({"systemprompt.md": "Candidate."}, tmp_path / "cand")
    store = HarnessStore(tmp_path / "root", {"text": tmp_path / "cand"})
    assert store.get("text").system == "Candidate."


async def test_agents_use_their_harness(container):
    root = container.settings.resolve_path(container.settings.harness.root)
    write_tree({"systemprompt.md": "Always cite the ticket id."}, root / "text")
    agent = container.agents.get("text")
    assert agent.system_prompt("classify") == SYSTEM_PROMPTS["classify"] + "\n\nAlways cite the ticket id."
    assert container.agents.get("reasoning").system_prompt("explain") == container.agents.get("reasoning").prompts["explain"]
    result = await agent.execute(task_for("text", "Classify: the export is broken", mode="classify"))
    assert result.ok
