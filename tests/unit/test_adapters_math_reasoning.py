"""GSM8KAdapter: benchmark loading, stage sequence, and per-stage context (M1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_failure_localization.adapters.math_reasoning import GSM8KAdapter

FIXTURE = Path(__file__).parent.parent / "fixtures" / "gsm8k_sample.jsonl"
STAGES = ["understanding", "extraction", "reasoning", "calculation", "synthesis"]


def _make_adapter(stages: list[str] = STAGES) -> GSM8KAdapter:
    return GSM8KAdapter(source=str(FIXTURE), split="test", stages=stages)


def test_stage_sequence_matches_config() -> None:
    assert _make_adapter().stage_sequence() == STAGES


def test_load_instances_from_local_fixture() -> None:
    instances = list(_make_adapter().load_instances())
    assert len(instances) == 3

    first = instances[0]
    assert first.task_id == "fixture-0"
    assert "Maria" in first.raw_input["question"]
    assert first.expected_answer == "31"


def test_context_for_first_stage_has_no_prior_outputs() -> None:
    adapter = _make_adapter()
    instance = next(iter(adapter.load_instances()))
    context = adapter.context_for_stage(instance, 0, [])
    assert context["stage_name"] == "understanding"
    assert "Maria" in context["prompt"]


def test_context_for_later_stage_includes_prior_outputs() -> None:
    adapter = _make_adapter()
    instance = next(iter(adapter.load_instances()))
    prior_outputs = ["Maria has 36 pencils and gives away 5."]
    context = adapter.context_for_stage(instance, 1, prior_outputs)
    assert context["stage_name"] == "extraction"
    assert "Maria has 36 pencils and gives away 5." in context["prompt"]


def test_unknown_stage_name_raises_at_construction() -> None:
    with pytest.raises(ValueError, match="not_a_real_stage"):
        _make_adapter(stages=["not_a_real_stage"])
