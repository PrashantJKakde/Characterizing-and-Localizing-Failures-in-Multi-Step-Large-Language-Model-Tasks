"""PipelineExecutor + write_records: the stage-by-stage run loop and JSONL logger (M3)."""

from __future__ import annotations

import json
from pathlib import Path

from llm_failure_localization.adapters.math_reasoning import GSM8KAdapter
from llm_failure_localization.pipeline.executor import PipelineExecutor, write_records
from llm_failure_localization.runners.echo import EchoRunner
from llm_failure_localization.schema import StageRecord

FIXTURE = Path(__file__).parent.parent / "fixtures" / "gsm8k_sample.jsonl"
STAGES = ["understanding", "extraction", "reasoning", "calculation", "synthesis"]


def _make_executor() -> PipelineExecutor:
    adapter = GSM8KAdapter(source=str(FIXTURE), split="test", stages=STAGES)
    runner = EchoRunner(model_id="echo")
    return PipelineExecutor(
        adapter,
        runner,
        model_id="echo",
        seed=0,
        max_new_tokens=1000,
        config_hash="testhash",
    )


def test_run_yields_one_record_per_instance_per_stage() -> None:
    records = list(_make_executor().run(limit=2))
    assert len(records) == 2 * len(STAGES)
    assert all(isinstance(r, StageRecord) for r in records)
    assert records[0].stage_index == 0
    assert records[0].stage_name == "understanding"
    assert records[0].run_metadata.config_hash == "testhash"


def test_stage_indices_are_sequential_per_instance() -> None:
    records = list(_make_executor().run(limit=1))
    assert [r.stage_index for r in records] == list(range(len(STAGES)))
    assert [r.stage_name for r in records] == STAGES


def test_all_instances_run_without_a_limit() -> None:
    records = list(_make_executor().run())
    assert len(records) == 3 * len(STAGES)


def test_write_records_produces_valid_jsonl(tmp_path: Path) -> None:
    out_path = tmp_path / "out.jsonl"
    records = list(_make_executor().run(limit=1))

    count = write_records(records, out_path)

    assert count == len(records)
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == len(records)

    parsed = json.loads(lines[0])
    assert parsed["task_domain"] == "math_reasoning"
    assert parsed["model_id"] == "echo"
    assert parsed["stage_index"] == 0
