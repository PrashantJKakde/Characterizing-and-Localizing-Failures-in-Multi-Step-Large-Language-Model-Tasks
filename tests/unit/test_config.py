"""Config loading + hashing (M1-M3 dependency, no milestone of its own)."""

from __future__ import annotations

from pathlib import Path

from llm_failure_localization.config import (
    ModelConfig,
    config_hash,
    load_model_config,
    load_stage_sequence,
    load_task_config,
)


def test_load_task_config(tmp_path: Path) -> None:
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "domain: math_reasoning\n"
        "name: gsm8k\n"
        "source: tests/fixtures/gsm8k_sample.jsonl\n"
        "split: test\n"
        "stage_schema: configs/stages/math_reasoning.example.yaml\n"
    )
    config = load_task_config(task_yaml)
    assert config.domain == "math_reasoning"
    assert config.name == "gsm8k"
    assert config.split == "test"


def test_load_model_config(tmp_path: Path) -> None:
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("model_id: echo\nbackend: echo\nmax_new_tokens: 32\nseed: 1\n")
    config = load_model_config(model_yaml)
    assert config.backend == "echo"
    assert config.max_new_tokens == 32
    assert config.checkpoint is None


def test_load_stage_sequence_reads_ordered_list() -> None:
    stages = load_stage_sequence("configs/stages/math_reasoning.example.yaml")
    assert stages == ["understanding", "extraction", "reasoning", "calculation", "synthesis"]


def test_config_hash_is_stable_and_sensitive_to_changes() -> None:
    a = ModelConfig(model_id="m", backend="echo", max_new_tokens=10, seed=0)
    b = ModelConfig(model_id="m", backend="echo", max_new_tokens=10, seed=0)
    c = ModelConfig(model_id="m", backend="echo", max_new_tokens=20, seed=0)

    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)
