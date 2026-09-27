"""`stagefail run` end-to-end through Click's test runner (M3), echo backend only."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from llm_failure_localization.cli import main

FIXTURE = Path(__file__).parent.parent / "fixtures" / "gsm8k_sample.jsonl"


def test_run_command_writes_one_line_per_stage(tmp_path: Path) -> None:
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text(
        "domain: math_reasoning\n"
        "name: gsm8k\n"
        f"source: {FIXTURE}\n"
        "split: test\n"
        "stage_schema: configs/stages/math_reasoning.example.yaml\n"
    )
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("model_id: echo\nbackend: echo\nmax_new_tokens: 50\nseed: 0\n")
    out_path = tmp_path / "out.jsonl"

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "run",
            "--task",
            str(task_yaml),
            "--model",
            str(model_yaml),
            "--out",
            str(out_path),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 5  # one record per stage, for the one instance we asked for


def test_run_command_defaults_out_path_under_data_dir(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    task_yaml = tmp_path / "task.yaml"

    # Point the stage schema at the repo's real config via an absolute path,
    # since we've chdir'd away from the repo root.
    repo_root = Path(__file__).parent.parent.parent
    stage_schema_path = repo_root / "configs" / "stages" / "math_reasoning.example.yaml"
    task_yaml.write_text(
        "domain: math_reasoning\n"
        "name: gsm8k\n"
        f"source: {FIXTURE}\n"
        "split: test\n"
        f"stage_schema: {stage_schema_path}\n"
    )
    model_yaml = tmp_path / "model.yaml"
    model_yaml.write_text("model_id: echo\nbackend: echo\nmax_new_tokens: 50\nseed: 0\n")

    runner = CliRunner()
    result = runner.invoke(
        main,
        ["run", "--task", str(task_yaml), "--model", str(model_yaml), "--limit", "1"],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "data" / "math_reasoning__echo.jsonl").exists()
