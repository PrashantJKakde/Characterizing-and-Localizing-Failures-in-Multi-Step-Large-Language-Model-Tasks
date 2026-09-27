"""Command-line entry point.

`version` and `run` are wired up (M0, M3). `evaluate`, `localize`, and
`compare` are placeholders for M6, M7-M8, and M8 respectively.
"""

from __future__ import annotations

import click

from . import __version__
from .adapters import build_adapter
from .config import config_hash, load_model_config, load_task_config
from .pipeline import PipelineExecutor, write_records
from .runners import load_runner


@click.group()
def main() -> None:
    """stagefail — characterize and localize failures in multi-step LLM tasks."""


@main.command()
def version() -> None:
    """Print the installed package version."""
    click.echo(__version__)


@main.command()
@click.option(
    "--task",
    "task_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a task config YAML (see configs/tasks/).",
)
@click.option(
    "--model",
    "model_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to a model config YAML (see configs/models/).",
)
@click.option(
    "--out",
    "out_path",
    default=None,
    help="Output JSONL path. Defaults to data/<task_domain>__<model_id>.jsonl",
)
@click.option("--limit", default=None, type=int, help="Only run the first N task instances.")
def run(task_path: str, model_path: str, out_path: str | None, limit: int | None) -> None:
    """Execute the pipeline for a configured task + model (Step 4, M3)."""
    task_config = load_task_config(task_path)
    model_config = load_model_config(model_path)

    adapter = build_adapter(task_config)
    runner = load_runner(model_config)
    executor = PipelineExecutor(
        adapter,
        runner,
        model_id=model_config.model_id,
        seed=model_config.seed,
        max_new_tokens=model_config.max_new_tokens,
        config_hash=config_hash(task_config, model_config),
    )

    resolved_out = out_path or f"data/{task_config.domain}__{model_config.model_id}.jsonl"
    count = write_records(executor.run(limit=limit), resolved_out)
    click.echo(f"Wrote {count} stage records to {resolved_out}")


@main.command()
def evaluate() -> None:
    """Run the per-stage correctness protocol over collected data (implemented in M6)."""
    raise NotImplementedError("Stage evaluator lands in Milestone M6.")


@main.command()
def localize() -> None:
    """Localize and classify failures over evaluated trajectories (implemented in M7)."""
    raise NotImplementedError("Failure localizer lands in Milestone M7.")


@main.command()
def compare() -> None:
    """Aggregate cross-model / cross-task comparisons (implemented in M8)."""
    raise NotImplementedError("Aggregator lands in Milestone M8.")


if __name__ == "__main__":
    main()
