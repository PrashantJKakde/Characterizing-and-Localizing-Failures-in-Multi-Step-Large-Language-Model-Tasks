"""Command-line entry point.

Only `version` is wired up at M0. `run`, `evaluate`, `localize`, and `compare`
are placeholders for M3, M6, M7-M8, and M8 respectively.
"""

from __future__ import annotations

import click

from . import __version__


@click.group()
def main() -> None:
    """stagefail — characterize and localize failures in multi-step LLM tasks."""


@main.command()
def version() -> None:
    """Print the installed package version."""
    click.echo(__version__)


@main.command()
def run() -> None:
    """Execute the pipeline for a configured task + model (implemented in M3)."""
    raise NotImplementedError("Pipeline executor lands in Milestone M3.")


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
