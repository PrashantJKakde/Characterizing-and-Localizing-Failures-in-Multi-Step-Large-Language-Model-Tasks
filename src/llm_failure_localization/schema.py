"""The one record shape every module in the pipeline reads and writes.

Keeping a single normalized schema means a new task domain, model, or analysis
script never needs a special case: everything downstream of data collection
(evaluation, localization, classification, propagation, comparison) consumes
this shape and nothing else.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class Correctness(str, Enum):
    """Step 5 of the methodology: the four-way per-stage correctness label."""

    CORRECT = "correct"
    INCORRECT = "incorrect"
    PARTIAL = "partial"
    UNSUPPORTED = "unsupported"


class RunMetadata(BaseModel):
    seed: int
    config_hash: str
    timestamp: str
    package_version: str


class StageRecord(BaseModel):
    """One stage's output for one task instance, run on one model.

    One JSON object per line (JSONL); one file per (task_domain, model_id) pair.
    """

    task_id: str
    task_domain: str
    model_id: str

    stage_index: int
    stage_name: str
    stage_output: Any
    context: dict[str, Any] = Field(default_factory=dict)

    # Filled in by evaluation.stage_evaluator (Step 5)
    correctness: Correctness | None = None

    # Filled in by localization.localizer (Step 6)
    is_root_failure: bool | None = None

    # Filled in by localization.taxonomy (Step 7)
    failure_category: str | None = None

    # Filled in by localization.dependency / .propagation (Steps 8-9)
    propagation_flag: bool | None = None

    run_metadata: RunMetadata
