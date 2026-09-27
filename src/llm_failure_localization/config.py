"""Typed configs loaded from YAML, plus a stable hash for `RunMetadata.config_hash`.

`TaskConfig` and `ModelConfig` mirror the example YAML files in `configs/tasks/`
and `configs/models/` one-to-one — a new task or model is a new YAML file, not
new code.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel


class TaskConfig(BaseModel):
    domain: str
    name: str
    source: str
    split: str = "test"
    stage_schema: str


class ModelConfig(BaseModel):
    model_id: str
    backend: str
    checkpoint: str | None = None
    quantization: str | None = None
    max_new_tokens: int = 512
    seed: int = 0


def _load_yaml(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_task_config(path: str | Path) -> TaskConfig:
    return TaskConfig(**_load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig(**_load_yaml(path))


def load_stage_sequence(path: str | Path) -> list[str]:
    """Read a stage-schema YAML file (Step 3) and return its ordered stage names."""
    data = _load_yaml(path)
    stages = data.get("stages")
    if not stages:
        raise ValueError(f"Stage schema {path} has no 'stages' list.")
    return list(stages)


def config_hash(*configs: BaseModel) -> str:
    """A short, stable hash of one or more configs, for `RunMetadata.config_hash`.

    Stable across process runs (unlike `hash()`) so two runs with identical
    task+model configs are recognizably the same experiment.
    """
    payload = json.dumps([c.model_dump() for c in configs], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
