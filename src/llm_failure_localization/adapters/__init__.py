"""Task-domain adapters: load a benchmark and define its stage split (Steps 1 & 3).

New adapters register themselves in `ADAPTER_REGISTRY`; nothing else in the
pipeline needs to change to add one (M1).
"""

from __future__ import annotations

from ..config import TaskConfig, load_stage_sequence
from .base import TaskAdapter, TaskInstance
from .math_reasoning import GSM8KAdapter

ADAPTER_REGISTRY: dict[str, type[TaskAdapter]] = {
    "gsm8k": GSM8KAdapter,
}


def build_adapter(task_config: TaskConfig) -> TaskAdapter:
    """Instantiate the adapter named in `task_config.name` from a `TaskConfig`.

    Every registered adapter is constructed the same way — `source`, `split`,
    and a `stages` list read from `task_config.stage_schema` — which is what
    lets a new task domain be a new YAML file plus one adapter class.
    """
    try:
        adapter_cls = ADAPTER_REGISTRY[task_config.name]
    except KeyError as exc:
        raise ValueError(
            f"No adapter registered for task '{task_config.name}'; "
            f"known tasks are {sorted(ADAPTER_REGISTRY)}."
        ) from exc
    stages = load_stage_sequence(task_config.stage_schema)
    return adapter_cls(source=task_config.source, split=task_config.split, stages=stages)


__all__ = ["ADAPTER_REGISTRY", "TaskAdapter", "TaskInstance", "build_adapter"]
