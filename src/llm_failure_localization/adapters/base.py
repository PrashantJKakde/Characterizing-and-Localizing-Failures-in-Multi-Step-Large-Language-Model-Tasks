"""Abstract task adapter.

A new task domain plugs in by subclassing `TaskAdapter` and implementing these
three methods — nothing else in the pipeline needs to change (M1).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class TaskInstance:
    """One benchmark item: its id, raw input, and the expected final answer."""

    def __init__(self, task_id: str, raw_input: Any, expected_answer: Any) -> None:
        self.task_id = task_id
        self.raw_input = raw_input
        self.expected_answer = expected_answer


class TaskAdapter(ABC):
    """One adapter per task domain (e.g. math_reasoning, multihop_qa, rag, agentic_tooluse)."""

    domain: str

    @abstractmethod
    def load_instances(self) -> Iterator[TaskInstance]:
        """Yield the benchmark's task instances."""

    @abstractmethod
    def stage_sequence(self) -> list[str]:
        """The fixed, ordered list of stage names for this task domain (Step 3)."""

    @abstractmethod
    def context_for_stage(
        self, instance: TaskInstance, stage_index: int, prior_outputs: list[Any]
    ) -> dict[str, Any]:
        """Build the prompt/context a model needs to produce this stage's output,
        given the instance and everything produced at earlier stages.
        """
