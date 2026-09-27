"""Math word-problem adapter (domain: `math_reasoning`), e.g. GSM8K.

Implements the three `TaskAdapter` methods (M1): load benchmark instances,
define the stage sequence, and build each stage's prompt/context from the
instance and everything produced at earlier stages.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from .base import TaskAdapter, TaskInstance

_ANSWER_MARKER = "####"

# One prompt template per stage name. A task config's stage schema (Step 3)
# must only name stages this adapter knows how to prompt for.
_STAGE_PROMPTS: dict[str, str] = {
    "understanding": (
        "Restate the following math word problem in your own words, making sure "
        "you have not lost or added any detail.\n\nProblem:\n{question}"
    ),
    "extraction": (
        "From the problem below, list every quantity given and every quantity "
        "that must be found. Do not solve the problem yet.\n\n"
        "Problem:\n{question}\n\nRestatement:\n{understanding}"
    ),
    "reasoning": (
        "Given the problem and the extracted quantities below, describe the "
        "sequence of arithmetic operations needed to solve it. Do not compute "
        "the final numbers yet.\n\nProblem:\n{question}\n\nExtracted quantities:\n{extraction}"
    ),
    "calculation": (
        "Carry out the arithmetic operations described below and show your work.\n\n"
        "Problem:\n{question}\n\nPlanned operations:\n{reasoning}"
    ),
    "synthesis": (
        "State the final numeric answer to the problem on its own, with no other "
        "text.\n\nProblem:\n{question}\n\nWorked calculation:\n{calculation}"
    ),
}


def _parse_final_answer(answer_field: str) -> str:
    """GSM8K's `answer` field is a chain of reasoning ending in `#### <number>`."""
    if _ANSWER_MARKER in answer_field:
        return answer_field.split(_ANSWER_MARKER)[-1].strip()
    return answer_field.strip()


class GSM8KAdapter(TaskAdapter):
    """GSM8K-style grade-school math word problems.

    `source` may be a local JSONL file — one `{"question": ..., "answer": ...}`
    object per line, GSM8K's native shape — for offline runs and tests, or a
    Hugging Face dataset id (requires the optional `datasets` package) for the
    real benchmark.
    """

    domain = "math_reasoning"

    def __init__(self, source: str, split: str, stages: list[str]) -> None:
        self.source = source
        self.split = split
        self._stages = stages
        missing = [s for s in self._stages if s not in _STAGE_PROMPTS]
        if missing:
            raise ValueError(
                f"GSM8KAdapter has no prompt template for stage(s) {missing}; "
                f"known stages are {sorted(_STAGE_PROMPTS)}."
            )

    def stage_sequence(self) -> list[str]:
        return list(self._stages)

    def load_instances(self) -> Iterator[TaskInstance]:
        local_path = Path(self.source)
        if local_path.is_file():
            yield from self._load_local(local_path)
        else:
            yield from self._load_hf_dataset()

    def _load_local(self, path: Path) -> Iterator[TaskInstance]:
        with path.open(encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield TaskInstance(
                    task_id=row.get("task_id", f"{path.stem}-{i}"),
                    raw_input={"question": row["question"]},
                    expected_answer=_parse_final_answer(row["answer"]),
                )

    def _load_hf_dataset(self) -> Iterator[TaskInstance]:
        try:
            import datasets
        except ImportError as exc:
            raise ImportError(
                f"Loading '{self.source}' requires the optional `datasets` package "
                "(pip install -e '.[inference]'), or pass a local JSONL file instead."
            ) from exc
        ds = datasets.load_dataset(self.source, split=self.split)
        for i, row in enumerate(ds):
            yield TaskInstance(
                task_id=f"{self.source.replace('/', '_')}-{self.split}-{i}",
                raw_input={"question": row["question"]},
                expected_answer=_parse_final_answer(row["answer"]),
            )

    def context_for_stage(
        self, instance: TaskInstance, stage_index: int, prior_outputs: list[Any]
    ) -> dict[str, Any]:
        stage_name = self._stages[stage_index]
        template_vars: dict[str, Any] = {"question": instance.raw_input["question"]}
        for name, output in zip(self._stages[:stage_index], prior_outputs, strict=True):
            template_vars[name] = output
        prompt = _STAGE_PROMPTS[stage_name].format(**template_vars)
        return {"prompt": prompt, "stage_name": stage_name}
