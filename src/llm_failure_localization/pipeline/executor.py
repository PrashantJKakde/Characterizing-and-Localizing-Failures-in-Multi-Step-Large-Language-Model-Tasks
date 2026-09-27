"""Executor + stage logger: runs one task instance stage-by-stage and writes
stage-indexed JSONL records (Step 4, M3).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from pathlib import Path

from .. import __version__
from ..adapters.base import TaskAdapter
from ..runners.base import ModelRunner
from ..schema import RunMetadata, StageRecord


class PipelineExecutor:
    """Runs every instance an adapter yields through every one of its stages."""

    def __init__(
        self,
        adapter: TaskAdapter,
        runner: ModelRunner,
        *,
        model_id: str,
        seed: int,
        max_new_tokens: int,
        config_hash: str,
    ) -> None:
        self.adapter = adapter
        self.runner = runner
        self.model_id = model_id
        self.seed = seed
        self.max_new_tokens = max_new_tokens
        self.config_hash = config_hash

    def run(self, limit: int | None = None) -> Iterator[StageRecord]:
        """Yield one StageRecord per (task instance, stage) pair, in order.

        Every record from one call shares a `run_metadata.timestamp`, so
        records from the same invocation of `stagefail run` are identifiable
        as one run even after the JSONL file has been split or re-aggregated.
        """
        stages = self.adapter.stage_sequence()
        run_metadata = RunMetadata(
            seed=self.seed,
            config_hash=self.config_hash,
            timestamp=datetime.now(timezone.utc).isoformat(),
            package_version=__version__,
        )
        for count, instance in enumerate(self.adapter.load_instances()):
            if limit is not None and count >= limit:
                break
            prior_outputs: list[str] = []
            for stage_index, stage_name in enumerate(stages):
                context = self.adapter.context_for_stage(instance, stage_index, prior_outputs)
                stage_output = self.runner.generate(
                    context["prompt"], max_new_tokens=self.max_new_tokens, seed=self.seed
                )
                prior_outputs.append(stage_output)
                yield StageRecord(
                    task_id=instance.task_id,
                    task_domain=self.adapter.domain,
                    model_id=self.model_id,
                    stage_index=stage_index,
                    stage_name=stage_name,
                    stage_output=stage_output,
                    context=context,
                    run_metadata=run_metadata,
                )


def write_records(records: Iterable[StageRecord], out_path: str | Path) -> int:
    """Write StageRecords as JSONL (one file per (task_domain, model_id)).

    Returns the number of records written.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(record.model_dump_json())
            fh.write("\n")
            count += 1
    return count
