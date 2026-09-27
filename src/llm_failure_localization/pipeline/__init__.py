"""Executor + stage logger: runs one task instance stage-by-stage and writes
stage-indexed JSONL records (Step 4, M3). See `.executor.PipelineExecutor`.
"""

from __future__ import annotations

from .executor import PipelineExecutor, write_records

__all__ = ["PipelineExecutor", "write_records"]
