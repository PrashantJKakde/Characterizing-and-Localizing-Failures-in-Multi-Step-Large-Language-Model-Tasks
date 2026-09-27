"""A no-model runner that echoes its prompt back.

Not a real backend — it exists purely so the pipeline's wiring (adapter ->
runner -> stage logger) can be smoke-tested with `stagefail run` and no GPU,
model download, or server. Never use it to produce data for analysis.
"""

from __future__ import annotations

from .base import ModelRunner


class EchoRunner(ModelRunner):
    def __init__(self, model_id: str = "echo") -> None:
        self.model_id = model_id

    def generate(self, prompt: str, *, max_new_tokens: int, seed: int) -> str:
        return prompt[:max_new_tokens]
