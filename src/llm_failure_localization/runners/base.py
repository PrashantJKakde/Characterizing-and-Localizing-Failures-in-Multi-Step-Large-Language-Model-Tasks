"""Abstract model runner.

Every backend (HF Transformers, vLLM, Ollama, ...) implements this one
`generate` method; swapping models is a config change, not a code change (M2).
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class ModelRunner(ABC):
    """Wraps one open-weights model checkpoint behind a single text-in/text-out call."""

    model_id: str

    @abstractmethod
    def generate(self, prompt: str, *, max_new_tokens: int, seed: int) -> str:
        """Return the model's raw text completion for `prompt`."""
