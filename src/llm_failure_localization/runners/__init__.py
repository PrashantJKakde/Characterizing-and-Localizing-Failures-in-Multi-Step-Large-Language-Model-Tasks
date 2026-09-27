"""Model serving/inference backends (open-weights only): HF Transformers, vLLM, Ollama.

Every runner exposes the same `generate(prompt, **kwargs) -> str` interface
(see `.base.ModelRunner`) so swapping models is a config change, not a code
change (M2). `hf_transformers` and `ollama` are implemented; `vllm` is not yet.
"""

from __future__ import annotations

from ..config import ModelConfig
from .base import ModelRunner
from .echo import EchoRunner
from .hf_transformers import HFTransformersRunner
from .ollama import OllamaRunner

RUNNER_REGISTRY: dict[str, type[ModelRunner]] = {
    "echo": EchoRunner,
    "hf_transformers": HFTransformersRunner,
    "ollama": OllamaRunner,
}


def load_runner(model_config: ModelConfig) -> ModelRunner:
    """Instantiate the backend named in `model_config.backend`."""
    if model_config.backend == "echo":
        return EchoRunner(model_id=model_config.model_id)

    if model_config.backend == "ollama":
        if model_config.checkpoint is None:
            raise ValueError("The 'ollama' backend requires 'checkpoint' (the Ollama model tag).")
        return OllamaRunner(model_id=model_config.model_id, checkpoint=model_config.checkpoint)

    if model_config.backend == "hf_transformers":
        if model_config.checkpoint is None:
            raise ValueError(
                "The 'hf_transformers' backend requires 'checkpoint' (a Hugging Face model id)."
            )
        return HFTransformersRunner(
            model_id=model_config.model_id,
            checkpoint=model_config.checkpoint,
            quantization=model_config.quantization,
        )

    if model_config.backend == "vllm":
        raise NotImplementedError(
            "The 'vllm' backend is not implemented yet; use 'hf_transformers' or 'ollama'."
        )

    raise ValueError(
        f"Unknown backend '{model_config.backend}'; known backends are "
        f"{sorted(RUNNER_REGISTRY)} (plus 'vllm', not yet implemented)."
    )


__all__ = ["RUNNER_REGISTRY", "ModelRunner", "load_runner"]
