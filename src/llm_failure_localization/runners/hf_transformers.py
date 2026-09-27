"""Hugging Face Transformers backend: loads a checkpoint locally with `pipeline`.

Optional dependency — install with `pip install -e ".[inference]"`. The model
is loaded once, at construction, and reused across `generate` calls.
"""

from __future__ import annotations

from .base import ModelRunner


class HFTransformersRunner(ModelRunner):
    def __init__(
        self,
        model_id: str,
        checkpoint: str,
        quantization: str | None = None,
        device_map: str = "auto",
    ) -> None:
        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "HFTransformersRunner requires the optional 'inference' extras: "
                "pip install -e '.[inference]'"
            ) from exc

        self.model_id = model_id
        self.checkpoint = checkpoint

        model_kwargs: dict[str, bool] = {}
        if quantization == "4bit":
            model_kwargs["load_in_4bit"] = True
        elif quantization == "8bit":
            model_kwargs["load_in_8bit"] = True
        elif quantization is not None:
            raise ValueError(
                f"Unsupported quantization '{quantization}'; use '4bit', '8bit', or null."
            )

        self._pipeline = transformers.pipeline(
            "text-generation",
            model=checkpoint,
            device_map=device_map,
            model_kwargs=model_kwargs,
        )

    def generate(self, prompt: str, *, max_new_tokens: int, seed: int) -> str:
        import transformers

        transformers.set_seed(seed)
        outputs = self._pipeline(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            return_full_text=False,
        )
        return outputs[0]["generated_text"]
