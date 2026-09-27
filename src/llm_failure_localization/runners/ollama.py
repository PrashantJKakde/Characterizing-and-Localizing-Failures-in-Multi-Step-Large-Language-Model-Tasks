"""Ollama backend: calls a local Ollama server's HTTP API.

Requires Ollama running locally (`ollama serve`) with the target model pulled
(`ollama pull <checkpoint>`). No extra Python dependency beyond `requests`.
"""

from __future__ import annotations

import requests

from .base import ModelRunner

DEFAULT_HOST = "http://localhost:11434"


class OllamaRunner(ModelRunner):
    def __init__(
        self,
        model_id: str,
        checkpoint: str,
        host: str = DEFAULT_HOST,
        timeout: float = 120.0,
    ) -> None:
        self.model_id = model_id
        self.checkpoint = checkpoint
        self.host = host.rstrip("/")
        self.timeout = timeout

    def generate(self, prompt: str, *, max_new_tokens: int, seed: int) -> str:
        response = requests.post(
            f"{self.host}/api/generate",
            json={
                "model": self.checkpoint,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_new_tokens, "seed": seed},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()["response"]
