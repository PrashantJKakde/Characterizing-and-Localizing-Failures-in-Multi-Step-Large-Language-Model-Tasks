"""Runner backends and the config-driven factory (M2).

Network calls (Ollama, HF Transformers) are never exercised here — only the
EchoRunner and, for Ollama, a monkeypatched `requests.post` — so this suite
stays fast and offline, matching CI.
"""

from __future__ import annotations

import pytest

from llm_failure_localization.config import ModelConfig
from llm_failure_localization.runners import load_runner
from llm_failure_localization.runners.echo import EchoRunner
from llm_failure_localization.runners.ollama import OllamaRunner


def test_echo_runner_echoes_and_truncates() -> None:
    runner = EchoRunner(model_id="echo")
    assert runner.generate("hello world", max_new_tokens=5, seed=0) == "hello"


def test_load_runner_echo_backend() -> None:
    config = ModelConfig(model_id="echo", backend="echo")
    runner = load_runner(config)
    assert isinstance(runner, EchoRunner)


def test_load_runner_unknown_backend_raises() -> None:
    config = ModelConfig(model_id="x", backend="not_a_backend")
    with pytest.raises(ValueError, match="Unknown backend"):
        load_runner(config)


def test_load_runner_vllm_raises_not_implemented() -> None:
    config = ModelConfig(model_id="x", backend="vllm")
    with pytest.raises(NotImplementedError):
        load_runner(config)


def test_load_runner_ollama_requires_checkpoint() -> None:
    config = ModelConfig(model_id="x", backend="ollama")
    with pytest.raises(ValueError, match="checkpoint"):
        load_runner(config)


def test_ollama_runner_generate_posts_expected_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": "42"}

    def fake_post(url: str, json: dict, timeout: float) -> FakeResponse:
        captured["url"] = url
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("llm_failure_localization.runners.ollama.requests.post", fake_post)

    runner = OllamaRunner(model_id="qwen", checkpoint="qwen2.5:7b-instruct")
    result = runner.generate("2 + 2 = ?", max_new_tokens=8, seed=3)

    assert result == "42"
    assert captured["url"].endswith("/api/generate")
    assert captured["json"]["model"] == "qwen2.5:7b-instruct"
    assert captured["json"]["options"]["seed"] == 3
    assert captured["json"]["options"]["num_predict"] == 8
