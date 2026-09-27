# Quickstart

## Smoke test (no model, no GPU, no network)

```bash
pip install -e ".[dev]"
stagefail run --task configs/tasks/gsm8k_fixture.example.yaml --model configs/models/echo.example.yaml
```

This runs the GSM8K adapter's 5-stage prompt sequence over the 3 fixture problems in
`tests/fixtures/gsm8k_sample.jsonl`, through `EchoRunner` (which just echoes the
prompt back), and writes `data/math_reasoning__echo.jsonl`. It exists to verify the
adapter → runner → pipeline wiring end-to-end — never use it to produce data for
analysis.

## Real run

Against Hugging Face Transformers (downloads the checkpoint locally):

```bash
pip install -e ".[dev,inference]"
stagefail run --task configs/tasks/gsm8k.example.yaml --model configs/models/qwen2.5-7b.example.yaml
```

Or against a local [Ollama](https://ollama.com) server instead — no `transformers`/`torch`
install needed:

```bash
ollama pull qwen2.5:7b-instruct
ollama serve  # in another terminal
```

then point a model config at it (copy `configs/models/qwen2.5-7b.example.yaml` and set
`backend: ollama` and `checkpoint: qwen2.5:7b-instruct`).

`--out` and `--limit` are optional; `--out` defaults to
`data/<task_domain>__<model_id>.jsonl` and `--limit` caps how many task instances run.

## Next steps (not yet implemented)

```bash
stagefail evaluate --data data/math_reasoning__qwen2.5-7b.jsonl       # M6
stagefail localize --data data/math_reasoning__qwen2.5-7b.evaluated.jsonl  # M7
stagefail compare --data data/                                        # M8
```

Each step reads and writes the one stage-indexed record shape defined in
`src/llm_failure_localization/schema.py` — see `docs/schema.md` for the field reference.
