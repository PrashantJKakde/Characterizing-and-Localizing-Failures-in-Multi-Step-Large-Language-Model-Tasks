# Characterizing and Localizing Failures in Multi-Step Large Language Model Tasks

A small pipeline that runs a task through an open-weights model, captures every intermediate
stage's output, and localizes, classifies, and traces the failure when the final answer is wrong.

This is the engineering counterpart to the thesis of the same name. It is a measurement
toolkit, not a self-correction or auto-repair system — it does not fix model outputs, it
characterizes how and where they fail.

Python package / CLI name: `llm_failure_localization` / `stagefail` (unchanged — see below).

## Scope

- **Open-weights models only.** No proprietary/closed-source APIs are used anywhere in this
  codebase, by design (cost, reproducibility, and experimental control — see the thesis
  proposal, Section 4.2). Model backends: Hugging Face Transformers, vLLM, or Ollama.
- **Analytical, not corrective.** The pipeline observes and labels failures; it never rewrites
  or "fixes" a model's output.

## Quickstart

```bash
pip install -e ".[dev]"
pytest
stagefail run --task configs/tasks/gsm8k_fixture.example.yaml --model configs/models/echo.example.yaml
```

The `stagefail run` line above is a no-model, no-network smoke test of the pipeline
wiring. See `docs/quickstart.md` for running a real model end-to-end.

## Repository layout

```text
src/llm_failure_localization/
├── adapters/       # task-domain adapters: load a benchmark, define its stage split
├── runners/         # model serving/inference backends (HF Transformers / vLLM / Ollama)
├── pipeline/         # executes a task instance stage-by-stage, logs outputs
├── evaluation/       # per-stage correctness protocol
├── localization/     # failure localization, taxonomy classification, dependency + propagation analysis
├── analysis/         # cross-model / cross-task comparison and reporting
└── schema.py         # the shared stage-indexed record format

configs/               # one YAML per task domain / model / stage schema
tests/                 # unit tests + fixtures
docs/                  # quickstart and schema documentation
```

## Status

- **M0** — repo scaffold, license, CI, src layout, stage schema, base adapter. Done.
- **M1** — concrete task adapter: `GSM8KAdapter` (`math_reasoning` domain, 5-stage prompt
  sequence, local-JSONL or Hugging Face `datasets` loading). Done.
- **M2** — model runners: `EchoRunner` (smoke test), `OllamaRunner`, `HFTransformersRunner`.
  `vllm` not yet implemented. Done.
- **M3** — pipeline executor + JSONL stage logger, wired to `stagefail run`. Done.
- **M6-M8** — per-stage evaluation, failure localization/classification/propagation,
  cross-model/task comparison. Not started; see `stagefail evaluate|localize|compare`.

See the project's implementation plan doc for the full milestone sequence.

## License

Apache-2.0 — see `LICENSE`.
