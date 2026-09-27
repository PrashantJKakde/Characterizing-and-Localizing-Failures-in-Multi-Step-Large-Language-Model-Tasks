# llm-multistep-failure-localization

Characterizing and localizing failures in multi-step LLM tasks: a small pipeline that runs a
task through an open-weights model, captures every intermediate stage's output, and localizes,
classifies, and traces the failure when the final answer is wrong.

This is the engineering counterpart to the thesis *"Characterizing and Localizing Failures in
Multi-Step Large Language Model Tasks."* It is a measurement toolkit, not a self-correction or
auto-repair system — it does not fix model outputs, it characterizes how and where they fail.

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
```

See `docs/quickstart.md` for running one task instance through one model end-to-end.

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

Early scaffold (Milestone M0). See the project's implementation plan doc for the full
milestone sequence.

## License

Apache-2.0 — see `LICENSE`.
