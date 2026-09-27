# Quickstart

> This will run end-to-end once M1-M3 land. For now it documents the intended shape.

```bash
pip install -e ".[dev,inference]"
stagefail run --task configs/tasks/gsm8k.example.yaml --model configs/models/qwen2.5-7b.example.yaml
stagefail evaluate --data data/math_reasoning__qwen2.5-7b.jsonl
stagefail localize --data data/math_reasoning__qwen2.5-7b.evaluated.jsonl
stagefail compare --data data/
```

Each step reads and writes the one stage-indexed record shape defined in
`src/llm_failure_localization/schema.py` — see `docs/schema.md` for the field reference.
