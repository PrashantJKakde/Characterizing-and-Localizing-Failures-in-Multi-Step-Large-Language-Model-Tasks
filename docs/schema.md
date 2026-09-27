# Data Schema — Stage-Indexed Intermediate Output

Every module reads and writes one record shape (`llm_failure_localization.schema.StageRecord`),
so a new task domain or analysis script never needs a special case.

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `task_id` | str | yes | unique id of the task instance within its benchmark |
| `task_domain` | str | yes | e.g. `math_reasoning`, `multihop_qa`, `rag`, `agentic_tooluse` |
| `model_id` | str | yes | model family + parameter scale, e.g. `qwen2.5-7b` |
| `stage_index` | int | yes | 0-based position in the task's defined stage sequence |
| `stage_name` | str | yes | e.g. `understanding`, `retrieval`, `reasoning`, `calculation`, `synthesis` |
| `stage_output` | str / json | yes | the model's raw output at this stage |
| `context` | json | yes | prompt, retrieved evidence, and tool calls feeding this stage |
| `correctness` | enum | yes | `correct` / `incorrect` / `partial` / `unsupported` (Step 5) |
| `is_root_failure` | bool | no | set once localization (Step 6) runs |
| `failure_category` | str | no | taxonomy label (Step 7); null until classified |
| `propagation_flag` | bool | no | true if attributed to an earlier root failure (Steps 8-9) |
| `run_metadata` | json | yes | seed, config hash, timestamp, package version |

Records are written as one JSON object per line (JSONL) — one file per `(task_domain, model_id)`
pair — so a run can be re-evaluated or re-aggregated without re-executing the model.
