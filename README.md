# TRACE: Task-Aware Adaptive Self-Evolving Agentic Jailbreaking

This repo contains the implementation used in the paper. TRACE is a task-aware, adaptive self-evolution framework for agentic jailbreaking that proceeds in three stages: task decomposition, task-aware subtask induction, and feedback-driven self-evolution.

## Demo Video 1
Download: https://anonymous.4open.science/api/repo/TRACE-99B3/file/asset/Stack-based%20Control%20Manipulation.mp4?v=43bcb4c2&download=true

<!-- <div align="center">
  <img src="asset/Stack-based Control Manipulation.gif" alt="Stack-based Control Manipulation" width="100%">
</div> -->


## Demo Video 2

Download: https://anonymous.4open.science/api/repo/TRACE-99B3/file/asset/Common-Modulus%20Key%20Compromise.mp4?v=d950f542&download=true

<!-- <div align="center">
  <img src="asset/Common-Modulus Key Compromise.gif" alt="Common-Modulus Key Compromise" width="100%">
</div> -->


## Pipeline Overview

1. Decompose tasks into multiple candidate sub-prompt sequences.
2. Score each sub-prompt for harmfulness.
3. Select the candidate sequence with the fewest harmful steps.
4. Execute each step and optimize harmful steps when needed.

## Environment

Set environment variables (see .env.example). At minimum, `OPENAI_API_KEY` is required for model calls.

Optional overrides:
- `OPENAI_BASE_URL`, `OPENAI_ORG`, `OPENAI_PROJECT`
- `OPENAI_TEMPLATE_MODEL` (default: `gpt-4o-2024-11-20`)
- `HF_GUARD3_MODEL`
- `HF_EMBEDDING_MODEL`

### Conda environment (merged)

We provide a merged environment file that combines dependencies from multiple setups:

```bash
conda env create -f environment.yaml
conda activate trace
```

## Usage

### 0) Generate component pools (role/environment/directive/tips)

```bash
python component_initialize.py --dataset practical
```

Parameters:
- `--dataset` one of `agentharm`, `advcua`, `practical`

### 1) Decompose tasks

```bash
python task_decomposer.py \
  --data_file_path data/agentharm_harmful_behaviors_detailed.json \
  --output_file_path data/task_decompose/agentharm_decomposed_candidates.jsonl \
  --num_decompositions_per_task 5
```

Parameters:
- `--model` (default: `gpt-4o-2024-11-20`)
- `--prompt_field` (override dataset prompt field)
- `--target_ids` (comma-separated ids)

### 2) Score sub-prompts

```bash
python subtask_score.py \
  --evaluation guard3 \
  --data_file_path data/task_decompose/agentharm_decomposed_candidates.jsonl \
  --output_file_path data/task_decompose/agentharm_decomposed_scored.jsonl
```

Parameters:
- `--evaluation` one of `guard3`, `template`
- `--guard3_model_path` (HF model path)
- `--template_model` (OpenAI model name for template scoring)

### 3) Select the least harmful candidate

```bash
python select_candidate_sub_prompts.py \
  --input_file data/task_decompose/agentharm_decomposed_scored.jsonl \
  --output_file data/task_decompose/agentharm_decomposed_selected.jsonl
```

### 4) Execute steps

```bash
python execute_subtask.py \
  --dataset agentharm \
  --data_file_path data/task_decompose/agentharm_decomposed_selected.jsonl \
  --component_pool_file_path data/initial_pool/agentharm_component_initial_pool.json \
  --target_model_name gpt-4o-2024-11-20
```

Parameters (common):
- `--intermediate_result_dir`, `--final_result_dir`
- `--threshold` (harmful score threshold)
- `--num_iters`, `--num_candidate_per_iter`, `--elitism`
- `--use_memory`, `--memory_path`, `--memory_top_k`, `--embedding_model_name`
- `--use_decision_matrix`, `--decision_matrix_path`

## Data Formats

- `task_decomposer.py` expects a JSON file containing a list of items with `id` and the prompt field.
- `subtask_score.py` expects a JSONL file with `id` and `sub_prompts`.
- `select_candidate_sub_prompts.py` expects a JSONL file with `id`, `sub_prompts`, and `harmful_scores`.
- `execute_subtask.py` expects a JSONL file with `id`, `sub_prompts`, and `harmful_scores`.

## Outputs

- Decomposition candidates are written as JSONL (one candidate per line).
- Scores and selected candidates are written as JSONL.
- Execution outputs are written to `final_results/` by default.
- Logs are written under `logs/`.
