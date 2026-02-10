# Running Evaluations

This document covers retrieval and end-to-end evaluation workflows using the current harness and CLI.

## Primary Entry Point

Use:

```bash
./scripts/py eval/scripts/run_eval.py
```

The script wires container dependencies, runs evaluation, saves artifacts, and prints a summary.

## Prerequisites

1. Built index (for example `artifacts/indexes/obsidian`)
2. Query dataset in JSONL format (`EvalQuery` rows)
3. `OPENAI_API_KEY` when using OpenAI models or LLM judge

## Common CLI Workflows

### Retrieval-only evaluation

```bash
./scripts/py eval/scripts/run_eval.py \
  --queries eval/datasets/curated_queries.jsonl \
  --index artifacts/indexes/obsidian
```

### Full pipeline evaluation with judge

```bash
./scripts/py eval/scripts/run_eval.py \
  --queries eval/datasets/curated_queries.jsonl \
  --index artifacts/indexes/obsidian \
  --run-generation \
  --use-llm-judge \
  --judge-model gpt-4o-mini \
  --top-k 10 \
  --keep-k 4 \
  --token-budget 1500
```

### Score retrieved IDs instead of reranked IDs

```bash
./scripts/py eval/scripts/run_eval.py \
  --queries eval/datasets/curated_queries.jsonl \
  --index artifacts/indexes/obsidian \
  --run-generation \
  --score-ids retrieved
```

## Important Flags

| Flag | Description |
|------|-------------|
| `--queries` | Path to eval queries JSONL |
| `--index` | Index directory path |
| `--output` | Parent directory for run outputs (default `eval/runs`) |
| `--run-name` | Optional label stored in run metadata |
| `--top-k` | Retrieval candidate count |
| `--keep-k` | Rerank keep count |
| `--token-budget` | Context token budget |
| `--run-generation` | Enable answer generation |
| `--use-llm-judge` | Enable groundedness/gold judge scoring |
| `--judge-model` | Judge model name |
| `--score-ids` | Retrieval metric IDs to score: `retrieved` or `reranked` |
| `--no-save` | Run without writing artifacts |

## Programmatic Usage

```python
from pathlib import Path

from openai import OpenAI

from rag.app.container import ContainerOverrides, build_container
from rag.eval.harness import load_eval_queries, run_full_eval, save_run

queries = load_eval_queries(Path("eval/datasets/curated_queries.jsonl"))
container = build_container(
    overrides=ContainerOverrides(
        store_backend="jsonl",
        jsonl_index_dir=Path("artifacts/indexes/obsidian"),
    )
)

judge_client = OpenAI()
run = run_full_eval(
    eval_queries=queries,
    container=container,
    queries_path="eval/datasets/curated_queries.jsonl",
    index_dir=Path("artifacts/indexes/obsidian"),
    top_k=10,
    keep_k=4,
    token_budget=1500,
    run_generation=True,
    use_llm_judge=True,
    judge_client=judge_client,
    judge_model="gpt-4o-mini",
    score_ids="reranked",
)

run = save_run(run, Path("eval/runs/run_manual"))
print(run.artifacts)
```

## Artifact Layout

Each run directory contains:

- `metrics.json` (aggregates + metadata)
- `results.jsonl` (per-query outputs)
- `traces.jsonl` (query traces when generation path logs)

The harness also updates `eval/runs/latest` symlink to the newest saved run.

## Post-Run Verdict

```bash
./scripts/py eval/scripts/verdict.py \
  --current eval/runs/latest \
  --baseline eval/runs/baseline \
  --output eval/verdicts \
  --fail-on-block
```

See [Verdict and Release Gating](verdict_release_gating.md) for thresholds and baseline management.
