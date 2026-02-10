# Evaluation System

This directory documents how to evaluate retrieval and generation quality for the RAG pipeline.

## What It Covers

- Query dataset format (`EvalQuery` JSONL)
- Running evaluations (`eval/scripts/run_eval.py`)
- Metrics definitions and interpretation
- Trace/log inspection
- Results analysis UI
- Verdict-based release gating

## Documents

| Document | Description |
|----------|-------------|
| [Running Evaluations](running_evaluations.md) | End-to-end eval workflow and CLI usage |
| [Metrics Reference](metrics.md) | Retrieval and answer quality metrics |
| [Traces and Logging](traces_and_logging.md) | Query trace schema and debugging workflow |
| [Results Analyzer](results_analyzer.md) | Streamlit UI for run analysis/comparison/trending |
| [Verdict and Release Gating](verdict_release_gating.md) | SHIP/BLOCK decision layer and CI gate |

## Quick Start

1. Build an index.

```bash
make index
```

2. Run an evaluation.

```bash
./scripts/py eval/scripts/run_eval.py \
  --queries eval/datasets/curated_queries.jsonl \
  --index artifacts/indexes/obsidian \
  --run-generation \
  --use-llm-judge
```

3. Analyze results.

```bash
make results
```

4. Produce a release verdict.

```bash
make verdict
```

## Key Locations

- Eval script: `eval/scripts/run_eval.py`
- Eval harness: `src/rag/eval/harness.py`
- Eval schema/models: `src/rag/eval/schema.py`, `src/rag/eval/models.py`
- Default dataset: `eval/datasets/curated_queries.jsonl`
- Run outputs: `eval/runs/run_YYYY_MM_DDTHH-MM/`
