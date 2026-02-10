# User Guide

Practical guide to build an index, ask questions, run evaluations, and inspect results.

## Prerequisites

- Python 3.11+
- Local virtual environment at `.venv`
- `OPENAI_API_KEY` when using OpenAI embeddings/generation

## Setup

1. Create and activate the virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies using repo wrappers.

```bash
./scripts/pip install -e ".[dev,openai]"
```

3. Set your API key.

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

## Build an Index

Use `make` (recommended):

```bash
make index
```

Or run directly with the pinned interpreter:

```bash
./scripts/py scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index
```

Dummy embeddings (no API cost):

```bash
make index-dummy
```

### Common Index Overrides

```bash
./scripts/py scripts/build_index.py \
  --corpus ~/obsidian-vault \
  --index-name my_index \
  --target-chars 3500 \
  --hard-max-chars 4800 \
  --overlap-blocks 1 \
  --no-heading-preamble
```

## Ask Questions

Use `make`:

```bash
make ask QUERY="What are the main concepts?"
```

Or:

```bash
./scripts/py scripts/ask.py \
  --index my_index \
  --q "What are the main concepts?" \
  --top-k 10 \
  --token-budget 1800
```

## Run Evaluations

Run an eval set:

```bash
./scripts/py eval/scripts/run_eval.py \
  --queries eval/datasets/curated_queries.jsonl \
  --index artifacts/indexes/obsidian \
  --run-generation \
  --use-llm-judge \
  --top-k 10 \
  --keep-k 4
```

Run outputs are written under `eval/runs/run_YYYY_MM_DDTHH-MM/`.

Generate a release verdict:

```bash
make verdict
```

## Analyze Evaluation Results

Launch the results analyzer:

```bash
make results
```

Equivalent pinned command:

```bash
./scripts/py -m streamlit run eval/app/results_analyzer.py
```

## Logs and Traces

Query traces are written to:

- `artifacts/logs/traces.jsonl` for normal CLI runs
- `eval/runs/<run>/traces.jsonl` for eval runs

View recent traces:

```bash
tail -f artifacts/logs/traces.jsonl | jq .
```

## Troubleshooting

### `OPENAI_API_KEY is required but not set`

```bash
export OPENAI_API_KEY='sk-your-key'
```

### `Missing config file: settings.toml`

Ensure you are running commands from the repository root and `settings.toml` exists.

### `No manifest.json` warning during ask

This means index compatibility checks were skipped for that index directory. Rebuild the index with `scripts/build_index.py` to regenerate a manifest.

### No results from retrieval

- Confirm the index name/path matches what you built
- Increase `--top-k`
- Check corpus contents were ingested

## Command Discipline

For local repo workflows:

- Use `make <target>` when available
- Otherwise use `./scripts/py ...` and `./scripts/pip ...`

Do not run `python`, `pip`, `pytest`, `ruff`, or `streamlit` directly for local development commands in this repo.
