# User Guide

Practical guide to ingest regulatory corpora, build indexes, ask questions, run evaluations, and inspect results.

## Prerequisites

- Python 3.11+
- Local virtual environment at `.venv`
- `OPENAI_API_KEY` when using OpenAI embeddings/generation
- `NRC_ADAMS_API_KEY` when fetching NRC case documents (optional)

## Setup

1. Create and activate the virtual environment.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies using repo wrappers.

```bash
./scripts/pip install -e ".[dev,openai,qdrant]"
```

3. Set your API key.

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

## Regulatory Corpus Ingestion

### Index eCFR (Code of Federal Regulations)

The primary corpus is 10 CFR, ingested from eCFR XML. This normalizes each section into canonical markdown, enriches metadata with cross-references and citation keys, then chunks and indexes.

```bash
# Index 10 CFR Part 50 from eCFR XML
make index-regulatory \
  REGULATORY_XML=data/ecfr/title-10-part-50.xml \
  REGULATORY_PART=50
```

With dummy embeddings (no API cost, for development):

```bash
make index-regulatory-dummy
```

#### Normalize Only (No Indexing)

To produce canonical markdown files without embedding and indexing:

```bash
make normalize-regulatory \
  REGULATORY_XML=data/ecfr/title-10-part-50.xml \
  REGULATORY_PART=50
```

#### Push Normalized Corpus to S3

```bash
make push-regulatory-s3 \
  REGULATORY_S3_BUCKET=my-bucket \
  REGULATORY_S3_PREFIX=regulatory/part-50 \
  REGULATORY_PART=50
```

### Fetch NRC Case Documents

Fetch case documents from the NRC ADAMS Public Search API:

```bash
export NRC_ADAMS_API_KEY='your-key-here'
./scripts/py scripts/fetch_nrc_cases.py
```

Configuration in `settings.toml`:

```toml
[case_ingestion]
output_dir = "corpus/us-nrc/cases"
document_types = ["Inspection Report", "Special Inspection", "Part 21 Correspondence"]
```

### Generate Evaluation Queries from Case Documents

```bash
./scripts/py scripts/generate_case_queries.py
```

## Build a General Index

For non-regulatory corpora (markdown, text files):

Use `make` (recommended):

```bash
make index
```

Or run directly with the pinned interpreter:

```bash
./scripts/py scripts/build_index.py \
  --corpus /path/to/corpus \
  --index-name my_index
```

Dummy embeddings (no API cost):

```bash
make index-dummy
```

### Common Index Overrides

```bash
./scripts/py scripts/build_index.py \
  --corpus /path/to/corpus \
  --index-name my_index \
  --target-chars 3500 \
  --hard-max-chars 4800 \
  --overlap-blocks 1 \
  --no-heading-preamble
```

## Ask Questions

Use `make`:

```bash
make ask QUERY="What are the requirements for ECCS under 10 CFR 50.46?"
```

Or:

```bash
./scripts/py scripts/ask.py \
  --index regulatory \
  --q "What emergency core cooling system acceptance criteria apply?" \
  --top-k 10 \
  --token-budget 1800
```

## Run Evaluations

Run an eval set:

```bash
./scripts/py eval/scripts/run_eval.py \
  --queries eval/datasets/curated_queries.jsonl \
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

## Curate Evaluation Queries

Launch the query curator UI:

```bash
make curate
```

## Logs and Traces

Query traces are written to:

- `artifacts/logs/traces.jsonl` for normal CLI runs
- `eval/runs/<run>/traces.jsonl` for eval runs

View recent traces:

```bash
tail -f artifacts/logs/traces.jsonl | jq .
```

## Remote Operations (ECS)

For distributed ingestion and remote evaluation on AWS:

```bash
# Distributed ingestion
make ingest-remote CORPUS_ID=regulatory WORKERS=3

# Remote evaluation
make eval-remote

# Remote ad-hoc query
make query-remote QUERY="What does 10 CFR 50.46 require?"
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
