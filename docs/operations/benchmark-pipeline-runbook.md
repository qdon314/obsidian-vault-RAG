# Benchmark Generation Pipeline — Runbook

Operational guide for running the NRC benchmark generation pipeline.

## Prerequisites

- Python 3.11+ environment with `.[dev]` extras installed
- OpenAI API key in `.env` or `OPENAI_API_KEY` environment variable
- Parsed eCFR corpus available (run `make index` first if needed)
- Sufficient API quota for LLM stages (see Cost Estimation below)

## Running a Full Pipeline

```bash
./scripts/py -m benchmark.pipeline.runner \
  --run-id "run_$(date +%Y%m%d_%H%M%S)" \
  --output-dir benchmark_runs/ \
  --model gpt-4o
```

### Output Directory Structure

```
benchmark_runs/<run_id>/
├── run_config.json              # Pipeline config snapshot
├── stage_0_spans.jsonl          # BenchmarkSourceSpan records
├── stage_1a_units.jsonl         # RegulatoryUnit (structural)
├── stage_1b_classified.jsonl    # RegulatoryUnit (LLM-enriched)
├── stage_2_evidence.jsonl       # EvidenceSet records
├── stage_3_candidates.jsonl     # QueryCandidate records
└── stage_5a_validated.jsonl     # ValidationResult records
```

## Checkpoint Files

Each stage writes a JSONL checkpoint on completion. One JSON object per
line, serialized via `dataclasses.asdict()`.

| File | Contains | Typical Size |
|------|----------|-------------|
| `stage_0_spans.jsonl` | `BenchmarkSourceSpan` records from corpus | Hundreds |
| `stage_1a_units.jsonl` | `RegulatoryUnit` records (structural) | Tens–hundreds |
| `stage_1b_classified.jsonl` | `RegulatoryUnit` records (LLM-enriched) | Same count as 1a |
| `stage_2_evidence.jsonl` | `EvidenceSet` records per unit | One per unit |
| `stage_3_candidates.jsonl` | `QueryCandidate` records | 2–5 per unit |
| `stage_5a_validated.jsonl` | `ValidationResult` records | One per candidate |

### Inspecting Checkpoints

```bash
# Count records in a checkpoint
wc -l benchmark_runs/<run_id>/stage_3_candidates.jsonl

# Pretty-print the first record
head -1 benchmark_runs/<run_id>/stage_3_candidates.jsonl | python3 -m json.tool

# Filter candidates by query class
./scripts/py -c "
import json, sys
for line in open(sys.argv[1]):
    rec = json.loads(line)
    if rec['query_class'] == 'citation_lookup':
        print(rec['query'])
" benchmark_runs/<run_id>/stage_3_candidates.jsonl
```

## Resuming a Run

If a run is interrupted or you want to re-run from a specific stage:

```bash
./scripts/py -m benchmark.pipeline.runner \
  --run-id "<existing_run_id>" \
  --output-dir benchmark_runs/ \
  --resume-from stage_3 \
  --model gpt-4o
```

This reads `stage_2_evidence.jsonl` as input and re-runs Stage 3 onward.

| `--resume-from` | Reads checkpoint | Re-runs |
|-----------------|-----------------|---------|
| `stage_1a` | `stage_0_spans.jsonl` | 1a → 1b → 2 → 3 → 5a |
| `stage_1b` | `stage_1a_units.jsonl` | 1b → 2 → 3 → 5a |
| `stage_2` | `stage_1b_classified.jsonl` | 2 → 3 → 5a |
| `stage_3` | `stage_2_evidence.jsonl` | 3 → 5a |
| `stage_5a` | `stage_3_candidates.jsonl` | 5a only |

## Cost Estimation

Approximate LLM API calls per stage at v1 target scale (~50 regulatory
units):

| Stage | LLM Calls | Notes |
|-------|-----------|-------|
| Stage 0 (source view) | 0 | Deterministic |
| Stage 1a (structural) | 0 | Deterministic |
| Stage 1b (classification) | ~50 | One per unit |
| Stage 2 (evidence) | ~50 | One per unit |
| Stage 3 (query gen) | ~50 | One per unit (citation_lookup only in M3) |
| Stage 5a (validation) | 0 | Deterministic |
| **Total** | **~150** | |

At typical GPT-4o pricing, a full run costs roughly $1–3 depending on
unit text length.

## Troubleshooting

### Snapshot Mismatch

```
ValueError: Corpus snapshot mismatch: expected 'abc123', got 'def456'
```

The pipeline's `corpus_snapshot_id` config doesn't match what Stage 0
produced. Either re-index the corpus or update the config to match.

### Missing Port

```
ValueError: Stage 3 requires a QueryGenerator but none was provided
```

A stage was reached that requires a port not passed to `PipelineRunner`.
Check that all required adapters are wired in the runner construction.

### Malformed Checkpoint

```
json.JSONDecodeError: Expecting value: line 47 column 1
```

A checkpoint file was corrupted (partial write during interruption).
Delete the corrupted file and resume from the prior stage:

```bash
rm benchmark_runs/<run_id>/stage_3_candidates.jsonl
./scripts/py -m benchmark.pipeline.runner \
  --run-id "<run_id>" --output-dir benchmark_runs/ --resume-from stage_3
```
