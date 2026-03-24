# Benchmark Generation Pipeline — Runbook

Operational guide for running the NRC benchmark generation pipeline.

## Prerequisites

- Python 3.11+ environment with `.[dev]` extras installed
- OpenAI API key in `.env` or `OPENAI_API_KEY` environment variable
- Parsed eCFR corpus available (run `make index` first if needed)
- Sufficient API quota for LLM stages (see Cost Estimation below)
- Qdrant running locally for Stage 5b hard negative mining (or use `--skip-hard-negatives`)

## Running a Full Pipeline

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "run_$(date +%Y%m%d_%H%M%S)" \
  --output-dir benchmark_runs/ \
  --model gpt-4o \
  --query-classes citation_lookup,unanswerable \
  --export-path eval/datasets/benchmark_v1.jsonl \
  --valid-as-of "$(date +%Y-%m-%d)"
```

To skip hard negative mining (no Qdrant required):

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "run_$(date +%Y%m%d_%H%M%S)" \
  --output-dir benchmark_runs/ \
  --model gpt-4o \
  --skip-hard-negatives
```

### Output Directory Structure

```
benchmark_runs/<run_id>/
├── run_config.json                  # Pipeline config snapshot
├── stage_0_spans.jsonl              # BenchmarkSourceSpan records
├── stage_1a_units.jsonl             # RegulatoryUnit (structural)
├── stage_1b_classified.jsonl        # RegulatoryUnit (LLM-enriched)
├── stage_2_evidence.jsonl           # EvidenceSet records
├── stage_3_candidates.jsonl         # QueryCandidate records
├── stage_5a_validated.jsonl         # ValidationResult records
├── stage_5a_queries.jsonl           # ValidatedQuery records (passing only)
├── stage_5a_refined_evidence.jsonl  # Refined EvidenceSet per validated query
├── stage_5b_hard_negatives.jsonl    # HardNegativeResult records (if retriever provided)
└── benchmark_records.jsonl          # BenchmarkRecord records (final assembled output)
```

The `--export-path` flag writes an additional EvalQuery-compatible JSONL file
outside the run directory, suitable for use with the eval harness.

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
| `stage_5a_queries.jsonl` | `ValidatedQuery` records (passing only) | Subset of 5a |
| `stage_5a_refined_evidence.jsonl` | Refined `EvidenceSet` per validated query | One per validated |
| `stage_5b_hard_negatives.jsonl` | `HardNegativeResult` records | One per validated query |
| `benchmark_records.jsonl` | `BenchmarkRecord` records (full assembled) | One per validated query |

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

# Check hard negative counts
./scripts/py -c "
import json, sys
from collections import Counter
counts = Counter()
for line in open(sys.argv[1]):
    rec = json.loads(line)
    counts['insufficient' if rec['insufficient'] else 'ok'] += 1
print(dict(counts))
" benchmark_runs/<run_id>/stage_5b_hard_negatives.jsonl
```

## Resuming a Run

If a run is interrupted or you want to re-run from a specific stage:

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "<existing_run_id>" \
  --output-dir benchmark_runs/ \
  --resume-from stage_3 \
  --model gpt-4o
```

This reads `stage_2_evidence.jsonl` as input and re-runs Stage 3 onward.

| `--resume-from` | Reads checkpoint | Re-runs |
|-----------------|-----------------|---------|
| `stage_1a` | `stage_0_spans.jsonl` | 1a → 1b → 2 → 3 → 5a → 5b → export |
| `stage_1b` | `stage_1a_units.jsonl` | 1b → 2 → 3 → 5a → 5b → export |
| `stage_2` | `stage_1b_classified.jsonl` | 2 → 3 → 5a → 5b → export |
| `stage_3` | `stage_2_evidence.jsonl` | 3 → 5a → 5b → export |
| `stage_5a` | `stage_3_candidates.jsonl` | 5a → 5b → export |
| `stage_5b` | `stage_5a_queries.jsonl` | 5b → export |
| `export` | `stage_5a_queries.jsonl` | export only |

## Cost Estimation

Approximate LLM API calls per stage at v1 target scale (~50 regulatory
units):

| Stage | LLM Calls | Notes |
|-------|-----------|-------|
| Stage 0 (source view) | 0 | Deterministic |
| Stage 1a (structural) | 0 | Deterministic |
| Stage 1b (classification) | ~50 | One per unit |
| Stage 2 (evidence) | ~50 | One per unit |
| Stage 3 (query gen, citation_lookup) | ~50 | One per unit |
| Stage 3 (query gen, unanswerable) | ~50 | One per unit (M4) |
| Stage 5a (validation, LLM scoring) | ~50 | One per passing candidate (M4) |
| Stage 5a (evidence refinement) | ~50 | One per passing candidate (M4) |
| Stage 5b (hard negatives) | 0 | Retriever calls, no LLM |
| **Total (all classes + LLM validator)** | **~300** | |

At typical GPT-4o pricing, a full run with all M4 stages costs roughly $3–8
depending on unit text length and query class mix. Use `--skip-hard-negatives`
to avoid the Qdrant dependency; hard negatives can be mined in a separate pass.

## Troubleshooting

### Snapshot Mismatch

```
ValueError: Corpus snapshot mismatch: expected 'abc123', got 'def456'
```

The pipeline's `corpus_snapshot_id` config doesn't match what Stage 0
produced. Either re-index the corpus or update the config to match.

### Missing Port

```
ValueError: Stage 5a requires a QueryValidator but none was provided
```

A stage was reached that requires a port not passed to `PipelineRunner`.
Check that all required adapters are wired in the runner construction.

### Hard Negative Mining Skipped

```
INFO benchmark.pipeline.runner: Stage 5b: no retriever provided — skipping hard negative mining
```

Normal when `--skip-hard-negatives` is passed or Qdrant is not running.
Hard negatives will be empty in `benchmark_records.jsonl`. To populate them,
run `--resume-from stage_5b` after Qdrant is available.

### Insufficient Hard Negatives

```
INFO benchmark.pipeline.runner: Candidate qc_...: only 1 hard negatives found (min=2)
```

The retriever returned few results not already in the evidence set. The
`HardNegativeResult.insufficient` flag will be `true` for this record.
This is normal for short regulatory units with high lexical overlap.

### Malformed Checkpoint

```
json.JSONDecodeError: Expecting value: line 47 column 1
```

A checkpoint file was corrupted (partial write during interruption).
Delete the corrupted file and resume from the prior stage:

```bash
rm benchmark_runs/<run_id>/stage_3_candidates.jsonl
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "<run_id>" --output-dir benchmark_runs/ --resume-from stage_3
```
