# Benchmark Generation Pipeline — Runbook

Operational guide for running the NRC benchmark generation pipeline.

## Prerequisites

- Python 3.11+ environment with `.[dev,openai]` extras installed
- OpenAI API key in `.env` or `OPENAI_API_KEY` environment variable
- eCFR XML file for the regulatory part you want to benchmark (Stage 0 input)
- Parsed eCFR corpus indexed (run `make index` first — needed for chunk overlap resolution)
- Sufficient API quota for LLM stages (see Cost Estimation below)
- Qdrant running locally for hard negative mining (or use `--skip-hard-negatives`)

## Running a Full Pipeline

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "run_$(date +%Y%m%d_%H%M%S)" \
  --output-dir benchmark_runs/ \
  --model gpt-4o \
  --ecfr-xml data/ecfr_part50.xml \
  --doc-id ecfr_part50 \
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
  --ecfr-xml data/ecfr_part50.xml \
  --doc-id ecfr_part50 \
  --skip-hard-negatives
```

### Key Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `--run-id` | Yes | Unique identifier for this run (used as output subdirectory) |
| `--ecfr-xml PATH` | For Stage 0 | Path to the eCFR XML file to benchmark. Not required when resuming from `unit_extraction` or later. |
| `--doc-id` | No | Document ID to assign to source spans (default: `ecfr`) |
| `--model` | No | OpenAI model for all LLM stages (default: `gpt-4o`) |
| `--query-classes` | No | Comma-separated query classes (default: `citation_lookup`) |
| `--valid-as-of` | No | ISO date for `valid_as_of` field on all records |
| `--skip-hard-negatives` | No | Skip Stage 5b; no Qdrant required |
| `--export-path` | No | Write EvalQuery-compatible JSONL for the eval harness |
| `--synthesize-gold-answers` | No | Enable Stage 6 gold answer synthesis (M5) |
| `--contamination-model` | No | Model ID for Stage 5c contamination probe (M5) |
| `--resume-from STAGE` | No | Resume from a specific stage; see Resuming a Run below |

### Output Directory Structure

```
benchmark_runs/<run_id>/
├── run_config.json                           # Pipeline config snapshot
├── source_spans.jsonl                        # BenchmarkSourceSpan records
├── unit_extraction.jsonl                     # RegulatoryUnit (structural)
├── unit_classification.jsonl                 # RegulatoryUnit (LLM-enriched)
├── evidence_tiers.jsonl                      # EvidenceSet records
├── candidate_generation.jsonl                # QueryCandidate records
├── query_validation_results.jsonl            # ValidationResult records
├── query_validation.jsonl                    # ValidatedQuery records (passing only)
├── query_validation_refined_evidence.jsonl   # Refined EvidenceSet per validated query
├── hard_negative_mining.jsonl                # HardNegativeResult records (if retriever provided)
├── gold_answer_synthesis.jsonl               # BenchmarkRecord records with gold answers (M5)
├── contamination_probe.jsonl                 # BenchmarkRecord records with probe results (M5)
└── benchmark_records.jsonl                   # BenchmarkRecord records (final assembled output)
```

The `--export-path` flag writes an additional EvalQuery-compatible JSONL file
outside the run directory, suitable for use with the eval harness.

## Checkpoint Files

Each stage writes a JSONL checkpoint on completion. One JSON object per
line, serialized via `dataclasses.asdict()`.

| File | Contains | Typical Size |
|------|----------|-------------|
| `source_spans.jsonl` | `BenchmarkSourceSpan` records from corpus | Hundreds |
| `unit_extraction.jsonl` | `RegulatoryUnit` records (structural) | Tens–hundreds |
| `unit_classification.jsonl` | `RegulatoryUnit` records (LLM-enriched) | Same count |
| `evidence_tiers.jsonl` | `EvidenceSet` records per unit | One per unit |
| `candidate_generation.jsonl` | `QueryCandidate` records | 2–5 per unit |
| `query_validation_results.jsonl` | `ValidationResult` records | One per candidate |
| `query_validation.jsonl` | `ValidatedQuery` records (passing only) | Subset |
| `query_validation_refined_evidence.jsonl` | Refined `EvidenceSet` per validated query | One per validated |
| `hard_negative_mining.jsonl` | `HardNegativeResult` records | One per validated query |
| `gold_answer_synthesis.jsonl` | `BenchmarkRecord` with gold answers (M5) | One per validated query |
| `contamination_probe.jsonl` | `BenchmarkRecord` with probe results (M5) | One per validated query |
| `benchmark_records.jsonl` | `BenchmarkRecord` records (full assembled) | One per validated query |

### Inspecting Checkpoints

```bash
# Count records in a checkpoint
wc -l benchmark_runs/<run_id>/candidate_generation.jsonl

# Pretty-print the first record
head -1 benchmark_runs/<run_id>/candidate_generation.jsonl | python3 -m json.tool

# Filter candidates by query class
./scripts/py -c "
import json, sys
for line in open(sys.argv[1]):
    rec = json.loads(line)
    if rec['query_class'] == 'citation_lookup':
        print(rec['query'])
" benchmark_runs/<run_id>/candidate_generation.jsonl

# Check hard negative counts
./scripts/py -c "
import json, sys
from collections import Counter
counts = Counter()
for line in open(sys.argv[1]):
    rec = json.loads(line)
    counts['insufficient' if rec['insufficient'] else 'ok'] += 1
print(dict(counts))
" benchmark_runs/<run_id>/hard_negative_mining.jsonl
```

## Resuming a Run

If a run is interrupted or you want to re-run from a specific stage:

```bash
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "<existing_run_id>" \
  --output-dir benchmark_runs/ \
  --resume-from candidate_generation \
  --model gpt-4o
```

This reads `evidence_tiers.jsonl` as input and re-runs candidate generation onward.
`--ecfr-xml` is not required when resuming from `unit_extraction` or later — Stage 0 is skipped.

| `--resume-from` | Reads checkpoint | Re-runs |
|-----------------|-----------------|---------|
| `unit_extraction` | `source_spans.jsonl` | unit_extraction → … → export |
| `unit_classification` | `unit_extraction.jsonl` | unit_classification → … → export |
| `evidence_tiers` | `unit_classification.jsonl` | evidence_tiers → … → export |
| `candidate_generation` | `evidence_tiers.jsonl` | candidate_generation → … → export |
| `query_validation` | `candidate_generation.jsonl` | query_validation → … → export |
| `hard_negative_mining` | `query_validation.jsonl` | hard_negative_mining → export |
| `gold_answer_synthesis` | `query_validation.jsonl` | gold_answer_synthesis → contamination_probe → export |
| `contamination_probe` | `gold_answer_synthesis.jsonl` | contamination_probe → export |
| `export` | `query_validation.jsonl` | export only |

## Cost Estimation

Approximate LLM API calls per stage at v1 target scale (~50 regulatory
units):

| Stage | LLM Calls | Notes |
|-------|-----------|-------|
| `source_spans` | 0 | Deterministic |
| `unit_extraction` | 0 | Deterministic |
| `unit_classification` | ~50 | One per unit |
| `evidence_tiers` | ~50 | One per unit |
| `candidate_generation` (citation_lookup) | ~50 | One per unit |
| `candidate_generation` (unanswerable) | ~50 | One per unit (M4) |
| `query_validation` (LLM scoring) | ~50 | One per passing candidate (M4) |
| `query_validation` (evidence refinement) | ~50 | One per passing candidate (M4) |
| `hard_negative_mining` | 0 | Retriever calls, no LLM |
| `gold_answer_synthesis` | ~50 | One per validated query (M5) |
| `contamination_probe` | ~100 | Two per record: generation + judge (M5) |
| **Total (all classes + LLM validator)** | **~300–450** | |

At typical GPT-4o pricing, a full run with all M4 stages costs roughly $3–8
depending on unit text length and query class mix. Use `--skip-hard-negatives`
to avoid the Qdrant dependency; hard negatives can be mined in a separate pass.

## Troubleshooting

### Snapshot Mismatch

```
ValueError: Corpus snapshot mismatch: expected 'abc123', got 'def456'
```

The pipeline's `corpus_snapshot_id` config doesn't match what `source_spans`
produced. Either re-index the corpus or update the config to match.

### Missing Port

```
ValueError: query_validation requires a QueryValidator but none was provided
```

A stage was reached that requires a port not passed to `PipelineRunner`.
Check that all required adapters are wired in the runner construction.

### Hard Negative Mining Skipped

```
INFO benchmark.pipeline.runner: hard_negative_mining: no retriever provided — skipping
```

Normal when `--skip-hard-negatives` is passed or Qdrant is not running.
Hard negatives will be empty in `benchmark_records.jsonl`. To populate them,
run `--resume-from hard_negative_mining` after Qdrant is available.

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
rm benchmark_runs/<run_id>/candidate_generation.jsonl
./scripts/py -m benchmark.scripts.run_benchmark_gen \
  --run-id "<run_id>" --output-dir benchmark_runs/ --resume-from candidate_generation
```
