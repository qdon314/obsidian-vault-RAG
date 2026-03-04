# Phase 8: Query Curation & S3 Eval Run Loading

**Date:** 2026-02-21
**Scope:** Expand Phase 8 of the NRC case ingestion plan to include S3 eval run loading in the Streamlit results analyzer, alongside the originally planned query curation workflow.

---

## Part A: S3 Eval Run Loading

### Problem

Remote eval runs (executed via `run_remote_eval.py` on ECS) push results to S3 (`s3://{bucket}/eval/runs/{label}/`). The Streamlit results analyzer only reads from the local `eval/runs/` directory. To view remote results, you must manually download the S3 prefix — no discoverability, no lazy loading.

### Design

#### A.1 S3RunLoader (New Adapter)

**File:** `eval/app/results/adapters/s3_loader.py`

Implements the existing `RunLoader` protocol (`eval/app/results/ports/run_loader.py`).

```python
@dataclass(frozen=True, slots=True)
class S3RunLoader:
    bucket: str
    prefix: str          # e.g. "eval/runs"
    cache_dir: Path      # e.g. eval/runs/.s3-cache
    client: Any = field(default=None, repr=False)
```

**Methods:**

- **`discover_runs() -> list[RunSummary]`** — Lists S3 prefixes under `{prefix}/`, downloads only `metrics.json` from each run prefix. Parses into `RunSummary` objects with `source="s3"`. Caches discovered summaries in memory (invalidated by `refresh()`).

- **`load_run(run_id) -> LoadedRun`** — Downloads the full run directory (`metrics.json` + `results.jsonl` + `traces.jsonl`) to `{cache_dir}/{run_id}/`. Delegates parsing to `FilesystemRunLoader._load_run_from_dir()` (reuse, not reimplementation). Cache persists on disk across Streamlit reruns.

- **`load_summary(run_id) -> RunSummary`** — Downloads and parses only `metrics.json` for a single run.

**S3 interaction pattern:** Uses `list_objects_v2` with `Delimiter="/"` to enumerate run prefixes efficiently, then `download_file` for individual files. Mirrors `_download_s3_prefix()` from `scripts/run_remote_eval.py`.

#### A.2 RunSummary Source Field

**File:** `eval/app/results/domain/models.py`

Add an optional `source` field to `RunSummary`:

```python
source: str = "local"  # "local" | "s3"
```

The `FilesystemRunLoader` leaves this as `"local"` (default). The `S3RunLoader` sets it to `"s3"`.

#### A.3 Repository Composition

**File:** `eval/app/results/adapters/repository.py`

`InMemoryRunRepository` gains an optional `s3_loader: S3RunLoader | None` field.

`_discover_all_runs()` merges runs from both loaders:
1. Discover local runs (existing behavior).
2. If `s3_loader` is set, discover S3 runs.
3. Deduplicate by `run_id` — local wins if the same run exists in both.
4. Sort by timestamp descending.

`get_run()` tries the local loader first, then the S3 loader.

#### A.4 UI Integration

**File:** `eval/app/results/ui/run_selector.py`

The run selector dropdown appends `[S3]` to the display name for remote runs. When an S3 run is selected for the first time, a Streamlit spinner shows during download. Subsequent selections are instant (disk cache hit).

#### A.5 Wiring

**File:** `eval/app/results_analyzer.py`

During app initialization:
- Read `RAG_EVAL_S3_BUCKET` env var (or fall back to `settings.toml` `chunk_storage.s3_bucket`).
- If a bucket is configured, instantiate `S3RunLoader(bucket=..., prefix="eval/runs", cache_dir=runs_dir / ".s3-cache")`.
- Pass to `InMemoryRunRepository(loader=fs_loader, s3_loader=s3_loader)`.
- If no bucket configured, `s3_loader=None` — no S3 UI, no errors.

#### A.6 Configuration

No new settings.toml sections needed. Uses existing:
- `chunk_storage.s3_bucket` (or `distributed_ingestion.corpus_s3_bucket`)
- `RAG_EVAL_S3_PREFIX` env var (default: `"eval"`)

AWS credentials come from the environment (same as `run_remote_eval.py`).

---

## Part B: Query Curation

### B.1 Query Curation Streamlit Page

**File:** `eval/app/query_curator.py` (new Streamlit page)

A Streamlit page for reviewing generated queries from the case-based query generation pipeline.

**Features:**
- Load queries from `case_generated_queries_DRAFT.jsonl`.
- Display each query with metadata: source case path, generation strategy, difficulty level, relevant citations.
- Show linked regulatory sections (clickable links to corpus files).
- Approve / Edit / Reject buttons per query.
- Batch actions: approve all from a strategy, reject all below a quality threshold.
- Export approved queries to `case_generated_queries.jsonl`.
- Session state persists review progress across Streamlit reruns.

**Query display fields:**
- `qid`, `query` (editable text area)
- `query_type`, `difficulty` (editable dropdowns)
- `relevant_citations` (list, editable)
- `source_case` (read-only, links to source markdown)
- `generation_strategy` (read-only)
- `adversarial_note` (if present)
- `is_unanswerable` + `unanswerable_reason` (if applicable)

### B.2 Manual Curation Process

**Input:** `eval/datasets/case_generated_queries_DRAFT.jsonl`
**Output:** `eval/datasets/case_generated_queries.jsonl`

Curation criteria:
- Clarity and naturalness of query text.
- Answer verifiability against the regulatory corpus.
- Appropriate difficulty rating.
- Adversarial value (for adversarial/facility-specific queries).
- Correct `relevant_citations` (can the answer be found in these sections?).

**Target:** Start with ~500-1000 draft queries, curate to ~250-500 production queries (30-50% pass rate).

### B.3 Dataset Merge

**File:** `eval/datasets/all_queries.jsonl` (new combined dataset)

Combine:
- `regulatory_adversarial.jsonl` (existing manual queries)
- `case_generated_queries.jsonl` (curated case-derived queries)

Add a `dataset_source` field to each query: `"manual"` | `"case_generated"`.

Deduplication: by query text similarity (exact match). If a duplicate exists, the case-generated version takes precedence (it has richer metadata from the generation pipeline).

Update eval scripts to support the combined dataset path.

---

## File Inventory

### New Files
| File | Purpose |
|------|---------|
| `eval/app/results/adapters/s3_loader.py` | S3RunLoader adapter |
| `eval/app/query_curator.py` | Streamlit curation page |
| `eval/datasets/all_queries.jsonl` | Combined query dataset (output) |

### Modified Files
| File | Change |
|------|--------|
| `eval/app/results/domain/models.py` | Add `source` field to `RunSummary` |
| `eval/app/results/adapters/repository.py` | Add `s3_loader` support to `InMemoryRunRepository` |
| `eval/app/results/ui/run_selector.py` | Show `[S3]` badge, spinner on download |
| `eval/app/results_analyzer.py` | Wire S3 loader conditionally |

### Unchanged
| File | Why |
|------|-----|
| `eval/app/results/adapters/filesystem_loader.py` | Existing behavior preserved |
| `eval/app/results/ports/run_loader.py` | Protocol unchanged (S3 loader satisfies it structurally) |
| `scripts/run_remote_eval.py` | Already pushes to correct S3 prefix |

---

## Validation Plan

### Part A (S3 Loading)
- Unit test: `S3RunLoader.discover_runs()` with mocked boto3 (list_objects_v2 response).
- Unit test: `S3RunLoader.load_run()` downloads to cache, parses correctly.
- Unit test: `InMemoryRunRepository` merges local + S3 runs, deduplicates correctly.
- Integration test: Point at real S3 bucket, verify runs appear in Streamlit.

### Part B (Query Curation)
- Load 50 draft queries in the curation UI, test approve/edit/reject workflow.
- Verify exported JSONL matches expected schema.
- Verify merged `all_queries.jsonl` deduplicated correctly.
- Run eval harness on merged dataset to confirm compatibility.

---

## Risks

### S3 Latency on Discovery
`discover_runs()` downloads `metrics.json` from every S3 run prefix. With 50+ runs, this could take 5-10 seconds.

**Mitigation:** Cache discovery results in Streamlit session state. Add a "Refresh" button that re-discovers. Consider parallel downloads with `ThreadPoolExecutor`.

### AWS Credentials in Local Dev
Local Streamlit usage requires valid AWS credentials to list/download S3 objects.

**Mitigation:** S3 loading is conditional — if no bucket or no credentials, it silently degrades to local-only mode. Log a warning, don't crash.

### Curation Bottleneck
Manual review of 500-1000 queries is time-intensive.

**Mitigation:** Batch actions in the curation UI (approve-all-by-strategy). Validation script from Phase 7 pre-filters obvious issues. Quality > quantity — 250 great queries beats 1000 mediocre ones.
