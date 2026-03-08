# Results Analyzer v2 — Design

## Goal

Rebuild `eval/app/results_analyzer.py` as a greenfield **evaluation workbench** in `eval/app_v2/`. The result is not a nicer dashboard — it is an analysis system where a bad aggregate can be traced to a bad query, a bad pipeline stage, and a deterministic root cause.

Reference specs: `docs/specs/results-app-redesign.md`, `docs/plans/v2-feedback.md`, `docs/plans/dash-feedback.md`

---

## Architecture principle

**Engine-first.** All analysis logic lives in a pure-Python `engine/` package with zero Streamlit imports. The Streamlit `ui/` layer is a thin renderer over typed engine objects. The engine is independently testable and usable from notebooks or CI.

The hard rule: `engine/` has no `import streamlit`. `ui/` imports from `engine/` but not vice versa.

---

## Directory layout

```
eval/app_v2/
├── engine/
│   ├── domain/
│   │   ├── enums.py            # All diagnostic/status/severity enums
│   │   ├── warnings.py         # BundleWarningCode, BundleWarning
│   │   └── models.py           # All frozen dataclasses
│   ├── loaders/
│   │   ├── base.py             # ArtifactLoader protocol, LoadedArtifact
│   │   ├── registry.py         # Loader registration
│   │   ├── bundle.py           # build_bundle() entry point
│   │   ├── metrics.py
│   │   ├── results.py
│   │   ├── traces.py
│   │   └── verdict.py
│   ├── derived/
│   │   ├── diagnostics.py      # build_query_diagnostic(), analyze_queries()
│   │   ├── stage_attribution.py # Classifier: QueryRecord -> DiagnosticCode
│   │   ├── health.py           # build_health() -> RunHealthSummary
│   │   ├── slices.py           # build_slice_table() -> SliceMetricTable
│   │   └── contributors.py     # Attribute aggregates to contributing queries
│   ├── services/
│   │   ├── forensics.py        # Query navigation/selection over analyzed queries
│   │   ├── comparison.py       # RunBundle x RunBundle -> ComparisonBundle
│   │   ├── trend.py            # [RunBundle] -> TrendBundle
│   │   └── filter.py           # Facet-based filtering of AnalyzedQuery lists
│   ├── facets/
│   │   └── registry.py         # Declarative FacetDef list
│   └── adapters/
│       └── pandas.py           # DataFrame conversion — kept outside engine core
├── ui/
│   ├── app.py                  # Shell, run selector, cache boundary
│   ├── pages/
│   │   ├── triage.py
│   │   ├── forensics.py
│   │   ├── compare.py
│   │   ├── trends.py
│   │   ├── verdicts.py
│   │   └── artifacts.py
│   └── widgets/
│       ├── facet_panel.py
│       ├── metric_cards.py
│       ├── trace_viewer.py
│       └── diagnostic_card.py
└── app.py                      # Entry point: streamlit run eval/app_v2/app.py
```

The existing `eval/app/` and `make results` target remain untouched during development.

---

## Domain models

### `engine/domain/enums.py`

```python
class DiagnosticCode(str, Enum):
    NO_CLEAR_FAILURE            = "no_clear_failure"
    RETRIEVAL_MISS              = "retrieval_miss"
    RETRIEVAL_PARTIAL           = "retrieval_partial"
    RERANK_DROPPED_RELEVANT     = "rerank_dropped_relevant"
    RERANK_DEGRADED_RANK        = "rerank_degraded_rank"
    PACKING_OMITTED_RELEVANT    = "packing_omitted_relevant"
    PACKING_TRUNCATED_RELEVANT  = "packing_truncated_relevant"
    GROUNDED_ANSWER             = "grounded_answer"
    UNSUPPORTED_ANSWER          = "unsupported_answer"
    BAD_ABSTAIN_ON_ANSWERABLE   = "bad_abstain_on_answerable"
    FAILED_ABSTAIN_ON_UNANSWERABLE = "failed_abstain_on_unanswerable"
    TRACE_MISSING               = "trace_missing"
    DATA_INSUFFICIENT           = "data_insufficient"

class Severity(str, Enum):
    OK       = "ok"
    MINOR    = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"

class RetrievalStatus(str, Enum):
    HIT     = "hit"
    PARTIAL = "partial"
    MISS    = "miss"
    UNKNOWN = "unknown"

class RerankStatus(str, Enum):
    IMPROVED = "improved"
    NEUTRAL  = "neutral"
    DEGRADED = "degraded"
    ABSENT   = "absent"
    UNKNOWN  = "unknown"

class PackingStatus(str, Enum):
    COMPLETE  = "complete"
    TRUNCATED = "truncated"
    OMITTED   = "omitted"
    ABSENT    = "absent"
    UNKNOWN   = "unknown"

class GenerationStatus(str, Enum):
    GROUNDED          = "grounded"
    UNSUPPORTED       = "unsupported"
    ABSTAINED         = "abstained"
    FAILED_TO_ABSTAIN = "failed_to_abstain"
    ABSENT            = "absent"
    UNKNOWN           = "unknown"

class DeltaDirection(str, Enum):
    IMPROVED    = "improved"
    REGRESSED   = "regressed"
    UNCHANGED   = "unchanged"
    INSUFFICIENT = "insufficient"

class ComparisonClassification(str, Enum):
    IMPROVED         = "improved"
    REGRESSED        = "regressed"
    MIXED            = "mixed"
    UNCHANGED        = "unchanged"
    INSUFFICIENT_DATA = "insufficient_data"
```

### `engine/domain/warnings.py`

```python
class BundleWarningCode(str, Enum):
    MISSING_TRACES          = "missing_traces"
    MISSING_VERDICT         = "missing_verdict"
    PARTIAL_TRACE_PARSE     = "partial_trace_parse"
    PARTIAL_RESULTS_PARSE   = "partial_results_parse"
    SCHEMA_VERSION_UNKNOWN  = "schema_version_unknown"
    TRACE_TEXT_REDACTED     = "trace_text_redacted"
    ORPHAN_TRACE            = "orphan_trace"
    MISSING_TRACE_FOR_RESULT = "missing_trace_for_result"

@dataclass(frozen=True, slots=True)
class BundleWarning:
    code: BundleWarningCode
    message: str
    artifact_name: str | None = None
```

### `engine/domain/models.py`

#### `QueryRecord`

Normalized per-query data. Traces are joined at load time — no runtime lookups. Stage outputs (`reranked_chunk_ids`, `packed_chunk_ids`) are promoted to explicit top-level fields so diagnostic functions never spelunk through raw trace JSON.

```python
@dataclass(frozen=True, slots=True)
class QueryRecord:
    qid: str
    query: str
    query_type: str | None
    difficulty: str | None
    is_unanswerable: bool
    requires_synthesis: bool
    tags: tuple[str, ...]

    # Retrieval
    relevant_chunk_ids: frozenset[str]
    retrieved_chunk_ids: tuple[str, ...]        # ordered by rank
    reranked_chunk_ids: tuple[str, ...] | None  # normalized from trace
    packed_chunk_ids: tuple[str, ...] | None    # normalized from trace

    # Per-query metrics
    per_query_recall_at_k: Mapping[int, float]
    per_query_precision_at_k: Mapping[int, float]
    per_query_ndcg_at_k: Mapping[int, float]
    per_query_hit_rate_at_k: Mapping[int, float]

    # Generation (optional)
    answer_text: str | None
    answer_metrics: AnswerMetrics | None
    groundedness: GroundednessOutcome | None
    latency_ms: int | None

    # Trace join key + joined trace (may be None if traces.jsonl is absent)
    trace_id: str | None
    trace: QueryTrace | None
```

#### `QueryDiagnostic`

Deterministically derived from a `QueryRecord`. The primary output of stage attribution.

```python
@dataclass(frozen=True, slots=True)
class QueryDiagnostic:
    qid: str
    diagnostic_code: DiagnosticCode
    severity: Severity
    retrieval_status: RetrievalStatus
    rerank_status: RerankStatus
    packing_status: PackingStatus
    generation_status: GenerationStatus
    root_cause_summary: str
    suggested_next_check: str | None
    evidence_present: bool
    trace_available: bool
```

#### `AnalyzedQuery`

Wrapper pairing a normalized record with its derived diagnostic. Kept separate from `QueryRecord` to allow multiple diagnostic strategies in future.

```python
@dataclass(frozen=True, slots=True)
class AnalyzedQuery:
    record: QueryRecord
    diagnostic: QueryDiagnostic
```

#### Slice models

```python
@dataclass(frozen=True, slots=True)
class SliceKey:
    parts: tuple[tuple[str, str], ...]  # e.g. (("query_type", "conceptual"), ("difficulty", "hard"))

@dataclass(frozen=True, slots=True)
class SliceMetricRow:
    key: SliceKey
    size: int
    metrics: Mapping[str, float | None]

@dataclass(frozen=True, slots=True)
class SliceMetricTable:
    group_by: tuple[str, ...]
    rows: tuple[SliceMetricRow, ...]
```

#### `RunHealthSummary`

Derived in a single pass over all `AnalyzedQuery` objects. Built over `DiagnosticCode` values, not strings.

```python
@dataclass(frozen=True, slots=True)
class RunHealthSummary:
    headline_recall_at_10: float
    headline_ndcg_at_10: float
    avg_quality_score: float | None
    avg_latency_ms: float | None
    severity_counts: Mapping[Severity, int]
    diagnostic_counts: Mapping[DiagnosticCode, int]
    dominant_failure_mode: DiagnosticCode | None
    dominant_failure_summary: str | None
    worst_slice: SliceKey | None
    verdict_status: Literal["SHIP", "BLOCK"] | None
```

#### `RunBundle`

Top-level unit. Replaces `LoadedRun`.

```python
@dataclass(frozen=True, slots=True)
class RunBundle:
    run_id: str
    display_name: str
    timestamp: datetime
    config: RunConfig               # normalized from EvalRunMeta
    aggregates: EvalAggregates      # reused from existing domain
    queries: tuple[AnalyzedQuery, ...]
    health: RunHealthSummary
    verdict: VerdictSummary | None
    warnings: tuple[BundleWarning, ...]
    raw_artifacts: Mapping[str, object]   # passthrough for Artifacts page
```

#### Comparison models

```python
@dataclass(frozen=True, slots=True)
class QueryDeltaSummary:
    retrieval: DeltaDirection
    groundedness: DeltaDirection
    latency: DeltaDirection
    severity: DeltaDirection

@dataclass(frozen=True, slots=True)
class ComparedQuery:
    qid: str
    query: str
    retrieval_delta: float | None
    ndcg_delta: float | None
    latency_delta_ms: float | None
    quality_delta: float | None
    diagnostic_before: QueryDiagnostic | None
    diagnostic_after: QueryDiagnostic | None
    delta_summary: QueryDeltaSummary
    classification: ComparisonClassification

@dataclass(frozen=True, slots=True)
class ComparisonBundle:
    run_a: RunBundle
    run_b: RunBundle
    aggregate_deltas: Mapping[str, float | None]
    slice_deltas: SliceMetricTable | None
    compared_queries: tuple[ComparedQuery, ...]
```

#### Trend models

```python
@dataclass(frozen=True, slots=True)
class ConfigFieldChange:
    field_name: str
    before: object
    after: object

@dataclass(frozen=True, slots=True)
class ConfigChangeEvent:
    from_run_id: str
    to_run_id: str
    timestamp: datetime
    changes: tuple[ConfigFieldChange, ...]
    annotation: str | None = None

@dataclass(frozen=True, slots=True)
class TrendBundle:
    runs: tuple[RunBundle, ...]
    timestamps: tuple[datetime, ...]
    metric_series: Mapping[str, tuple[float | None, ...]]
    diagnostic_rate_series: Mapping[DiagnosticCode, tuple[float | None, ...]]
    verdict_series: tuple[str | None, ...]
    config_change_events: tuple[ConfigChangeEvent, ...]
```

---

## Artifact loader boundary

```python
@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    artifact_name: str
    payload: object | None
    warnings: tuple[BundleWarning, ...]

class ArtifactLoader(Protocol):
    artifact_name: str
    def can_load(self, run_dir: Path) -> bool: ...
    def load(self, run_dir: Path) -> LoadedArtifact: ...
```

`build_bundle()` iterates registered loaders, collects payloads and warnings, assembles normalized `QueryRecord` objects, runs stage attribution, and produces the final `RunBundle`. Today's loaders: `MetricsLoader`, `ResultsLoader`, `TracesLoader`, `VerdictLoader`.

---

## Derived-data layer

All derived objects are computed once inside `build_bundle()`. Nothing is recomputed inside Streamlit callbacks.

### Stage attribution (`engine/derived/stage_attribution.py`)

A classifier that returns a `DiagnosticCode` first, then separately maps that code to stage statuses, severity, and human-readable text. Decision order:

1. **Data sufficiency** — malformed row → `DATA_INSUFFICIENT`; trace required but absent → `TRACE_MISSING`
2. **Unanswerable behavior** — `is_unanswerable=True` and answer not abstained → `FAILED_ABSTAIN_ON_UNANSWERABLE`; `is_unanswerable=False` and abstained despite evidence → `BAD_ABSTAIN_ON_ANSWERABLE`
3. **Retrieval** — no relevant chunk retrieved → `RETRIEVAL_MISS`; partial → `RETRIEVAL_PARTIAL`
4. **Rerank** — relevant dropped after rerank → `RERANK_DROPPED_RELEVANT`; ranks materially worsened → `RERANK_DEGRADED_RANK`
5. **Packing** — relevant survived rerank but absent from packed set → `PACKING_OMITTED_RELEVANT`; token budget forced omission → `PACKING_TRUNCATED_RELEVANT`
6. **Generation** — evidence present, answer unsupported → `UNSUPPORTED_ANSWER`; grounded → `GROUNDED_ANSWER`
7. **Fallback** — `NO_CLEAR_FAILURE`

Severity mapping lives in a single module — never scattered across pages:

| Severity | Conditions |
|---|---|
| CRITICAL | retrieval miss + unsupported answer on answerable query; failed abstain on unanswerable; unsupported answer with clear supporting evidence |
| MODERATE | retrieval miss only; rerank dropped relevant; packing omitted/truncated; bad abstain on answerable |
| MINOR | retrieval partial; rerank degraded rank without total loss; trace missing on otherwise decent query |
| OK | grounded answer; no clear failure |

### `forensics.py` service (`engine/services/forensics.py`)

Navigation and selection over already-derived diagnostics. Does not construct new diagnosis.

```python
def get_query(bundle: RunBundle, qid: str) -> AnalyzedQuery | None: ...
def list_queries_by_code(bundle: RunBundle, code: DiagnosticCode) -> tuple[AnalyzedQuery, ...]: ...
def list_queries_by_slice(bundle: RunBundle, slice_key: SliceKey) -> tuple[AnalyzedQuery, ...]: ...
def worst_queries(bundle: RunBundle, limit: int = 10) -> tuple[AnalyzedQuery, ...]: ...
def contributor_queries_for_failure_mode(bundle: RunBundle, code: DiagnosticCode, limit: int = 20) -> tuple[AnalyzedQuery, ...]: ...
```

Rule of thumb: `derived/` = "what does this query mean?"; `services/` = "which queries should the user look at?"

### Slice analysis (`engine/derived/slices.py`)

```python
def build_slice_table(
    queries: Sequence[AnalyzedQuery],
    group_by: Sequence[str],
) -> SliceMetricTable: ...
```

Accepts any list of `QueryRecord` field names as grouping keys. Replaces hardcoded by-type / by-difficulty sections.

### Config change detection (`engine/services/trend.py`)

```python
def detect_config_change_events(
    runs: Sequence[RunBundle],
    tracked_fields: set[str] | None = None,
) -> tuple[ConfigChangeEvent, ...]: ...
```

Detection mechanism: structural diff of adjacent normalized `RunConfig` objects, restricted to a curated tracked-field set. Optional `annotation` from run metadata may be attached but is not used for detection.

**Track:** retriever implementation, index name, reranker model/top_n, chunker strategy/size/overlap, generator model, retrieval k, token budget, hybrid weights.

**Ignore:** output paths, timestamps, run labels, logging verbosity.

### `ComparisonClassification` rule

Each `ComparedQuery` carries per-dimension `DeltaDirection` first; the summary `ComparisonClassification` is derived from those.

Per-dimension materiality thresholds (illustrative):
- Retrieval: recall@k or ndcg@k delta ≥ 0.05
- Groundedness: hallucination status changed, or groundedness score delta exceeds threshold
- Latency: > 100 ms or > 10% change
- Severity: diagnostic severity enum moved at least one level

Classification rule:
1. No usable dimensions → `INSUFFICIENT_DATA`
2. No material changes → `UNCHANGED`
3. One or more improvements, zero regressions → `IMPROVED`
4. One or more regressions, zero improvements → `REGRESSED`
5. At least one improvement and at least one regression → `MIXED`

Severity-aware override: a critical behavioral regression (e.g. groundedness or severity moved to CRITICAL) can dominate a minor retrieval improvement, yielding `REGRESSED` instead of `MIXED`. The per-dimension breakdown is always preserved in `QueryDeltaSummary` regardless of the summary label.

---

## Streamlit app shell

```python
@st.cache_data(show_spinner="Building run bundle...")
def load_bundle(run_id: str, run_dir: str) -> RunBundle:
    return build_bundle(Path(run_dir))

def main():
    summaries = discover_runs(DEFAULT_RUNS_DIR)
    with st.sidebar:
        page = st.radio("Page", list(PAGES))
        selected = run_selector_widget(summaries)
    bundle = load_bundle(selected.run_id, str(selected.run_dir)) if selected else None
    PAGES[page](bundle)
```

Pages are plain callables `(RunBundle | None) -> None`.

**Session state conventions — two keys only:**
- `st.session_state["selected_run_id"]` — persists across page switches
- `st.session_state["forensics_qid"]` — lets Triage push a qid directly into Forensics

**Widgets** in `ui/widgets/` are stateless functions. They accept typed engine objects and render them. They never call services or load data.

---

## Pages

### Triage

Landing page. Shows: KPI cards from `RunHealthSummary`, severity breakdown bar (ok/minor/moderate/critical), dominant failure mode banner with `DiagnosticCode`, worst-slice table, verdict badge (SHIP/BLOCK) with failed rules, top-N critical queries as `diagnostic_card` widgets each with a "Forensics" link that sets `session_state["forensics_qid"]`.

### Forensics

One-query deep inspection. Shows:
- Query header: qid, full text, type, difficulty, tags, unanswerable flag
- `QueryDiagnostic` card: severity badge, `DiagnosticCode`, stage status indicators, root cause sentence, suggested next check
- Retrieval panel: relevant/retrieved/matched/missed/extra chunk sets with per-query metrics
- Pipeline drill-down (collapsible): retriever candidates+scores, reranker rank changes+score deltas+dropped chunks, context packing (packed vs available, token budget), generation (answer text, citations), groundedness (supported/unsupported claims by role)

### Compare

Two-run comparison. Shows: aggregate delta metrics, `SliceMetricTable` delta, `ComparedQuery` list filtered by `ComparisonClassification`, per-query `QueryDeltaSummary` breakdown alongside summary label (so `MIXED` is never opaque).

### Trends

Multi-run analysis. Shows: time-series chart per metric, `diagnostic_rate_series` chart (e.g. RERANK_DROPPED_RELEVANT rate over time), `ConfigChangeEvent` annotations on all charts, verdict timeline overlay, run summary table.

### Verdicts

Shows: SHIP/BLOCK badge, failed threshold checks (rule name, threshold, actual value, delta), contributor queries per failed rule (from `contributors.py`), historical verdict timeline.

### Artifacts

Shows: loader warning list (`BundleWarning`), raw artifact JSON/JSONL viewers, download links, schema version info. Present in Phase 3 alongside Triage/Forensics — useful for loader debugging and schema mismatch inspection early.

---

## Extensibility hooks

### Declarative facets (`engine/facets/registry.py`)

```python
@dataclass(frozen=True)
class FacetDef:
    key: str
    label: str
    value_type: Literal["enum", "bool", "numeric_bucket"]
    extract: Callable[[QueryRecord], Any]
    higher_is_better: bool = True

FACETS: list[FacetDef] = [
    FacetDef("query_type",          "Query Type",    "enum", ...),
    FacetDef("difficulty",          "Difficulty",    "enum", ...),
    FacetDef("requires_synthesis",  "Synthesis",     "bool", ...),
    FacetDef("is_unanswerable",     "Unanswerable",  "bool", ...),
    FacetDef("diagnostic.severity", "Severity",      "enum", ...),
]
```

`facet_panel` reads `FACETS` and renders the correct widget per type automatically. Adding a new facet is one list entry.

### Pandas boundary

`engine/adapters/pandas.py` contains all DataFrame conversion. No `import pandas` inside `engine/domain/`, `engine/derived/`, or `engine/services/`. The engine stays usable from plain Python.

### Metric registry

Intentionally deferred to Phase 5. All current metrics are typed fields on existing models. The registry abstraction should be introduced once the "add a new metric" pattern is felt concretely.

---

## Phased delivery

### Phase 1 — Engine backbone

1. `engine/domain/enums.py` — all diagnostic/status/severity enums
2. `engine/domain/warnings.py` — `BundleWarningCode`, `BundleWarning`
3. `engine/domain/models.py` — all dataclasses
4. `engine/loaders/base.py` — `ArtifactLoader`, `LoadedArtifact`
5. `engine/loaders/metrics.py`
6. `engine/loaders/results.py`
7. `engine/loaders/traces.py`
8. `engine/loaders/verdict.py`
9. `engine/loaders/bundle.py` — `build_bundle(run_dir: Path) -> RunBundle`

**Exit criterion:** load a run directory into a complete typed bundle with warnings.

### Phase 2 — Deterministic diagnosis

10. `engine/derived/stage_attribution.py` — classifier returning `DiagnosticCode`, severity mapping, prose mapping
11. `engine/derived/diagnostics.py` — `build_query_diagnostic()`, `analyze_queries()`
12. `engine/derived/health.py` — `build_health() -> RunHealthSummary`
13. `engine/derived/slices.py` — generic `build_slice_table()`

**Exit criterion:** the engine is already useful without Streamlit.

### Phase 3 — Minimum viable UI

14. `ui/app.py` — sidebar, run selection, cache boundary
15. `ui/widgets/metric_cards.py`
16. `ui/widgets/diagnostic_card.py`
17. `ui/pages/triage.py`
18. `ui/pages/forensics.py`
19. `ui/pages/artifacts.py`

**Exit criterion:** load run → identify bad queries → drill into pipeline-stage root cause.

### Phase 4 — Comparison and verdict

20. `engine/services/comparison.py` — `ComparedQuery`, `ComparisonBundle`, classification logic
21. `engine/derived/contributors.py` — contributor queries for verdict failures
22. `engine/services/forensics.py` — query navigation functions
23. `ui/pages/compare.py`
24. `ui/pages/verdicts.py`

**Exit criterion:** app is useful for release decisions.

### Phase 5 — Extensibility

25. `engine/facets/registry.py`
26. `ui/widgets/facet_panel.py`
27. `engine/services/filter.py`
28. Schema version detection and graceful fallback in loaders
29. Metric registry (if pattern has emerged from Phase 1–4 work)

### Phase 6 — Trends and advanced analysis

30. `engine/services/trend.py` — `TrendBundle`, `detect_config_change_events()`
31. `ui/pages/trends.py`
32. Chunk-centric views
33. Cost/latency correlation views
34. Query clustering / cohort analysis

---

## What is not changing

- `eval/app/results/` and `eval/app/results_analyzer.py` remain untouched
- `make results` continues to point at the existing analyzer until `app_v2` is ready
- Existing domain models (`EvalAggregates`, `EvalResult`, `EvalRunMeta`, `QueryTrace`) are reused directly where possible — `RunBundle` wraps rather than replaces them
