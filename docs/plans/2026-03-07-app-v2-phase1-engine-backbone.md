# Results Analyzer v2 — Phase 1: Engine Backbone

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Scaffold `eval/app_v2/` and produce a fully-typed `RunBundle` from a real run directory.

**Architecture:** Pure-Python `engine/` package with no Streamlit imports. All derived objects computed once inside `build_bundle()`. Existing `EvalResult`, `EvalAggregates`, `EvalRunMeta` from `src/rag/eval/models.py` are reused, not reimplemented.

**Tech Stack:** Python 3.12, dataclasses (frozen + slots), `dataclasses_json`, existing `rag.eval` models.

**Run directory structure (check a real one before implementing loaders):**
```
eval/runs/run_YYYY_MM_DDTHH-MM/
    metrics.json      # has "meta" section (EvalRunMeta) + "overall"/"by_type"/"by_difficulty"
    results.jsonl     # one EvalResult per line, parse via EvalResult.from_results_dict()
    traces.jsonl      # optional, one trace object per line
```
**Verdict lives at:** `eval/verdicts/verdict.json` (not per-run; loaded by path or convention).

**Parallel execution map:**
```
Task 1 (scaffold)
  └─ Task 2 (enums)
       └─ Task 3 (warnings)
            └─ Task 4 (models) ──────────────────────────────────┐
                 ├─ Task 5 (loader base)                         │
                 │    ├─ Tasks 6,7,8,9 (loaders — PARALLEL)      │
                 │    └─ Task 10 (registry)                      │
                 │         └─ Task 11 (bundle) ◄─────────────────┘
                 ├─ Task 12 (stage_attribution) ─────────────────►┤
                 │    └─ Task 13 (diagnostics)  ─────────────────►┤
                 │         └─ Task 14 (health)  ─────────────────►┘ (all feed Task 11)
                 └─ Task 15 (slices) ────────────────────────────►┘
```

---

## Task 1: Package scaffold

**Depends on:** nothing

**Files:**
- Create: `eval/app_v2/__init__.py`
- Create: `eval/app_v2/engine/__init__.py`
- Create: `eval/app_v2/engine/domain/__init__.py`
- Create: `eval/app_v2/engine/loaders/__init__.py`
- Create: `eval/app_v2/engine/derived/__init__.py`
- Create: `eval/app_v2/engine/services/__init__.py`
- Create: `eval/app_v2/engine/facets/__init__.py`
- Create: `eval/app_v2/engine/adapters/__init__.py`
- Create: `eval/app_v2/ui/__init__.py`
- Create: `eval/app_v2/ui/pages/__init__.py`
- Create: `eval/app_v2/ui/widgets/__init__.py`
- Create: `tests/eval/app_v2/__init__.py`
- Create: `tests/eval/app_v2/engine/__init__.py`

**Step 1: Create all `__init__.py` files**

Each file is empty. Use `touch` or write empty strings. Example for one:

```python
# eval/app_v2/__init__.py
# (empty)
```

**Step 2: Verify import path works**

```bash
./scripts/py -c "import eval.app_v2.engine"
```
Expected: no error.

**Step 3: Commit**

```bash
git add eval/app_v2/ tests/eval/app_v2/
git commit -m "chore(app-v2): scaffold package structure"
```

**Acceptance criteria:** All `__init__.py` files exist. `import eval.app_v2.engine` succeeds.

---

## Task 2: `engine/domain/enums.py`

**Depends on:** Task 1

**Files:**
- Create: `eval/app_v2/engine/domain/enums.py`
- Create: `tests/eval/app_v2/engine/test_enums.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_enums.py
from eval.app_v2.engine.domain.enums import (
    DiagnosticCode,
    Severity,
    RetrievalStatus,
    RerankStatus,
    PackingStatus,
    GenerationStatus,
    DeltaDirection,
    ComparisonClassification,
)


def test_diagnostic_codes_are_strings():
    assert DiagnosticCode.RETRIEVAL_MISS == "retrieval_miss"
    assert DiagnosticCode.NO_CLEAR_FAILURE == "no_clear_failure"


def test_severity_ordering():
    severities = [Severity.OK, Severity.MINOR, Severity.MODERATE, Severity.CRITICAL]
    assert len(severities) == 4


def test_all_enums_importable():
    assert RetrievalStatus.HIT == "hit"
    assert RerankStatus.IMPROVED == "improved"
    assert PackingStatus.COMPLETE == "complete"
    assert GenerationStatus.GROUNDED == "grounded"
    assert DeltaDirection.IMPROVED == "improved"
    assert ComparisonClassification.IMPROVED == "improved"
```

**Step 2: Run test — verify it fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_enums.py -v
```
Expected: `ModuleNotFoundError`.

**Step 3: Implement**

```python
# eval/app_v2/engine/domain/enums.py
from enum import StrEnum


class DiagnosticCode(StrEnum):
    NO_CLEAR_FAILURE             = "no_clear_failure"
    RETRIEVAL_MISS               = "retrieval_miss"
    RETRIEVAL_PARTIAL            = "retrieval_partial"
    RERANK_DROPPED_RELEVANT      = "rerank_dropped_relevant"
    RERANK_DEGRADED_RANK         = "rerank_degraded_rank"
    PACKING_OMITTED_RELEVANT     = "packing_omitted_relevant"
    PACKING_TRUNCATED_RELEVANT   = "packing_truncated_relevant"
    GROUNDED_ANSWER              = "grounded_answer"
    UNSUPPORTED_ANSWER           = "unsupported_answer"
    BAD_ABSTAIN_ON_ANSWERABLE    = "bad_abstain_on_answerable"
    FAILED_ABSTAIN_ON_UNANSWERABLE = "failed_abstain_on_unanswerable"
    TRACE_MISSING                = "trace_missing"
    DATA_INSUFFICIENT            = "data_insufficient"


class Severity(StrEnum):
    OK       = "ok"
    MINOR    = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class RetrievalStatus(StrEnum):
    HIT     = "hit"
    PARTIAL = "partial"
    MISS    = "miss"
    UNKNOWN = "unknown"


class RerankStatus(StrEnum):
    IMPROVED = "improved"
    NEUTRAL  = "neutral"
    DEGRADED = "degraded"
    ABSENT   = "absent"
    UNKNOWN  = "unknown"


class PackingStatus(StrEnum):
    COMPLETE  = "complete"
    TRUNCATED = "truncated"
    OMITTED   = "omitted"
    ABSENT    = "absent"
    UNKNOWN   = "unknown"


class GenerationStatus(StrEnum):
    GROUNDED          = "grounded"
    UNSUPPORTED       = "unsupported"
    ABSTAINED         = "abstained"
    FAILED_TO_ABSTAIN = "failed_to_abstain"
    ABSENT            = "absent"
    UNKNOWN           = "unknown"


class DeltaDirection(StrEnum):
    IMPROVED     = "improved"
    REGRESSED    = "regressed"
    UNCHANGED    = "unchanged"
    INSUFFICIENT = "insufficient"


class ComparisonClassification(StrEnum):
    IMPROVED          = "improved"
    REGRESSED         = "regressed"
    MIXED             = "mixed"
    UNCHANGED         = "unchanged"
    INSUFFICIENT_DATA = "insufficient_data"
```

**Step 4: Run test — verify it passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_enums.py -v
```
Expected: all PASS.

**Step 5: Commit**

```bash
git add eval/app_v2/engine/domain/enums.py tests/eval/app_v2/engine/test_enums.py
git commit -m "feat(app-v2): add domain enums"
```

**Acceptance criteria:** All 8 enum classes importable. String values match spec exactly.

---

## Task 3: `engine/domain/warnings.py`

**Depends on:** Task 2

**Files:**
- Create: `eval/app_v2/engine/domain/warnings.py`
- Create: `tests/eval/app_v2/engine/test_warnings.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_warnings.py
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode


def test_bundle_warning_is_frozen():
    w = BundleWarning(code=BundleWarningCode.MISSING_TRACES, message="no traces")
    import dataclasses
    assert dataclasses.is_dataclass(w)
    try:
        w.message = "changed"  # type: ignore
        assert False, "should be frozen"
    except (AttributeError, TypeError):
        pass


def test_bundle_warning_optional_artifact():
    w = BundleWarning(code=BundleWarningCode.ORPHAN_TRACE, message="orphan", artifact_name="traces.jsonl")
    assert w.artifact_name == "traces.jsonl"

    w2 = BundleWarning(code=BundleWarningCode.MISSING_VERDICT, message="no verdict")
    assert w2.artifact_name is None
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_warnings.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/domain/warnings.py
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class BundleWarningCode(StrEnum):
    MISSING_TRACES           = "missing_traces"
    MISSING_VERDICT          = "missing_verdict"
    PARTIAL_TRACE_PARSE      = "partial_trace_parse"
    PARTIAL_RESULTS_PARSE    = "partial_results_parse"
    SCHEMA_VERSION_UNKNOWN   = "schema_version_unknown"
    TRACE_TEXT_REDACTED      = "trace_text_redacted"
    ORPHAN_TRACE             = "orphan_trace"
    MISSING_TRACE_FOR_RESULT = "missing_trace_for_result"


@dataclass(frozen=True, slots=True)
class BundleWarning:
    code: BundleWarningCode
    message: str
    artifact_name: str | None = None
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_warnings.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/domain/warnings.py tests/eval/app_v2/engine/test_warnings.py
git commit -m "feat(app-v2): add BundleWarning domain types"
```

**Acceptance criteria:** `BundleWarning` is frozen, has optional `artifact_name`, and all 8 warning codes exist.

---

## Task 4: `engine/domain/models.py`

**Depends on:** Tasks 2, 3

**Files:**
- Create: `eval/app_v2/engine/domain/models.py`
- Create: `tests/eval/app_v2/engine/test_models.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_models.py
import dataclasses
from datetime import datetime, UTC

from eval.app_v2.engine.domain.models import (
    QueryRecord,
    QueryDiagnostic,
    AnalyzedQuery,
    RunBundle,
    RunHealthSummary,
    RunConfig,
    SliceKey,
    SliceMetricRow,
    SliceMetricTable,
    VerdictSummary,
)
from eval.app_v2.engine.domain.enums import (
    DiagnosticCode, Severity, RetrievalStatus,
    RerankStatus, PackingStatus, GenerationStatus,
)


def _make_query_record() -> QueryRecord:
    return QueryRecord(
        qid="q1",
        query="what is X?",
        query_type="factual",
        difficulty="easy",
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1", "c2"),
        reranked_chunk_ids=None,
        packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 0.5},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None,
        answer_metrics=None,
        groundedness=None,
        latency_ms=None,
        trace_id=None,
        trace=None,
    )


def test_query_record_is_frozen():
    r = _make_query_record()
    assert dataclasses.is_dataclass(r)
    try:
        r.qid = "changed"  # type: ignore
        assert False
    except (AttributeError, TypeError):
        pass


def test_analyzed_query_pairs_record_and_diagnostic():
    record = _make_query_record()
    diag = QueryDiagnostic(
        qid="q1",
        diagnostic_code=DiagnosticCode.GROUNDED_ANSWER,
        severity=Severity.OK,
        retrieval_status=RetrievalStatus.HIT,
        rerank_status=RerankStatus.ABSENT,
        packing_status=PackingStatus.ABSENT,
        generation_status=GenerationStatus.GROUNDED,
        root_cause_summary="grounded answer with full retrieval",
        suggested_next_check=None,
        evidence_present=True,
        trace_available=False,
    )
    aq = AnalyzedQuery(record=record, diagnostic=diag)
    assert aq.record.qid == aq.diagnostic.qid


def test_run_config_frozen():
    cfg = RunConfig(
        retriever="HydratingRetriever",
        index_name="obsidian",
        reranker_model="heuristic_v1",
        reranker_top_n=None,
        generator_model=None,
        embedder_model="text-embedding-3-large",
        top_k=10,
        token_budget=1500,
    )
    assert cfg.top_k == 10
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_models.py -v
```

**Step 3: Implement**

Key types to define. Note `AnswerMetrics` and `GroundednessOutcome` are type aliases for existing rag.eval types — import them to avoid duplication.

```python
# eval/app_v2/engine/domain/models.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Mapping

from eval.app_v2.engine.domain.enums import (
    ComparisonClassification,
    DeltaDirection,
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.warnings import BundleWarning

# Re-use existing eval domain types rather than reimplementing
from rag.eval.answer_metrics import AnswerQualityMetrics as AnswerMetrics
from rag.eval.judges import GroundednessJudgeResult as GroundednessOutcome
from rag.eval.models import EvalAggregates
from rag.eval.verdict import Verdict


@dataclass(frozen=True, slots=True)
class RunConfig:
    """Normalized subset of EvalRunMeta used for config-change detection."""
    retriever: str | None
    index_name: str | None
    reranker_model: str | None
    reranker_top_n: int | None
    generator_model: str | None
    embedder_model: str | None
    top_k: int
    token_budget: int


@dataclass(frozen=True, slots=True)
class QueryTrace:
    """Normalized trace for a single query, joined from traces.jsonl."""
    trace_id: str
    reranked_chunk_ids: tuple[str, ...] | None
    packed_chunk_ids: tuple[str, ...] | None
    raw_data: dict[str, Any]


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
    retrieved_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...] | None
    packed_chunk_ids: tuple[str, ...] | None

    # Per-query metrics
    per_query_recall_at_k: Mapping[int, float]
    per_query_precision_at_k: Mapping[int, float]
    per_query_ndcg_at_k: Mapping[int, float]
    per_query_hit_rate_at_k: Mapping[int, float]

    # Generation
    answer_text: str | None
    answer_metrics: AnswerMetrics | None
    groundedness: GroundednessOutcome | None
    latency_ms: int | None

    # Trace
    trace_id: str | None
    trace: QueryTrace | None


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


@dataclass(frozen=True, slots=True)
class AnalyzedQuery:
    record: QueryRecord
    diagnostic: QueryDiagnostic


@dataclass(frozen=True, slots=True)
class SliceKey:
    parts: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SliceMetricRow:
    key: SliceKey
    size: int
    metrics: Mapping[str, float | None]


@dataclass(frozen=True, slots=True)
class SliceMetricTable:
    group_by: tuple[str, ...]
    rows: tuple[SliceMetricRow, ...]


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


@dataclass(frozen=True, slots=True)
class VerdictSummary:
    """Thin wrapper around Verdict for display in RunBundle."""
    decision: Literal["SHIP", "BLOCK"]
    failed_check_names: tuple[str, ...]
    raw: Verdict


@dataclass(frozen=True, slots=True)
class RunBundle:
    run_id: str
    display_name: str
    timestamp: datetime
    config: RunConfig
    aggregates: EvalAggregates
    queries: tuple[AnalyzedQuery, ...]
    health: RunHealthSummary
    verdict: VerdictSummary | None
    warnings: tuple[BundleWarning, ...]
    raw_artifacts: Mapping[str, object]


# ── Comparison models ─────────────────────────────────────────────────────────

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


# ── Trend models ──────────────────────────────────────────────────────────────

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

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_models.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/domain/models.py tests/eval/app_v2/engine/test_models.py
git commit -m "feat(app-v2): add all domain models"
```

**Acceptance criteria:** All dataclasses frozen. `AnalyzedQuery`, `RunBundle`, `ComparisonBundle`, `TrendBundle`, comparison and trend models all importable and constructible.

---

## Task 5: `engine/loaders/base.py`

**Depends on:** Task 3

**Files:**
- Create: `eval/app_v2/engine/loaders/base.py`
- Create: `tests/eval/app_v2/engine/test_loader_base.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_loader_base.py
from pathlib import Path
from eval.app_v2.engine.loaders.base import LoadedArtifact
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode


def test_loaded_artifact_no_warnings():
    a = LoadedArtifact(artifact_name="metrics.json", payload={"foo": 1}, warnings=())
    assert a.payload == {"foo": 1}
    assert a.warnings == ()


def test_loaded_artifact_with_warnings():
    w = BundleWarning(code=BundleWarningCode.SCHEMA_VERSION_UNKNOWN, message="unknown schema")
    a = LoadedArtifact(artifact_name="metrics.json", payload=None, warnings=(w,))
    assert len(a.warnings) == 1
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_loader_base.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/loaders/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from eval.app_v2.engine.domain.warnings import BundleWarning


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    artifact_name: str
    payload: Any
    warnings: tuple[BundleWarning, ...]


@runtime_checkable
class ArtifactLoader(Protocol):
    artifact_name: str

    def can_load(self, run_dir: Path) -> bool: ...
    def load(self, run_dir: Path) -> LoadedArtifact: ...
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_loader_base.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/loaders/base.py tests/eval/app_v2/engine/test_loader_base.py
git commit -m "feat(app-v2): add ArtifactLoader protocol and LoadedArtifact"
```

**Acceptance criteria:** `LoadedArtifact` is frozen. `ArtifactLoader` is a `runtime_checkable` Protocol.

---

## Tasks 6–9: Individual loaders (implement in parallel)

**Depends on:** Tasks 4, 5

**Files:**
- Create: `eval/app_v2/engine/loaders/metrics.py`
- Create: `eval/app_v2/engine/loaders/results.py`
- Create: `eval/app_v2/engine/loaders/traces.py`
- Create: `eval/app_v2/engine/loaders/verdict.py`
- Create: `tests/eval/app_v2/engine/test_loaders.py`

**Step 1: Inspect a real run directory before implementing**

```bash
./scripts/py -c "
import json
p = 'eval/runs/run_2026_02_12T20-40'
print(list(__import__('pathlib').Path(p).iterdir()))
"
```

Also inspect `metrics.json` top-level keys:

```bash
./scripts/py -c "
import json
d = json.load(open('eval/runs/run_2026_02_12T20-40/metrics.json'))
print(list(d.keys()))
"
```

**Step 2: Write the failing tests**

```python
# tests/eval/app_v2/engine/test_loaders.py
from pathlib import Path
import pytest

REAL_RUN = Path("eval/runs/run_2026_02_12T20-40")


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_metrics_loader_loads_aggregates():
    from eval.app_v2.engine.loaders.metrics import MetricsLoader
    from rag.eval.models import EvalAggregates
    loader = MetricsLoader()
    assert loader.can_load(REAL_RUN)
    artifact = loader.load(REAL_RUN)
    assert artifact.payload is not None
    agg, meta = artifact.payload
    assert isinstance(agg, EvalAggregates)
    assert meta.top_k >= 1


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_results_loader_loads_eval_results():
    from eval.app_v2.engine.loaders.results import ResultsLoader
    from rag.eval.models import EvalResult
    loader = ResultsLoader()
    assert loader.can_load(REAL_RUN)
    artifact = loader.load(REAL_RUN)
    results = artifact.payload
    assert isinstance(results, tuple)
    assert len(results) > 0
    assert isinstance(results[0], EvalResult)


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_traces_loader_absent_dir():
    from eval.app_v2.engine.loaders.traces import TracesLoader
    loader = TracesLoader()
    no_traces_dir = Path("eval/runs/run_2026_02_20T19-49")  # known to have no traces
    # can_load returns False when file absent
    if not (no_traces_dir / "traces.jsonl").exists():
        assert not loader.can_load(no_traces_dir)


@pytest.mark.skipif(not (REAL_RUN / "traces.jsonl").exists(), reason="no traces")
def test_traces_loader_loads_dict():
    from eval.app_v2.engine.loaders.traces import TracesLoader
    loader = TracesLoader()
    artifact = loader.load(REAL_RUN)
    traces = artifact.payload  # dict[trace_id, QueryTrace]
    assert isinstance(traces, dict)
```

**Step 3: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_loaders.py -v
```

**Step 4: Implement `metrics.py`**

```python
# eval/app_v2/engine/loaders/metrics.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import ArtifactLoader, LoadedArtifact
from rag.eval.models import EvalAggregates, EvalRunMeta

logger = logging.getLogger(__name__)


class MetricsLoader:
    artifact_name = "metrics.json"

    def can_load(self, run_dir: Path) -> bool:
        return (run_dir / "metrics.json").exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        try:
            data = json.loads((run_dir / "metrics.json").read_text())
            meta = EvalRunMeta.from_dict(data.get("meta", {}))  # type: ignore[attr-defined]
            aggregates = EvalAggregates.from_flat_dict(data)
            return LoadedArtifact(
                artifact_name=self.artifact_name,
                payload=(aggregates, meta),
                warnings=tuple(warnings),
            )
        except Exception as exc:
            warnings.append(BundleWarning(
                code=BundleWarningCode.PARTIAL_RESULTS_PARSE,
                message=f"Failed to parse metrics.json: {exc}",
                artifact_name=self.artifact_name,
            ))
            return LoadedArtifact(artifact_name=self.artifact_name, payload=None, warnings=tuple(warnings))
```

**Step 5: Implement `results.py`**

```python
# eval/app_v2/engine/loaders/results.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import ArtifactLoader, LoadedArtifact
from rag.eval.models import EvalResult

logger = logging.getLogger(__name__)


class ResultsLoader:
    artifact_name = "results.jsonl"

    def can_load(self, run_dir: Path) -> bool:
        return (run_dir / "results.jsonl").exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        results: list[EvalResult] = []
        path = run_dir / "results.jsonl"
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(EvalResult.from_results_dict(json.loads(line)))
            except Exception as exc:
                warnings.append(BundleWarning(
                    code=BundleWarningCode.PARTIAL_RESULTS_PARSE,
                    message=f"Row {i} parse error: {exc}",
                    artifact_name=self.artifact_name,
                ))
        return LoadedArtifact(
            artifact_name=self.artifact_name,
            payload=tuple(results),
            warnings=tuple(warnings),
        )
```

**Step 6: Implement `traces.py`**

The trace schema has `trace_id`, `retrieved_candidates`, `reranked_candidates`, `packed_chunk_ids` etc — inspect a real trace row to confirm field names before implementing.

```python
# eval/app_v2/engine/loaders/traces.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.models import QueryTrace
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import ArtifactLoader, LoadedArtifact

logger = logging.getLogger(__name__)


def _extract_reranked_ids(row: dict) -> tuple[str, ...] | None:
    candidates = row.get("reranked_candidates")
    if candidates is None:
        return None
    return tuple(c.get("chunk_id", c.get("id", "")) for c in candidates if c)


def _extract_packed_ids(row: dict) -> tuple[str, ...] | None:
    packed = row.get("packed_chunk_ids")
    if packed is None:
        return None
    return tuple(packed)


class TracesLoader:
    artifact_name = "traces.jsonl"

    def can_load(self, run_dir: Path) -> bool:
        return (run_dir / "traces.jsonl").exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        traces: dict[str, QueryTrace] = {}
        path = run_dir / "traces.jsonl"
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                tid = row.get("trace_id") or row.get("id")
                if not tid:
                    warnings.append(BundleWarning(
                        code=BundleWarningCode.ORPHAN_TRACE,
                        message=f"Row {i} has no trace_id",
                        artifact_name=self.artifact_name,
                    ))
                    continue
                traces[tid] = QueryTrace(
                    trace_id=tid,
                    reranked_chunk_ids=_extract_reranked_ids(row),
                    packed_chunk_ids=_extract_packed_ids(row),
                    raw_data=row,
                )
            except Exception as exc:
                warnings.append(BundleWarning(
                    code=BundleWarningCode.PARTIAL_TRACE_PARSE,
                    message=f"Row {i} parse error: {exc}",
                    artifact_name=self.artifact_name,
                ))
        return LoadedArtifact(
            artifact_name=self.artifact_name,
            payload=traces,
            warnings=tuple(warnings),
        )
```

**Step 7: Implement `verdict.py`**

The verdict file is at `eval/verdicts/verdict.json` (not inside the run dir). The loader checks a conventional path relative to the repo root.

```python
# eval/app_v2/engine/loaders/verdict.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.models import VerdictSummary
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import ArtifactLoader, LoadedArtifact
from rag.eval.verdict import verdict_from_dict

logger = logging.getLogger(__name__)

_DEFAULT_VERDICT_PATH = Path("eval/verdicts/verdict.json")


class VerdictLoader:
    artifact_name = "verdict.json"

    def __init__(self, verdict_path: Path = _DEFAULT_VERDICT_PATH) -> None:
        self._verdict_path = verdict_path

    def can_load(self, run_dir: Path) -> bool:
        # Verdict is run-agnostic; check the conventional path
        return self._verdict_path.exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        if not self._verdict_path.exists():
            warnings.append(BundleWarning(
                code=BundleWarningCode.MISSING_VERDICT,
                message=f"No verdict file at {self._verdict_path}",
            ))
            return LoadedArtifact(artifact_name=self.artifact_name, payload=None, warnings=tuple(warnings))
        try:
            data = json.loads(self._verdict_path.read_text())
            verdict = verdict_from_dict(data)
            decision_str = verdict.decision.value.upper()  # "SHIP" or "BLOCK"
            failed = tuple(c.name for c in verdict.checks if not c.passed)
            summary = VerdictSummary(
                decision=decision_str,  # type: ignore[arg-type]
                failed_check_names=failed,
                raw=verdict,
            )
            return LoadedArtifact(artifact_name=self.artifact_name, payload=summary, warnings=())
        except Exception as exc:
            warnings.append(BundleWarning(
                code=BundleWarningCode.MISSING_VERDICT,
                message=f"Verdict parse error: {exc}",
            ))
            return LoadedArtifact(artifact_name=self.artifact_name, payload=None, warnings=tuple(warnings))
```

**Step 8: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_loaders.py -v
```

**Step 9: Commit**

```bash
git add eval/app_v2/engine/loaders/ tests/eval/app_v2/engine/test_loaders.py
git commit -m "feat(app-v2): add MetricsLoader, ResultsLoader, TracesLoader, VerdictLoader"
```

**Acceptance criteria:** Each loader parses its artifact without crashing on real run data. Missing files produce `BundleWarning`, not exceptions.

---

## Task 10: `engine/loaders/registry.py`

**Depends on:** Task 5

**Files:**
- Create: `eval/app_v2/engine/loaders/registry.py`

**Step 1: Implement (no test needed — registry is trivial, tested implicitly via bundle)**

```python
# eval/app_v2/engine/loaders/registry.py
from __future__ import annotations

from eval.app_v2.engine.loaders.base import ArtifactLoader
from eval.app_v2.engine.loaders.metrics import MetricsLoader
from eval.app_v2.engine.loaders.results import ResultsLoader
from eval.app_v2.engine.loaders.traces import TracesLoader
from eval.app_v2.engine.loaders.verdict import VerdictLoader

DEFAULT_LOADERS: tuple[ArtifactLoader, ...] = (
    MetricsLoader(),
    ResultsLoader(),
    TracesLoader(),
    VerdictLoader(),
)
```

**Step 2: Commit**

```bash
git add eval/app_v2/engine/loaders/registry.py
git commit -m "feat(app-v2): add loader registry"
```

---

## Tasks 12–15: Derived-data layer (implement after Task 4, in parallel with Tasks 6–10)

> These 4 tasks depend only on Task 4 (domain models). They can be worked on in parallel with Tasks 6–10.

### Task 12: `engine/derived/stage_attribution.py`

**Depends on:** Task 4

**Files:**
- Create: `eval/app_v2/engine/derived/stage_attribution.py`
- Create: `tests/eval/app_v2/engine/test_stage_attribution.py`

**Step 1: Write the failing tests**

```python
# tests/eval/app_v2/engine/test_stage_attribution.py
from eval.app_v2.engine.derived.stage_attribution import classify_query
from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import QueryRecord


def _record(**kwargs) -> QueryRecord:
    defaults = dict(
        qid="q1", query="test", query_type=None, difficulty=None,
        is_unanswerable=False, requires_synthesis=False, tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",),
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )
    defaults.update(kwargs)
    return QueryRecord(**defaults)


def test_retrieval_miss():
    r = _record(relevant_chunk_ids=frozenset(["c1"]), retrieved_chunk_ids=("c2", "c3"))
    code, severity = classify_query(r)
    assert code == DiagnosticCode.RETRIEVAL_MISS
    assert severity == Severity.MODERATE


def test_grounded_answer():
    r = _record(relevant_chunk_ids=frozenset(["c1"]), retrieved_chunk_ids=("c1",))
    code, severity = classify_query(r)
    assert code == DiagnosticCode.GROUNDED_ANSWER
    assert severity == Severity.OK


def test_retrieval_partial():
    r = _record(
        relevant_chunk_ids=frozenset(["c1", "c2"]),
        retrieved_chunk_ids=("c1",),
        per_query_recall_at_k={10: 0.5},
    )
    code, severity = classify_query(r)
    assert code == DiagnosticCode.RETRIEVAL_PARTIAL
    assert severity == Severity.MINOR


def test_no_relevant_chunks_is_data_insufficient():
    r = _record(relevant_chunk_ids=frozenset())
    code, severity = classify_query(r)
    assert code == DiagnosticCode.DATA_INSUFFICIENT
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_stage_attribution.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/derived/stage_attribution.py
"""
Classify a QueryRecord into a (DiagnosticCode, Severity) pair.

Decision order (see design doc):
1. Data sufficiency
2. Unanswerable behavior
3. Retrieval
4. Rerank
5. Packing
6. Generation
7. Fallback
"""
from __future__ import annotations

from eval.app_v2.engine.domain.enums import (
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.models import QueryRecord

# Severity per code — single source of truth
_SEVERITY: dict[DiagnosticCode, Severity] = {
    DiagnosticCode.DATA_INSUFFICIENT:              Severity.MODERATE,
    DiagnosticCode.TRACE_MISSING:                  Severity.MINOR,
    DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE: Severity.CRITICAL,
    DiagnosticCode.BAD_ABSTAIN_ON_ANSWERABLE:      Severity.MODERATE,
    DiagnosticCode.RETRIEVAL_MISS:                 Severity.MODERATE,
    DiagnosticCode.RETRIEVAL_PARTIAL:              Severity.MINOR,
    DiagnosticCode.RERANK_DROPPED_RELEVANT:        Severity.MODERATE,
    DiagnosticCode.RERANK_DEGRADED_RANK:           Severity.MINOR,
    DiagnosticCode.PACKING_OMITTED_RELEVANT:       Severity.MODERATE,
    DiagnosticCode.PACKING_TRUNCATED_RELEVANT:     Severity.MODERATE,
    DiagnosticCode.UNSUPPORTED_ANSWER:             Severity.CRITICAL,
    DiagnosticCode.GROUNDED_ANSWER:                Severity.OK,
    DiagnosticCode.NO_CLEAR_FAILURE:               Severity.OK,
}


def _retrieval_hit_set(record: QueryRecord) -> frozenset[str]:
    return record.relevant_chunk_ids & frozenset(record.retrieved_chunk_ids)


def classify_query(record: QueryRecord) -> tuple[DiagnosticCode, Severity]:
    """Return (DiagnosticCode, Severity) for a single QueryRecord."""
    relevant = record.relevant_chunk_ids
    retrieved = frozenset(record.retrieved_chunk_ids)

    # 1. Data sufficiency
    if not relevant:
        code = DiagnosticCode.DATA_INSUFFICIENT
        return code, _SEVERITY[code]

    hits = relevant & retrieved

    # 2. Unanswerable behavior (requires generation data)
    if record.is_unanswerable and record.answer_text is not None:
        # answered when it should have abstained
        code = DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE
        return code, _SEVERITY[code]

    # 3. Retrieval
    if not hits:
        code = DiagnosticCode.RETRIEVAL_MISS
        return code, _SEVERITY[code]

    if len(hits) < len(relevant):
        code = DiagnosticCode.RETRIEVAL_PARTIAL
        return code, _SEVERITY[code]

    # 4. Rerank — check if relevant chunk was dropped
    if record.reranked_chunk_ids is not None:
        reranked_set = frozenset(record.reranked_chunk_ids)
        if hits and not (hits & reranked_set):
            code = DiagnosticCode.RERANK_DROPPED_RELEVANT
            return code, _SEVERITY[code]

    # 5. Packing — check if relevant survived rerank but lost in packing
    if record.packed_chunk_ids is not None:
        packed_set = frozenset(record.packed_chunk_ids)
        reranked_set = frozenset(record.reranked_chunk_ids) if record.reranked_chunk_ids else retrieved
        survived_rerank = hits & reranked_set
        if survived_rerank and not (survived_rerank & packed_set):
            code = DiagnosticCode.PACKING_OMITTED_RELEVANT
            return code, _SEVERITY[code]

    # 6. Generation — if groundedness says unsupported
    if record.groundedness is not None:
        gnd = record.groundedness
        # GroundednessJudgeResult has `supported` / `unsupported_claims` attributes
        # Check against the actual field — see src/rag/eval/judges.py
        if hasattr(gnd, "has_unsupported_claims") and gnd.has_unsupported_claims:
            code = DiagnosticCode.UNSUPPORTED_ANSWER
            return code, _SEVERITY[code]

    # 6b. Abstain check on answerable query
    if not record.is_unanswerable and record.answer_text is None and record.trace is not None:
        code = DiagnosticCode.BAD_ABSTAIN_ON_ANSWERABLE
        return code, _SEVERITY[code]

    # 7. Fallback — full retrieval, no failure detected
    if hits == relevant:
        code = DiagnosticCode.GROUNDED_ANSWER
        return code, _SEVERITY[code]

    code = DiagnosticCode.NO_CLEAR_FAILURE
    return code, _SEVERITY[code]


def derive_stage_statuses(
    record: QueryRecord, code: DiagnosticCode
) -> tuple[RetrievalStatus, RerankStatus, PackingStatus, GenerationStatus]:
    """Map DiagnosticCode back to per-stage status enums."""
    relevant = record.relevant_chunk_ids
    retrieved = frozenset(record.retrieved_chunk_ids)
    hits = relevant & retrieved

    if not relevant:
        return RetrievalStatus.UNKNOWN, RerankStatus.UNKNOWN, PackingStatus.UNKNOWN, GenerationStatus.UNKNOWN

    if not hits:
        ret = RetrievalStatus.MISS
    elif len(hits) < len(relevant):
        ret = RetrievalStatus.PARTIAL
    else:
        ret = RetrievalStatus.HIT

    if record.reranked_chunk_ids is None:
        rrk = RerankStatus.ABSENT
    elif hits and not (hits & frozenset(record.reranked_chunk_ids)):
        rrk = RerankStatus.DEGRADED
    else:
        rrk = RerankStatus.NEUTRAL

    if record.packed_chunk_ids is None:
        pck = PackingStatus.ABSENT
    elif hits and not (hits & frozenset(record.packed_chunk_ids)):
        pck = PackingStatus.OMITTED
    else:
        pck = PackingStatus.COMPLETE

    if record.answer_text is None:
        gen = GenerationStatus.ABSENT
    elif code == DiagnosticCode.UNSUPPORTED_ANSWER:
        gen = GenerationStatus.UNSUPPORTED
    elif code == DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE:
        gen = GenerationStatus.FAILED_TO_ABSTAIN
    else:
        gen = GenerationStatus.GROUNDED

    return ret, rrk, pck, gen
```

> **Important:** Before finalizing, inspect `src/rag/eval/judges.py` to confirm the exact attribute name for "has unsupported claims" on `GroundednessJudgeResult`. The implementation above uses `has_unsupported_claims` as a placeholder — replace with the real field.

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_stage_attribution.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/derived/stage_attribution.py tests/eval/app_v2/engine/test_stage_attribution.py
git commit -m "feat(app-v2): add stage attribution classifier"
```

**Acceptance criteria:** All 7 decision branches produce a deterministic `(DiagnosticCode, Severity)`. Severity mapping is in a single dict, not scattered.

---

### Task 13: `engine/derived/diagnostics.py`

**Depends on:** Tasks 4, 12

**Files:**
- Create: `eval/app_v2/engine/derived/diagnostics.py`
- Create: `tests/eval/app_v2/engine/test_diagnostics.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_diagnostics.py
from eval.app_v2.engine.derived.diagnostics import build_query_diagnostic, analyze_queries
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import QueryRecord

def _record(qid="q1", relevant=frozenset(["c1"]), retrieved=("c1",)):
    return QueryRecord(
        qid=qid, query="q", query_type=None, difficulty=None,
        is_unanswerable=False, requires_synthesis=False, tags=(),
        relevant_chunk_ids=relevant,
        retrieved_chunk_ids=retrieved,
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )


def test_build_query_diagnostic_returns_diagnostic():
    from eval.app_v2.engine.domain.models import QueryDiagnostic
    diag = build_query_diagnostic(_record())
    assert isinstance(diag, QueryDiagnostic)
    assert diag.qid == "q1"


def test_analyze_queries_returns_analyzed_queries():
    from eval.app_v2.engine.domain.models import AnalyzedQuery
    records = [_record("q1"), _record("q2", retrieved=("c2",))]
    analyzed = analyze_queries(records)
    assert len(analyzed) == 2
    assert all(isinstance(a, AnalyzedQuery) for a in analyzed)
    codes = {a.diagnostic.diagnostic_code for a in analyzed}
    assert DiagnosticCode.RETRIEVAL_MISS in codes
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_diagnostics.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/derived/diagnostics.py
from __future__ import annotations

from collections.abc import Sequence

from eval.app_v2.engine.derived.stage_attribution import classify_query, derive_stage_statuses
from eval.app_v2.engine.domain.models import AnalyzedQuery, QueryDiagnostic, QueryRecord

_ROOT_CAUSE: dict = {
    # Populated via a simple lookup table for human-readable summaries.
    # Import DiagnosticCode inline to avoid circular.
}


def _prose(code) -> tuple[str, str | None]:
    """Return (root_cause_summary, suggested_next_check) for a DiagnosticCode."""
    from eval.app_v2.engine.domain.enums import DiagnosticCode
    mapping = {
        DiagnosticCode.RETRIEVAL_MISS:               ("No relevant chunks retrieved", "Check embedder / index coverage"),
        DiagnosticCode.RETRIEVAL_PARTIAL:             ("Some relevant chunks missed at retrieval", "Increase top_k or check embedding quality"),
        DiagnosticCode.RERANK_DROPPED_RELEVANT:       ("Reranker dropped relevant chunks", "Inspect reranker scores for this query"),
        DiagnosticCode.RERANK_DEGRADED_RANK:          ("Reranker degraded rank of relevant chunks", "Review reranker model or heuristic weights"),
        DiagnosticCode.PACKING_OMITTED_RELEVANT:      ("Packing omitted relevant chunks within token budget", "Increase token budget or check packing order"),
        DiagnosticCode.PACKING_TRUNCATED_RELEVANT:    ("Token budget forced truncation of relevant content", "Increase token budget"),
        DiagnosticCode.UNSUPPORTED_ANSWER:            ("Generated answer not grounded in retrieved context", "Inspect citations and groundedness judge"),
        DiagnosticCode.GROUNDED_ANSWER:               ("Answer is grounded and retrieval succeeded", None),
        DiagnosticCode.BAD_ABSTAIN_ON_ANSWERABLE:     ("Model abstained despite evidence present", "Review generator prompt / abstain threshold"),
        DiagnosticCode.FAILED_ABSTAIN_ON_UNANSWERABLE:("Model answered an unanswerable question", "Review abstain instructions in prompt"),
        DiagnosticCode.TRACE_MISSING:                 ("Trace unavailable for this query", "Re-run with tracing enabled"),
        DiagnosticCode.DATA_INSUFFICIENT:             ("No relevant chunks defined; cannot diagnose", "Check query dataset annotations"),
        DiagnosticCode.NO_CLEAR_FAILURE:              ("No clear failure mode detected", None),
    }
    summary, suggestion = mapping.get(code, ("Unknown diagnostic", None))
    return summary, suggestion


def build_query_diagnostic(record: QueryRecord) -> QueryDiagnostic:
    code, severity = classify_query(record)
    ret_status, rrk_status, pck_status, gen_status = derive_stage_statuses(record, code)
    summary, suggestion = _prose(code)
    return QueryDiagnostic(
        qid=record.qid,
        diagnostic_code=code,
        severity=severity,
        retrieval_status=ret_status,
        rerank_status=rrk_status,
        packing_status=pck_status,
        generation_status=gen_status,
        root_cause_summary=summary,
        suggested_next_check=suggestion,
        evidence_present=bool(record.relevant_chunk_ids & frozenset(record.retrieved_chunk_ids)),
        trace_available=record.trace is not None,
    )


def analyze_queries(records: Sequence[QueryRecord]) -> tuple[AnalyzedQuery, ...]:
    return tuple(
        AnalyzedQuery(record=r, diagnostic=build_query_diagnostic(r))
        for r in records
    )
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_diagnostics.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/derived/diagnostics.py tests/eval/app_v2/engine/test_diagnostics.py
git commit -m "feat(app-v2): add build_query_diagnostic and analyze_queries"
```

---

### Task 14: `engine/derived/health.py`

**Depends on:** Task 13

**Files:**
- Create: `eval/app_v2/engine/derived/health.py`
- Create: `tests/eval/app_v2/engine/test_health.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_health.py
from eval.app_v2.engine.derived.health import build_health
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.models import QueryRecord, RunHealthSummary
from eval.app_v2.engine.domain.enums import Severity, DiagnosticCode


def _records():
    def r(qid, relevant, retrieved):
        return QueryRecord(
            qid=qid, query="q", query_type=None, difficulty=None,
            is_unanswerable=False, requires_synthesis=False, tags=(),
            relevant_chunk_ids=frozenset(relevant),
            retrieved_chunk_ids=tuple(retrieved),
            reranked_chunk_ids=None, packed_chunk_ids=None,
            per_query_recall_at_k={10: len(set(relevant) & set(retrieved)) / max(len(relevant), 1)},
            per_query_precision_at_k={10: 0.5},
            per_query_ndcg_at_k={10: 0.7},
            per_query_hit_rate_at_k={10: 1.0},
            answer_text=None, answer_metrics=None, groundedness=None,
            latency_ms=100, trace_id=None, trace=None,
        )
    return [r("q1", ["c1"], ["c1"]), r("q2", ["c2"], ["c3"])]


def test_build_health_returns_summary():
    analyzed = analyze_queries(_records())
    health = build_health(analyzed, recall_at_10=0.5, ndcg_at_10=0.7)
    assert isinstance(health, RunHealthSummary)
    assert health.severity_counts[Severity.MODERATE] >= 1
    assert health.dominant_failure_mode == DiagnosticCode.RETRIEVAL_MISS
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_health.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/derived/health.py
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery, RunHealthSummary


def build_health(
    analyzed: Sequence[AnalyzedQuery],
    recall_at_10: float,
    ndcg_at_10: float,
    verdict_status: Literal["SHIP", "BLOCK"] | None = None,
    worst_slice=None,
) -> RunHealthSummary:
    severity_counter: Counter[Severity] = Counter()
    code_counter: Counter[DiagnosticCode] = Counter()
    latencies: list[float] = []
    quality_scores: list[float] = []

    for aq in analyzed:
        severity_counter[aq.diagnostic.severity] += 1
        code_counter[aq.diagnostic.diagnostic_code] += 1
        if aq.record.latency_ms is not None:
            latencies.append(float(aq.record.latency_ms))
        if aq.record.answer_metrics is not None and hasattr(aq.record.answer_metrics, "quality_score"):
            qs = aq.record.answer_metrics.quality_score
            if qs is not None:
                quality_scores.append(float(qs))

    # Dominant failure = most common non-OK code
    failure_codes = {
        c: n for c, n in code_counter.items()
        if c not in (DiagnosticCode.GROUNDED_ANSWER, DiagnosticCode.NO_CLEAR_FAILURE)
    }
    dominant = max(failure_codes, key=failure_codes.get, default=None) if failure_codes else None

    return RunHealthSummary(
        headline_recall_at_10=recall_at_10,
        headline_ndcg_at_10=ndcg_at_10,
        avg_quality_score=sum(quality_scores) / len(quality_scores) if quality_scores else None,
        avg_latency_ms=sum(latencies) / len(latencies) if latencies else None,
        severity_counts=dict(severity_counter),
        diagnostic_counts=dict(code_counter),
        dominant_failure_mode=dominant,
        dominant_failure_summary=str(dominant) if dominant else None,
        worst_slice=worst_slice,
        verdict_status=verdict_status,
    )
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_health.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/derived/health.py tests/eval/app_v2/engine/test_health.py
git commit -m "feat(app-v2): add build_health"
```

---

### Task 15: `engine/derived/slices.py`

**Depends on:** Task 4

**Files:**
- Create: `eval/app_v2/engine/derived/slices.py`
- Create: `tests/eval/app_v2/engine/test_slices.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_slices.py
from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.models import QueryRecord, SliceMetricTable


def _r(qid, qtype, recall):
    return QueryRecord(
        qid=qid, query="q", query_type=qtype, difficulty=None,
        is_unanswerable=False, requires_synthesis=False, tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",),
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: recall},
        per_query_precision_at_k={10: recall},
        per_query_ndcg_at_k={10: recall},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )


def test_build_slice_table_groups_by_query_type():
    analyzed = analyze_queries([_r("q1", "factual", 1.0), _r("q2", "factual", 0.5), _r("q3", "conceptual", 0.0)])
    table = build_slice_table(analyzed, group_by=["query_type"])
    assert isinstance(table, SliceMetricTable)
    keys = [dict(r.key.parts)["query_type"] for r in table.rows]
    assert "factual" in keys
    assert "conceptual" in keys


def test_build_slice_table_multi_group():
    analyzed = analyze_queries([_r("q1", "factual", 1.0), _r("q2", "conceptual", 0.0)])
    table = build_slice_table(analyzed, group_by=["query_type", "difficulty"])
    assert isinstance(table, SliceMetricTable)
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_slices.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/derived/slices.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from typing import Any

from eval.app_v2.engine.domain.models import (
    AnalyzedQuery,
    SliceKey,
    SliceMetricRow,
    SliceMetricTable,
)


def _get_field(aq: AnalyzedQuery, field: str) -> str:
    """Extract a grouping field value from record or diagnostic."""
    val = getattr(aq.record, field, None)
    if val is None:
        return "__none__"
    return str(val)


def build_slice_table(
    queries: Sequence[AnalyzedQuery],
    group_by: Sequence[str],
) -> SliceMetricTable:
    groups: dict[tuple, list[AnalyzedQuery]] = defaultdict(list)
    for aq in queries:
        key_parts = tuple(_get_field(aq, f) for f in group_by)
        groups[key_parts].append(aq)

    rows: list[SliceMetricRow] = []
    for key_vals, members in groups.items():
        slice_key = SliceKey(parts=tuple(zip(group_by, key_vals)))
        recall_vals = [aq.record.per_query_recall_at_k.get(10) for aq in members]
        ndcg_vals = [aq.record.per_query_ndcg_at_k.get(10) for aq in members]
        metrics: dict[str, float | None] = {
            "recall@10": sum(v for v in recall_vals if v is not None) / len(recall_vals) if recall_vals else None,
            "ndcg@10":   sum(v for v in ndcg_vals if v is not None) / len(ndcg_vals) if ndcg_vals else None,
            "size":      float(len(members)),
        }
        rows.append(SliceMetricRow(key=slice_key, size=len(members), metrics=metrics))

    return SliceMetricTable(group_by=tuple(group_by), rows=tuple(rows))
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_slices.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/derived/slices.py tests/eval/app_v2/engine/test_slices.py
git commit -m "feat(app-v2): add build_slice_table"
```

---

## Task 11: `engine/loaders/bundle.py` — `build_bundle()`

**Depends on:** Tasks 6, 7, 8, 9, 10, 12, 13, 14, 15

**Files:**
- Create: `eval/app_v2/engine/loaders/bundle.py`
- Create: `tests/eval/app_v2/engine/test_bundle.py`

**Step 1: Write the failing test (integration test against real run dir)**

```python
# tests/eval/app_v2/engine/test_bundle.py
from pathlib import Path
import pytest

REAL_RUN = Path("eval/runs/run_2026_02_12T20-40")


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_build_bundle_produces_run_bundle():
    from eval.app_v2.engine.loaders.bundle import build_bundle
    from eval.app_v2.engine.domain.models import RunBundle

    bundle = build_bundle(REAL_RUN)
    assert isinstance(bundle, RunBundle)
    assert bundle.run_id
    assert len(bundle.queries) > 0
    assert bundle.health.headline_recall_at_10 >= 0.0


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_build_bundle_queries_have_diagnostics():
    from eval.app_v2.engine.loaders.bundle import build_bundle
    from eval.app_v2.engine.domain.models import AnalyzedQuery

    bundle = build_bundle(REAL_RUN)
    assert all(isinstance(q, AnalyzedQuery) for q in bundle.queries)
    assert all(q.diagnostic is not None for q in bundle.queries)
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_bundle.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/loaders/bundle.py
from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path

from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.derived.health import build_health
from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.domain.models import (
    QueryRecord,
    QueryTrace,
    RunBundle,
    RunConfig,
    VerdictSummary,
)
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.registry import DEFAULT_LOADERS
from rag.eval.models import EvalAggregates, EvalResult, EvalRunMeta

_RUN_DIR_PATTERN = re.compile(r"run_(\d{4}_\d{2}_\d{2}T\d{2}-\d{2})")


def _parse_timestamp(dirname: str) -> datetime:
    m = _RUN_DIR_PATTERN.match(dirname)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y_%m_%dT%H-%M").replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


def _normalize_config(meta: EvalRunMeta) -> RunConfig:
    return RunConfig(
        retriever=meta.extra.get("retriever_class") if meta.extra else None,
        index_name=meta.index_name,
        reranker_model=meta.reranker_name,
        reranker_top_n=meta.keep_k,
        generator_model=meta.generator_model,
        embedder_model=meta.embedder_model,
        top_k=meta.top_k,
        token_budget=meta.token_budget,
    )


def _build_query_record(
    result: EvalResult,
    traces: dict[str, QueryTrace],
) -> QueryRecord:
    rr = result.retrieval_result
    trace = traces.get(result.trace_id) if result.trace_id else None

    reranked = trace.reranked_chunk_ids if trace else None
    packed = trace.packed_chunk_ids if trace else None

    # Per-query metrics: compute from retrieval result
    relevant = frozenset(rr.relevant_chunk_ids)
    retrieved = tuple(rr.retrieved_chunk_ids)
    retrieved_set = frozenset(retrieved[:10])
    hits_at_10 = relevant & retrieved_set
    recall_10 = len(hits_at_10) / len(relevant) if relevant else 0.0
    hit_rate_10 = 1.0 if hits_at_10 else 0.0

    tags: tuple[str, ...] = ()
    requires_synthesis = False  # not in EvalResult; extend via result.extra if available

    return QueryRecord(
        qid=result.qid,
        query=result.query,
        query_type=result.query_type.value if result.query_type else None,
        difficulty=result.difficulty.value if result.difficulty else None,
        is_unanswerable=result.is_unanswerable,
        requires_synthesis=requires_synthesis,
        tags=tags,
        relevant_chunk_ids=relevant,
        retrieved_chunk_ids=retrieved,
        reranked_chunk_ids=reranked,
        packed_chunk_ids=packed,
        per_query_recall_at_k={10: recall_10},
        per_query_precision_at_k={10: len(hits_at_10) / 10 if retrieved else 0.0},
        per_query_ndcg_at_k={10: recall_10},  # simplified; replace with ndcg from result if available
        per_query_hit_rate_at_k={10: hit_rate_10},
        answer_text=result.answer.text if result.answer else None,
        answer_metrics=result.answer_metrics,
        groundedness=result.groundedness_result,
        latency_ms=result.latency_ms,
        trace_id=result.trace_id,
        trace=trace,
    )


def build_bundle(run_dir: Path) -> RunBundle:
    run_dir = run_dir.resolve()
    all_warnings: list[BundleWarning] = []
    raw_artifacts: dict[str, object] = {}

    # Run all loaders
    artifacts = {}
    for loader in DEFAULT_LOADERS:
        artifact = loader.load(run_dir) if loader.can_load(run_dir) else None
        if artifact is None:
            continue
        artifacts[loader.artifact_name] = artifact
        all_warnings.extend(artifact.warnings)
        raw_artifacts[loader.artifact_name] = artifact.payload

    # Unpack
    metrics_payload = artifacts.get("metrics.json")
    results_payload = artifacts.get("results.jsonl")
    traces_payload = artifacts.get("traces.jsonl")
    verdict_payload = artifacts.get("verdict.json")

    aggregates: EvalAggregates | None = None
    meta: EvalRunMeta | None = None
    if metrics_payload and metrics_payload.payload:
        aggregates, meta = metrics_payload.payload

    results: tuple[EvalResult, ...] = ()
    if results_payload and results_payload.payload:
        results = results_payload.payload

    traces: dict[str, QueryTrace] = {}
    if traces_payload and traces_payload.payload:
        traces = traces_payload.payload
    elif results:
        all_warnings.append(BundleWarning(
            code=BundleWarningCode.MISSING_TRACES,
            message="traces.jsonl not found; pipeline drill-down unavailable",
        ))

    verdict_summary: VerdictSummary | None = None
    if verdict_payload and verdict_payload.payload:
        verdict_summary = verdict_payload.payload

    # Build QueryRecords and analyze
    if meta is None:
        from rag.eval.models import EvalRunMeta as _Meta
        meta = _Meta()
    if aggregates is None:
        from rag.eval.models import EvalAggregates, RetrievalSummary
        aggregates = EvalAggregates(overall=RetrievalSummary(num_queries=0, avg_retrieved=0.0))

    records = [_build_query_record(r, traces) for r in results]
    analyzed = analyze_queries(records)

    recall_10 = aggregates.overall.recall_at_k.get(10, 0.0)
    ndcg_10 = aggregates.overall.ndcg_at_k.get(10, 0.0)
    slice_table = build_slice_table(analyzed, group_by=["query_type", "difficulty"])
    worst_slice = slice_table.rows[0].key if slice_table.rows else None

    verdict_flag = verdict_summary.decision if verdict_summary else None
    health = build_health(analyzed, recall_10, ndcg_10, verdict_status=verdict_flag, worst_slice=worst_slice)

    run_id = meta.run_id or run_dir.name
    timestamp = _parse_timestamp(run_dir.name)
    config = _normalize_config(meta)

    return RunBundle(
        run_id=run_id,
        display_name=meta.run_name or run_dir.name,
        timestamp=timestamp,
        config=config,
        aggregates=aggregates,
        queries=analyzed,
        health=health,
        verdict=verdict_summary,
        warnings=tuple(all_warnings),
        raw_artifacts=raw_artifacts,
    )
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_bundle.py -v
```

**Step 5: Run full engine test suite**

```bash
./scripts/py -m pytest tests/eval/app_v2/ -v
```
Expected: all PASS. Fix any failures before committing.

**Step 6: Commit**

```bash
git add eval/app_v2/engine/loaders/bundle.py tests/eval/app_v2/engine/test_bundle.py
git commit -m "feat(app-v2): add build_bundle — Phase 1 complete"
```

**Acceptance criteria (Phase 1 exit):** `build_bundle(Path("eval/runs/run_2026_02_12T20-40"))` returns a `RunBundle` with at least 1 `AnalyzedQuery`, a non-None `health`, and `health.headline_recall_at_10 >= 0`. All Phase 1 tests pass.

---

## Phase 1 validation

```bash
./scripts/py -m pytest tests/eval/app_v2/ -v
./scripts/py -m mypy eval/app_v2/engine/ --ignore-missing-imports
```
