# Results Analyzer v2 — Phase 4: Comparison and Verdict

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** App becomes useful for release decisions: compare two runs head-to-head and render the verdict gate.

**Architecture:** `engine/services/comparison.py` and `engine/services/forensics.py` do pure computation over `RunBundle` objects. Pages call these services then render results.

**Prerequisite:** Phase 3 complete.

**Parallel execution map:**
```
Phase 3 complete
  ├─ Task 22 (comparison service) ───────────────┐
  ├─ Task 23 (contributors derived) ─────────────┤
  └─ Task 24 (forensics service) ─────────────────┤
       ↓ (all three are independent of each other) │
  Task 25 (compare page) ◄── 22, 24              │
  Task 26 (verdicts page) ◄── 23                 │
```

---

## Task 22: `engine/services/comparison.py`

**Depends on:** Phase 1 (domain models)

**Files:**
- Create: `eval/app_v2/engine/services/comparison.py`
- Create: `tests/eval/app_v2/engine/test_comparison.py`

**Step 1: Write the failing tests**

```python
# tests/eval/app_v2/engine/test_comparison.py
from eval.app_v2.engine.domain.enums import (
    ComparisonClassification, DeltaDirection, DiagnosticCode, Severity,
    RetrievalStatus, RerankStatus, PackingStatus, GenerationStatus,
)
from eval.app_v2.engine.domain.models import QueryDiagnostic
from eval.app_v2.engine.services.comparison import (
    compare_diagnostics,
    classify_compared_query,
    RETRIEVAL_DELTA_THRESHOLD,
)


def _diag(code, severity, retrieval=RetrievalStatus.HIT):
    return QueryDiagnostic(
        qid="q1",
        diagnostic_code=code,
        severity=severity,
        retrieval_status=retrieval,
        rerank_status=RerankStatus.ABSENT,
        packing_status=PackingStatus.ABSENT,
        generation_status=GenerationStatus.GROUNDED,
        root_cause_summary="",
        suggested_next_check=None,
        evidence_present=True,
        trace_available=False,
    )


def test_improved_retrieval_delta():
    from eval.app_v2.engine.domain.models import QueryDeltaSummary
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.0,
        recall_after=1.0,
    )
    assert delta.retrieval == DeltaDirection.IMPROVED
    assert delta.severity == DeltaDirection.IMPROVED


def test_regressed_classification():
    from eval.app_v2.engine.domain.models import QueryDeltaSummary
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE),
        recall_before=1.0,
        recall_after=0.0,
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.RETRIEVAL_MISS, Severity.MODERATE))
    assert result == ComparisonClassification.REGRESSED


def test_unchanged_within_threshold():
    from eval.app_v2.engine.domain.models import QueryDeltaSummary
    delta = compare_diagnostics(
        diag_before=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK),
        recall_before=0.80,
        recall_after=0.82,  # < RETRIEVAL_DELTA_THRESHOLD
    )
    result = classify_compared_query(delta, diag_after=_diag(DiagnosticCode.GROUNDED_ANSWER, Severity.OK))
    assert result == ComparisonClassification.UNCHANGED
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_comparison.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/services/comparison.py
from __future__ import annotations

from collections.abc import Sequence

from eval.app_v2.engine.domain.enums import (
    ComparisonClassification,
    DeltaDirection,
    DiagnosticCode,
    Severity,
)
from eval.app_v2.engine.domain.models import (
    AnalyzedQuery,
    ComparedQuery,
    ComparisonBundle,
    QueryDeltaSummary,
    QueryDiagnostic,
    RunBundle,
    SliceMetricTable,
)
from eval.app_v2.engine.derived.slices import build_slice_table

# Materiality thresholds
RETRIEVAL_DELTA_THRESHOLD = 0.05   # recall or ndcg delta to count as material
LATENCY_DELTA_MS_THRESHOLD = 100.0
LATENCY_DELTA_PCT_THRESHOLD = 0.10

# Severity ordering for delta classification
_SEV_ORDER = {Severity.OK: 0, Severity.MINOR: 1, Severity.MODERATE: 2, Severity.CRITICAL: 3}


def _severity_direction(before: Severity, after: Severity) -> DeltaDirection:
    diff = _SEV_ORDER[after] - _SEV_ORDER[before]
    if diff < 0:
        return DeltaDirection.IMPROVED
    if diff > 0:
        return DeltaDirection.REGRESSED
    return DeltaDirection.UNCHANGED


def compare_diagnostics(
    *,
    diag_before: QueryDiagnostic | None,
    diag_after: QueryDiagnostic | None,
    recall_before: float | None = None,
    recall_after: float | None = None,
    ndcg_before: float | None = None,
    ndcg_after: float | None = None,
    latency_before: int | None = None,
    latency_after: int | None = None,
) -> QueryDeltaSummary:
    # Retrieval direction
    if recall_before is not None and recall_after is not None:
        delta = recall_after - recall_before
        if abs(delta) < RETRIEVAL_DELTA_THRESHOLD:
            ret_dir = DeltaDirection.UNCHANGED
        elif delta > 0:
            ret_dir = DeltaDirection.IMPROVED
        else:
            ret_dir = DeltaDirection.REGRESSED
    else:
        ret_dir = DeltaDirection.INSUFFICIENT

    # Groundedness direction (severity is a proxy if no direct groundedness delta)
    if diag_before and diag_after:
        sev_dir = _severity_direction(diag_before.severity, diag_after.severity)
        gnd_dir = sev_dir  # simplified: severity captures grounding degradation
    else:
        sev_dir = gnd_dir = DeltaDirection.INSUFFICIENT

    # Latency direction
    if latency_before is not None and latency_after is not None:
        lat_delta_ms = latency_after - latency_before
        lat_delta_pct = abs(lat_delta_ms) / max(latency_before, 1)
        if abs(lat_delta_ms) < LATENCY_DELTA_MS_THRESHOLD and lat_delta_pct < LATENCY_DELTA_PCT_THRESHOLD:
            lat_dir = DeltaDirection.UNCHANGED
        elif lat_delta_ms < 0:
            lat_dir = DeltaDirection.IMPROVED
        else:
            lat_dir = DeltaDirection.REGRESSED
    else:
        lat_dir = DeltaDirection.INSUFFICIENT

    return QueryDeltaSummary(
        retrieval=ret_dir,
        groundedness=gnd_dir,
        latency=lat_dir,
        severity=sev_dir,
    )


def classify_compared_query(
    delta: QueryDeltaSummary,
    *,
    diag_after: QueryDiagnostic | None = None,
) -> ComparisonClassification:
    dims = [delta.retrieval, delta.groundedness, delta.latency, delta.severity]
    material = [d for d in dims if d != DeltaDirection.INSUFFICIENT]

    if not material:
        return ComparisonClassification.INSUFFICIENT_DATA

    improvements = [d for d in material if d == DeltaDirection.IMPROVED]
    regressions  = [d for d in material if d == DeltaDirection.REGRESSED]

    # Severity override: CRITICAL regression dominates
    if diag_after and diag_after.severity == Severity.CRITICAL and delta.severity == DeltaDirection.REGRESSED:
        return ComparisonClassification.REGRESSED

    if not improvements and not regressions:
        return ComparisonClassification.UNCHANGED
    if improvements and not regressions:
        return ComparisonClassification.IMPROVED
    if regressions and not improvements:
        return ComparisonClassification.REGRESSED
    return ComparisonClassification.MIXED


def _index_queries(bundle: RunBundle) -> dict[str, AnalyzedQuery]:
    return {aq.record.qid: aq for aq in bundle.queries}


def build_comparison(run_a: RunBundle, run_b: RunBundle) -> ComparisonBundle:
    """Compare run_b against run_a (b = after, a = before)."""
    index_a = _index_queries(run_a)
    index_b = _index_queries(run_b)
    all_qids = sorted(set(index_a) | set(index_b))

    compared: list[ComparedQuery] = []
    for qid in all_qids:
        aq_a = index_a.get(qid)
        aq_b = index_b.get(qid)

        recall_a = aq_a.record.per_query_recall_at_k.get(10) if aq_a else None
        recall_b = aq_b.record.per_query_recall_at_k.get(10) if aq_b else None
        ndcg_a   = aq_a.record.per_query_ndcg_at_k.get(10)   if aq_a else None
        ndcg_b   = aq_b.record.per_query_ndcg_at_k.get(10)   if aq_b else None
        lat_a    = aq_a.record.latency_ms if aq_a else None
        lat_b    = aq_b.record.latency_ms if aq_b else None

        delta_summary = compare_diagnostics(
            diag_before=aq_a.diagnostic if aq_a else None,
            diag_after=aq_b.diagnostic if aq_b else None,
            recall_before=recall_a, recall_after=recall_b,
            ndcg_before=ndcg_a, ndcg_after=ndcg_b,
            latency_before=lat_a, latency_after=lat_b,
        )
        classification = classify_compared_query(
            delta_summary,
            diag_after=aq_b.diagnostic if aq_b else None,
        )

        query_text = (aq_b or aq_a).record.query if (aq_b or aq_a) else ""

        compared.append(ComparedQuery(
            qid=qid,
            query=query_text,
            retrieval_delta=recall_b - recall_a if (recall_a is not None and recall_b is not None) else None,
            ndcg_delta=ndcg_b - ndcg_a if (ndcg_a is not None and ndcg_b is not None) else None,
            latency_delta_ms=float(lat_b - lat_a) if (lat_a is not None and lat_b is not None) else None,
            quality_delta=None,  # extend when quality score is available on both sides
            diagnostic_before=aq_a.diagnostic if aq_a else None,
            diagnostic_after=aq_b.diagnostic if aq_b else None,
            delta_summary=delta_summary,
            classification=classification,
        ))

    # Aggregate deltas
    agg_a = run_a.aggregates.overall
    agg_b = run_b.aggregates.overall
    agg_deltas: dict[str, float | None] = {}
    for k in (5, 10):
        r_a = agg_a.recall_at_k.get(k)
        r_b = agg_b.recall_at_k.get(k)
        agg_deltas[f"recall@{k}"] = r_b - r_a if (r_a is not None and r_b is not None) else None
        n_a = agg_a.ndcg_at_k.get(k)
        n_b = agg_b.ndcg_at_k.get(k)
        agg_deltas[f"ndcg@{k}"] = n_b - n_a if (n_a is not None and n_b is not None) else None

    return ComparisonBundle(
        run_a=run_a,
        run_b=run_b,
        aggregate_deltas=agg_deltas,
        slice_deltas=None,  # extend in Phase 5
        compared_queries=tuple(compared),
    )
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_comparison.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/services/comparison.py tests/eval/app_v2/engine/test_comparison.py
git commit -m "feat(app-v2): add comparison service"
```

**Acceptance criteria:** `build_comparison(run_a, run_b)` returns `ComparisonBundle`. Classification handles all 5 cases (`IMPROVED`, `REGRESSED`, `MIXED`, `UNCHANGED`, `INSUFFICIENT_DATA`). Critical-severity regression overrides `MIXED`.

---

## Task 23: `engine/derived/contributors.py`

**Depends on:** Phase 1 (domain models)

**Files:**
- Create: `eval/app_v2/engine/derived/contributors.py`
- Create: `tests/eval/app_v2/engine/test_contributors.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_contributors.py
from eval.app_v2.engine.derived.contributors import contributor_queries_for_code
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import QueryRecord


def _r(qid, relevant, retrieved):
    return QueryRecord(
        qid=qid, query="q", query_type=None, difficulty=None,
        is_unanswerable=False, requires_synthesis=False, tags=(),
        relevant_chunk_ids=frozenset(relevant),
        retrieved_chunk_ids=tuple(retrieved),
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: 0.0},
        per_query_precision_at_k={10: 0.0},
        per_query_ndcg_at_k={10: 0.0},
        per_query_hit_rate_at_k={10: 0.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )


def test_contributor_queries_for_retrieval_miss():
    analyzed = analyze_queries([
        _r("q1", ["c1"], ["c2"]),  # miss
        _r("q2", ["c2"], ["c2"]),  # hit
        _r("q3", ["c3"], ["c4"]),  # miss
    ])
    contributors = contributor_queries_for_code(analyzed, DiagnosticCode.RETRIEVAL_MISS, limit=10)
    assert len(contributors) == 2
    qids = {a.record.qid for a in contributors}
    assert qids == {"q1", "q3"}
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_contributors.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/derived/contributors.py
from __future__ import annotations

from collections.abc import Sequence

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery

_SEV_ORDER = {Severity.OK: 0, Severity.MINOR: 1, Severity.MODERATE: 2, Severity.CRITICAL: 3}


def contributor_queries_for_code(
    analyzed: Sequence[AnalyzedQuery],
    code: DiagnosticCode,
    *,
    limit: int = 20,
) -> tuple[AnalyzedQuery, ...]:
    """Return queries matching a DiagnosticCode, sorted by severity descending."""
    matching = [aq for aq in analyzed if aq.diagnostic.diagnostic_code == code]
    matching.sort(key=lambda aq: _SEV_ORDER[aq.diagnostic.severity], reverse=True)
    return tuple(matching[:limit])


def worst_queries(
    analyzed: Sequence[AnalyzedQuery],
    *,
    limit: int = 10,
) -> tuple[AnalyzedQuery, ...]:
    """Return queries sorted by severity descending."""
    sorted_qs = sorted(analyzed, key=lambda aq: _SEV_ORDER[aq.diagnostic.severity], reverse=True)
    return tuple(sorted_qs[:limit])
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_contributors.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/derived/contributors.py tests/eval/app_v2/engine/test_contributors.py
git commit -m "feat(app-v2): add contributor query attribution"
```

---

## Task 24: `engine/services/forensics.py`

**Depends on:** Phase 1 (domain models)

**Files:**
- Create: `eval/app_v2/engine/services/forensics.py`
- Create: `tests/eval/app_v2/engine/test_forensics_service.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_forensics_service.py
from pathlib import Path
import pytest

REAL_RUN = Path("eval/runs/run_2026_02_12T20-40")


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_get_query_returns_analyzed_query():
    from eval.app_v2.engine.loaders.bundle import build_bundle
    from eval.app_v2.engine.services.forensics import get_query, list_queries_by_code
    from eval.app_v2.engine.domain.enums import DiagnosticCode

    bundle = build_bundle(REAL_RUN)
    first_qid = bundle.queries[0].record.qid

    aq = get_query(bundle, first_qid)
    assert aq is not None
    assert aq.record.qid == first_qid

    misses = list_queries_by_code(bundle, DiagnosticCode.RETRIEVAL_MISS)
    # just verify it returns a tuple without error
    assert isinstance(misses, tuple)
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_forensics_service.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/services/forensics.py
"""
Navigation and selection over already-derived diagnostics.
Does NOT construct new diagnoses.
"""
from __future__ import annotations

from eval.app_v2.engine.derived.contributors import worst_queries
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import AnalyzedQuery, RunBundle, SliceKey


def get_query(bundle: RunBundle, qid: str) -> AnalyzedQuery | None:
    for aq in bundle.queries:
        if aq.record.qid == qid:
            return aq
    return None


def list_queries_by_code(bundle: RunBundle, code: DiagnosticCode) -> tuple[AnalyzedQuery, ...]:
    return tuple(aq for aq in bundle.queries if aq.diagnostic.diagnostic_code == code)


def list_queries_by_slice(bundle: RunBundle, slice_key: SliceKey) -> tuple[AnalyzedQuery, ...]:
    """Return queries whose record fields match all parts of a SliceKey."""
    def matches(aq: AnalyzedQuery) -> bool:
        for field, value in slice_key.parts:
            if value == "__none__":
                if getattr(aq.record, field, None) is not None:
                    return False
            elif str(getattr(aq.record, field, None)) != value:
                return False
        return True
    return tuple(aq for aq in bundle.queries if matches(aq))


def get_worst_queries(bundle: RunBundle, *, limit: int = 10) -> tuple[AnalyzedQuery, ...]:
    return worst_queries(bundle.queries, limit=limit)


def contributor_queries_for_failure_mode(
    bundle: RunBundle, code: DiagnosticCode, *, limit: int = 20
) -> tuple[AnalyzedQuery, ...]:
    from eval.app_v2.engine.derived.contributors import contributor_queries_for_code
    return contributor_queries_for_code(bundle.queries, code, limit=limit)
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_forensics_service.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/services/forensics.py tests/eval/app_v2/engine/test_forensics_service.py
git commit -m "feat(app-v2): add forensics navigation service"
```

---

## Task 25: `ui/pages/compare.py`

**Depends on:** Tasks 22, 24, and the ui shell (Task 16)

**Files:**
- Create: `eval/app_v2/ui/pages/compare.py`

**Step 1: Update `app.py` entry point**

Add a "Compare" entry to `PAGES` and a second run selector in the sidebar (run_b).

**Step 2: Implement**

```python
# eval/app_v2/ui/pages/compare.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import ComparisonClassification
from eval.app_v2.engine.domain.models import ComparisonBundle, RunBundle
from eval.app_v2.engine.services.comparison import build_comparison

_CLASS_COLORS = {
    ComparisonClassification.IMPROVED:         "🟢",
    ComparisonClassification.REGRESSED:        "🔴",
    ComparisonClassification.MIXED:            "🟡",
    ComparisonClassification.UNCHANGED:        "⚪",
    ComparisonClassification.INSUFFICIENT_DATA:"❓",
}


def _render_aggregate_deltas(cb: ComparisonBundle) -> None:
    st.subheader("Aggregate deltas (B − A)")
    cols = st.columns(len(cb.aggregate_deltas) or 1)
    for col, (metric, delta) in zip(cols, cb.aggregate_deltas.items()):
        if delta is None:
            col.metric(metric, "—")
        else:
            col.metric(metric, f"{delta:+.1%}")


def _render_compared_queries(cb: ComparisonBundle, filter_class: ComparisonClassification | None) -> None:
    queries = cb.compared_queries
    if filter_class:
        queries = tuple(q for q in queries if q.classification == filter_class)

    st.markdown(f"**{len(queries)} queries** matching filter")
    for cq in queries[:50]:  # cap display at 50
        badge = _CLASS_COLORS.get(cq.classification, "")
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([0.05, 0.5, 0.15, 0.15, 0.15])
            c1.markdown(badge)
            c2.markdown(f"`{cq.qid}` — {cq.query[:60]}")
            c3.caption(f"Recall Δ: {cq.retrieval_delta:+.2f}" if cq.retrieval_delta is not None else "Recall Δ: —")
            c4.caption(f"NDCG Δ: {cq.ndcg_delta:+.2f}" if cq.ndcg_delta is not None else "NDCG Δ: —")
            c5.caption(f"Lat Δ: {cq.latency_delta_ms:+.0f}ms" if cq.latency_delta_ms is not None else "Lat Δ: —")


def render(bundle_a: RunBundle | None, bundle_b: RunBundle | None = None) -> None:
    st.header("Compare")

    if bundle_a is None or bundle_b is None:
        st.info("Select two runs (A and B) from the sidebar to compare.")
        return

    cb = build_comparison(bundle_a, bundle_b)

    st.markdown(f"**A:** `{bundle_a.display_name}` → **B:** `{bundle_b.display_name}`")
    _render_aggregate_deltas(cb)
    st.divider()

    filter_opts = ["All"] + [c.value for c in ComparisonClassification]
    choice = st.selectbox("Filter by classification", filter_opts)
    filter_class = None if choice == "All" else ComparisonClassification(choice)
    _render_compared_queries(cb, filter_class)
```

**Note:** The Compare page needs **two** bundles. Update `app.py` to expose a second run selector (`selected_b`) and pass both bundles to the Compare page:

```python
# In app.py, add to sidebar:
selected_b = run_selector_widget(runs, key="run_b", label="Compare to (B)")
bundle_b = load_bundle(*selected_b) if selected_b else None

# In PAGES dispatch:
PAGES["Compare"](bundle, bundle_b)  # compare page takes two args
```

**Step 3: Commit**

```bash
git add eval/app_v2/ui/pages/compare.py eval/app_v2/app.py
git commit -m "feat(app-v2): add Compare page"
```

**Acceptance criteria:** Compare page shows aggregate deltas, per-query comparison list filterable by `ComparisonClassification`. `MIXED` classification shows the per-dimension breakdown.

---

## Task 26: `ui/pages/verdicts.py`

**Depends on:** Task 23 (contributors), Task 16 (ui shell)

**Files:**
- Create: `eval/app_v2/ui/pages/verdicts.py`

**Step 1: Implement**

```python
# eval/app_v2/ui/pages/verdicts.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.derived.contributors import contributor_queries_for_code
from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_card


def render(bundle: RunBundle | None) -> None:
    st.header("Verdict")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    v = bundle.verdict
    if v is None:
        st.warning("No verdict file found for this run. Run `make verdict` to generate one.")
        return

    # SHIP/BLOCK badge
    if v.decision == "SHIP":
        st.success(f"## ✅ SHIP")
    else:
        st.error(f"## 🚫 BLOCK")

    st.divider()

    # Failed checks
    raw = v.raw
    st.subheader(f"Threshold checks ({len(raw.checks)} total)")
    for check in raw.checks:
        icon = "✅" if check.passed else "❌"
        delta = f" (baseline: {check.baseline:.3f})" if check.baseline is not None else ""
        st.markdown(
            f"{icon} **{check.name}** — current: `{check.current:.3f}` / threshold: `{check.threshold:.3f}`{delta}"
        )

    st.divider()

    # Contributor queries for each failed check
    failed_names = set(v.failed_check_names)
    if failed_names:
        st.subheader("Contributing queries (worst per failure mode)")
        # Map failed check names to DiagnosticCodes heuristically
        # Exact mapping depends on check name conventions in verdict.py
        # For now, show worst queries regardless of code
        from eval.app_v2.engine.derived.contributors import worst_queries
        worst = worst_queries(bundle.queries, limit=10)
        for aq in worst:
            render_diagnostic_card(aq, show_forensics_link=True)
```

**Step 2: Update `app.py` to import verdicts page**

**Step 3: Commit**

```bash
git add eval/app_v2/ui/pages/verdicts.py
git commit -m "feat(app-v2): add Verdicts page"
```

**Acceptance criteria:** Verdict page shows SHIP/BLOCK badge, all threshold checks with pass/fail, and top contributing queries with Forensics links.

---

## Phase 4 validation

```bash
./scripts/py -m pytest tests/eval/app_v2/ -v
./scripts/py -m streamlit run eval/app_v2/app.py
```

Manually verify:
1. Compare page with two real runs shows per-query classification counts
2. Verdicts page shows correct decision from `eval/verdicts/verdict.json`
3. Verdict BLOCK state lists all failed checks

**Phase 4 exit criterion:** The app is useful for release decisions.
