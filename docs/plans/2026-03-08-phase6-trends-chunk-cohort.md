# Phase 6: Trends, Chunk Analysis, Correlation & Cohort Views — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement Phase 6 of the Results Analyzer v2: multi-run trend analysis, chunk-level problem detection, latency/quality correlation charts, and cohort (slice) analysis.

**Architecture:** All analysis logic lives in `engine/` with zero Streamlit imports. The `ui/` layer renders typed engine objects. The `TrendBundle`, `ConfigChangeEvent`, and `ConfigFieldChange` domain models are **already defined** in `engine/domain/models.py` — Phase 6 fills in the service logic and wires the UI. The `Trends` page requires a different call signature than other pages (`runs: list[tuple[str, Path]]` instead of `RunBundle | None`) because it needs multi-run selection.

**Tech Stack:** Python dataclasses, Streamlit (`st.line_chart`, `st.scatter_chart`, `st.dataframe`), pytest

**Key file locations:**
- Engine services: `eval/app_v2/engine/services/`
- Engine derived: `eval/app_v2/engine/derived/`
- UI pages: `eval/app_v2/ui/pages/`
- UI widgets: `eval/app_v2/ui/widgets/`
- App shell: `eval/app_v2/app.py`
- Tests: `tests/eval/app_v2/engine/`

**Test runner:** `./scripts/py -m pytest <path> -v`
**Type check:** `make typecheck`
**Never use:** bare `python`, `pytest`, or `streamlit` commands directly.

---

## Task 1: `detect_config_change_events` in `engine/services/trend.py`

**Files:**
- Create: `eval/app_v2/engine/services/trend.py`
- Create: `tests/eval/app_v2/engine/test_trend_service.py`

**Acceptance criteria:**
- `detect_config_change_events` returns an empty tuple when configs are identical across runs.
- When `top_k` changes between adjacent runs, a `ConfigChangeEvent` is returned with a `ConfigFieldChange` entry for `top_k`.
- A single-run input always returns an empty tuple.
- Only fields in the tracked set are reported (no irrelevant keys leaked).

---

### Step 1: Write the failing test

Create `tests/eval/app_v2/engine/test_trend_service.py`:

```python
from datetime import UTC, datetime, timedelta

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import (
    RunBundle,
    RunConfig,
    RunHealthSummary,
)
from eval.app_v2.engine.services.trend import detect_config_change_events


def _make_config(**kwargs) -> RunConfig:
    defaults = dict(
        retriever="dense",
        index_name="idx",
        reranker_model=None,
        reranker_top_n=None,
        generator_model="gpt-4",
        embedder_model="ada-002",
        top_k=10,
        token_budget=4000,
    )
    defaults.update(kwargs)
    return RunConfig(**defaults)


def _make_health(recall: float = 0.8) -> RunHealthSummary:
    return RunHealthSummary(
        headline_recall_at_10=recall,
        headline_ndcg_at_10=recall,
        avg_quality_score=None,
        avg_latency_ms=None,
        severity_counts={Severity.OK: 10},
        diagnostic_counts={DiagnosticCode.GROUNDED_ANSWER: 10},
        dominant_failure_mode=None,
        dominant_failure_summary=None,
        worst_slice=None,
        verdict_status=None,
    )


def _make_bundle(
    run_id: str,
    ts: datetime,
    config: RunConfig | None = None,
    recall: float = 0.8,
) -> RunBundle:
    from rag.eval.models import EvalAggregates, RetrievalSummary
    return RunBundle(
        run_id=run_id,
        display_name=run_id,
        timestamp=ts,
        config=config or _make_config(),
        aggregates=EvalAggregates(
            overall=RetrievalSummary(num_queries=10, avg_retrieved=10.0)
        ),
        queries=(),
        health=_make_health(recall),
        verdict=None,
        warnings=(),
        raw_artifacts={},
    )


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_no_change_produces_empty_events():
    runs = [
        _make_bundle("r1", _T0),
        _make_bundle("r2", _T0 + timedelta(days=1)),
    ]
    assert detect_config_change_events(runs) == ()


def test_top_k_change_detected():
    runs = [
        _make_bundle("r1", _T0, _make_config(top_k=10)),
        _make_bundle("r2", _T0 + timedelta(days=1), _make_config(top_k=20)),
    ]
    events = detect_config_change_events(runs)
    assert len(events) == 1
    assert events[0].from_run_id == "r1"
    assert events[0].to_run_id == "r2"
    change_fields = {c.field_name for c in events[0].changes}
    assert "top_k" in change_fields


def test_single_run_produces_no_events():
    assert detect_config_change_events([_make_bundle("r1", _T0)]) == ()


def test_custom_tracked_fields_ignores_others():
    runs = [
        _make_bundle("r1", _T0, _make_config(top_k=10, generator_model="gpt-3")),
        _make_bundle("r2", _T0 + timedelta(days=1), _make_config(top_k=20, generator_model="gpt-4")),
    ]
    # Only track generator_model — top_k change should be ignored
    events = detect_config_change_events(runs, tracked_fields={"generator_model"})
    assert len(events) == 1
    change_fields = {c.field_name for c in events[0].changes}
    assert "generator_model" in change_fields
    assert "top_k" not in change_fields


def test_three_runs_two_changes():
    runs = [
        _make_bundle("r1", _T0, _make_config(top_k=10)),
        _make_bundle("r2", _T0 + timedelta(days=1), _make_config(top_k=20)),
        _make_bundle("r3", _T0 + timedelta(days=2), _make_config(top_k=30)),
    ]
    events = detect_config_change_events(runs)
    assert len(events) == 2
    assert events[0].from_run_id == "r1"
    assert events[1].from_run_id == "r2"
```

### Step 2: Run to verify it fails

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_trend_service.py -v
```

Expected: `ImportError` — `trend` module does not exist yet.

### Step 3: Implement `detect_config_change_events`

Create `eval/app_v2/engine/services/trend.py`:

```python
# eval/app_v2/engine/services/trend.py
from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import (
    ConfigChangeEvent,
    ConfigFieldChange,
    RunBundle,
    TrendBundle,
)

_DEFAULT_TRACKED_FIELDS: frozenset[str] = frozenset({
    "retriever",
    "index_name",
    "reranker_model",
    "reranker_top_n",
    "generator_model",
    "embedder_model",
    "top_k",
    "token_budget",
})


def detect_config_change_events(
    runs: Sequence[RunBundle],
    tracked_fields: set[str] | None = None,
) -> tuple[ConfigChangeEvent, ...]:
    """Return a ConfigChangeEvent for each adjacent pair of runs where tracked config fields differ."""
    if len(runs) < 2:
        return ()
    fields = tracked_fields if tracked_fields is not None else _DEFAULT_TRACKED_FIELDS
    events: list[ConfigChangeEvent] = []
    for prev, curr in zip(runs, runs[1:]):
        prev_cfg = dataclasses.asdict(prev.config)
        curr_cfg = dataclasses.asdict(curr.config)
        changes = tuple(
            ConfigFieldChange(field_name=f, before=prev_cfg.get(f), after=curr_cfg.get(f))
            for f in sorted(fields)
            if prev_cfg.get(f) != curr_cfg.get(f)
        )
        if changes:
            events.append(ConfigChangeEvent(
                from_run_id=prev.run_id,
                to_run_id=curr.run_id,
                timestamp=curr.timestamp,
                changes=changes,
            ))
    return tuple(events)


def build_trend_bundle(runs: Sequence[RunBundle]) -> TrendBundle:
    """Assemble a TrendBundle from a collection of RunBundles, sorted by timestamp."""
    sorted_runs = tuple(sorted(runs, key=lambda r: r.timestamp))
    timestamps = tuple(r.timestamp for r in sorted_runs)

    metric_series: dict[str, tuple[float | None, ...]] = {
        "recall@10": tuple(r.health.headline_recall_at_10 for r in sorted_runs),
        "ndcg@10": tuple(r.health.headline_ndcg_at_10 for r in sorted_runs),
        "avg_latency_ms": tuple(r.health.avg_latency_ms for r in sorted_runs),
        "avg_quality_score": tuple(r.health.avg_quality_score for r in sorted_runs),
    }

    diagnostic_rate_series: dict[DiagnosticCode, tuple[float | None, ...]] = {}
    for code in DiagnosticCode:
        rates: list[float | None] = []
        for run in sorted_runs:
            total = sum(run.health.severity_counts.values())
            count = run.health.diagnostic_counts.get(code, 0)
            rates.append(count / total if total > 0 else None)
        diagnostic_rate_series[code] = tuple(rates)

    return TrendBundle(
        runs=sorted_runs,
        timestamps=timestamps,
        metric_series=metric_series,
        diagnostic_rate_series=diagnostic_rate_series,
        verdict_series=tuple(r.health.verdict_status for r in sorted_runs),
        config_change_events=detect_config_change_events(sorted_runs),
    )
```

### Step 4: Run to verify it passes

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_trend_service.py -v
```

Expected: all 5 tests PASS.

### Step 5: Commit

```bash
git add eval/app_v2/engine/services/trend.py tests/eval/app_v2/engine/test_trend_service.py
git commit -m "feat(app-v2): add trend service with detect_config_change_events and build_trend_bundle"
```

---

## Task 2: `build_trend_bundle` tests

> This task adds the `build_trend_bundle` test cases to the same file. The implementation already exists from Task 1 — just add the tests and confirm they pass.

**Files:**
- Modify: `tests/eval/app_v2/engine/test_trend_service.py`

**Acceptance criteria:**
- `build_trend_bundle` always returns runs ordered by timestamp, regardless of input order.
- `metric_series["recall@10"]` contains one value per run in sorted order.
- `config_change_events` inside the returned bundle reflects config diffs between adjacent sorted runs.
- A run with `verdict_status=None` produces `None` in `verdict_series`.

---

### Step 1: Add tests to the existing file

Append these test functions to `tests/eval/app_v2/engine/test_trend_service.py`:

```python
from eval.app_v2.engine.services.trend import build_trend_bundle


def test_build_trend_bundle_orders_by_timestamp():
    runs = [
        _make_bundle("r2", _T0 + timedelta(days=1), recall=0.9),
        _make_bundle("r1", _T0, recall=0.8),
    ]
    bundle = build_trend_bundle(runs)
    assert bundle.runs[0].run_id == "r1"
    assert bundle.runs[1].run_id == "r2"


def test_build_trend_bundle_metric_series():
    runs = [
        _make_bundle("r1", _T0, recall=0.8),
        _make_bundle("r2", _T0 + timedelta(days=1), recall=0.9),
    ]
    bundle = build_trend_bundle(runs)
    assert bundle.metric_series["recall@10"] == (0.8, 0.9)
    assert len(bundle.timestamps) == 2


def test_build_trend_bundle_includes_config_changes():
    runs = [
        _make_bundle("r1", _T0, config=_make_config(top_k=10)),
        _make_bundle("r2", _T0 + timedelta(days=1), config=_make_config(top_k=20)),
    ]
    bundle = build_trend_bundle(runs)
    assert len(bundle.config_change_events) == 1


def test_build_trend_bundle_verdict_series_none_when_absent():
    runs = [
        _make_bundle("r1", _T0),
        _make_bundle("r2", _T0 + timedelta(days=1)),
    ]
    bundle = build_trend_bundle(runs)
    assert all(v is None for v in bundle.verdict_series)


def test_build_trend_bundle_diagnostic_rate_series_sums_to_one():
    runs = [_make_bundle("r1", _T0)]
    bundle = build_trend_bundle(runs)
    from eval.app_v2.engine.domain.enums import DiagnosticCode
    total_rate = sum(
        (bundle.diagnostic_rate_series[c][0] or 0.0)
        for c in DiagnosticCode
    )
    assert abs(total_rate - 1.0) < 1e-9
```

### Step 2: Run to verify they pass

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_trend_service.py -v
```

Expected: all tests PASS (no new implementation needed).

### Step 3: Run typecheck

```bash
make typecheck
```

Expected: no new errors in `eval/app_v2/engine/services/trend.py`.

### Step 4: Commit

```bash
git add tests/eval/app_v2/engine/test_trend_service.py
git commit -m "test(app-v2): add build_trend_bundle tests for trend service"
```

---

## Task 3: Trends page UI and app shell wiring

**Files:**
- Create: `eval/app_v2/ui/pages/trends.py`
- Modify: `eval/app_v2/app.py`

**Acceptance criteria:**
- Selecting fewer than 2 runs in the multiselect shows an informational message and no charts.
- With 2+ runs selected, recall@10 and ndcg@10 line charts render.
- Config change events appear as collapsible expanders with field-level before/after values.
- The verdict timeline and run summary table render for all selected runs.
- A `make typecheck` pass has no errors in the new file.
- The `Trends` option appears in the sidebar radio and navigates to the trends page without errors when the app is run.

---

### Step 1: Create `ui/pages/trends.py`

```python
# eval/app_v2/ui/pages/trends.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.services.trend import build_trend_bundle
from eval.app_v2.ui.app import load_bundle


def render(runs: list[tuple[str, Path]]) -> None:
    st.header("Trends")

    if not runs:
        st.info("No runs found in eval/runs/.")
        return

    names = [name for name, _ in runs]
    default_selection = names[:min(5, len(names))]
    selected_names = st.multiselect("Select runs to compare", names, default=default_selection)

    if len(selected_names) < 2:
        st.info("Select at least 2 runs to view trends.")
        return

    selected_pairs = [(n, p) for n, p in runs if n in selected_names]
    bundles = [load_bundle(name, str(path)) for name, path in selected_pairs]
    trend = build_trend_bundle(bundles)

    # ── Metric trends ──────────────────────────────────────────────────────────
    st.subheader("Metric trends")
    run_labels = [r.display_name for r in trend.runs]
    for metric_name in ("recall@10", "ndcg@10"):
        series = trend.metric_series[metric_name]
        if any(v is not None for v in series):
            chart_data = {
                "Run": run_labels,
                metric_name: [v if v is not None else 0.0 for v in series],
            }
            import pandas as pd
            df = pd.DataFrame(chart_data).set_index("Run")
            st.line_chart(df, x_label="Run", y_label=metric_name)

    # ── Config change events ───────────────────────────────────────────────────
    if trend.config_change_events:
        st.subheader("Configuration changes")
        for evt in trend.config_change_events:
            label = f"{evt.from_run_id} → {evt.to_run_id} ({evt.timestamp.strftime('%Y-%m-%d')})"
            with st.expander(label):
                for c in evt.changes:
                    st.markdown(f"- **{c.field_name}**: `{c.before}` → `{c.after}`")

    # ── Diagnostic failure-mode rates ─────────────────────────────────────────
    st.subheader("Failure mode rates over time")
    failure_codes = [
        c for c in DiagnosticCode
        if c not in (DiagnosticCode.GROUNDED_ANSWER, DiagnosticCode.NO_CLEAR_FAILURE)
    ]
    active_codes = [
        c for c in failure_codes
        if any((v or 0.0) > 0.0 for v in trend.diagnostic_rate_series.get(c, ()))
    ]
    if active_codes:
        import pandas as pd
        rate_data: dict[str, list[float]] = {"Run": run_labels}
        for code in active_codes:
            rate_data[code.value] = [v if v is not None else 0.0 for v in trend.diagnostic_rate_series[code]]
        df_rates = pd.DataFrame(rate_data).set_index("Run")
        st.line_chart(df_rates)
    else:
        st.caption("No active failure modes across selected runs.")

    # ── Verdict timeline ───────────────────────────────────────────────────────
    st.subheader("Verdict timeline")
    for run, verdict in zip(trend.runs, trend.verdict_series):
        icon = "✅" if verdict == "SHIP" else ("🚫" if verdict == "BLOCK" else "—")
        st.markdown(f"- **{run.display_name}**: {icon} {verdict or 'n/a'}")

    # ── Run summary table ──────────────────────────────────────────────────────
    st.subheader("Run summary")
    import pandas as pd
    rows = [
        {
            "Run": run.display_name,
            "recall@10": f"{run.health.headline_recall_at_10:.3f}",
            "ndcg@10": f"{run.health.headline_ndcg_at_10:.3f}",
            "Avg latency": f"{run.health.avg_latency_ms:.0f} ms" if run.health.avg_latency_ms else "—",
            "Verdict": run.health.verdict_status or "—",
            "Config changes": sum(
                1 for e in trend.config_change_events
                if e.to_run_id == run.run_id
            ),
        }
        for run in trend.runs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
```

### Step 2: Update `eval/app_v2/app.py`

The current `app.py` routes all pages through `PAGES[page](bundle)`. Trends needs `runs` instead. Read the existing file first, then apply this diff:

Replace the existing `main()` body in `eval/app_v2/app.py` with:

```python
"""Entry point: streamlit run eval/app_v2/app.py"""
from __future__ import annotations

import streamlit as st

from eval.app_v2.ui.app import DEFAULT_RUNS_DIR, discover_runs, load_bundle, run_selector_widget
from eval.app_v2.ui.pages.artifacts import render as artifacts_page
from eval.app_v2.ui.pages.forensics import render as forensics_page
from eval.app_v2.ui.pages.triage import render as triage_page
from eval.app_v2.ui.pages.trends import render as trends_page

SINGLE_RUN_PAGES = {
    "Triage": triage_page,
    "Forensics": forensics_page,
    "Artifacts": artifacts_page,
}

ALL_PAGE_NAMES = list(SINGLE_RUN_PAGES) + ["Trends"]


def main() -> None:
    st.set_page_config(page_title="Results Analyzer v2", layout="wide")
    runs = discover_runs(DEFAULT_RUNS_DIR)

    with st.sidebar:
        st.title("Results Analyzer v2")
        page = st.radio("Page", ALL_PAGE_NAMES)
        # Run selector only shown for single-run pages
        selected = run_selector_widget(runs) if page != "Trends" else None

    if page == "Trends":
        trends_page(runs)
    else:
        bundle = None
        if selected:
            name, run_dir = selected
            bundle = load_bundle(name, str(run_dir))
        SINGLE_RUN_PAGES[page](bundle)


if __name__ == "__main__":
    main()
```

### Step 3: Run typecheck

```bash
make typecheck
```

Fix any type errors before proceeding. Common issue: `pandas` import needs `import pandas as pd` at the top of the function (avoiding module-level import if pandas is optional). The current code already does this inline.

### Step 4: Smoke test (manual)

```bash
streamlit run eval/app_v2/app.py
```

Navigate to **Trends** in the sidebar. Select 2+ runs. Verify line charts appear. Then select fewer than 2 — verify the info message appears.

### Step 5: Commit

```bash
git add eval/app_v2/ui/pages/trends.py eval/app_v2/app.py
git commit -m "feat(app-v2): add Trends page with multi-run metric and failure-mode charts"
```

---

## Task 4: Chunk statistics engine (`engine/derived/chunk_stats.py`)

**Files:**
- Create: `eval/app_v2/engine/derived/chunk_stats.py`
- Create: `tests/eval/app_v2/engine/test_chunk_stats.py`

**Acceptance criteria:**
- A chunk that appears in `relevant_chunk_ids` but not in `retrieved_chunk_ids` for any query gets `miss_rate=1.0`.
- A chunk retrieved in every query where it is relevant gets `miss_rate=0.0`.
- A chunk dropped at rerank (present in `retrieved_chunk_ids` but absent from `reranked_chunk_ids`) has `rerank_drop_rate > 0.0`.
- `build_chunk_stats` returns results sorted by `miss_rate` descending (worst chunks first).
- `queries_where_relevant=0` and `queries_where_retrieved=0` never both occur in the same `ChunkStat` (every returned stat references at least one query).

---

### Step 1: Write the failing test

Create `tests/eval/app_v2/engine/test_chunk_stats.py`:

```python
from eval.app_v2.engine.derived.chunk_stats import ChunkStat, build_chunk_stats
from eval.app_v2.engine.domain.enums import (
    DiagnosticCode,
    GenerationStatus,
    PackingStatus,
    RerankStatus,
    RetrievalStatus,
    Severity,
)
from eval.app_v2.engine.domain.models import AnalyzedQuery, QueryDiagnostic, QueryRecord


def _diag(qid: str = "q1") -> QueryDiagnostic:
    return QueryDiagnostic(
        qid=qid,
        diagnostic_code=DiagnosticCode.GROUNDED_ANSWER,
        severity=Severity.OK,
        retrieval_status=RetrievalStatus.HIT,
        rerank_status=RerankStatus.ABSENT,
        packing_status=PackingStatus.ABSENT,
        generation_status=GenerationStatus.GROUNDED,
        root_cause_summary="",
        suggested_next_check=None,
        evidence_present=True,
        trace_available=False,
    )


def _aq(
    qid: str,
    relevant: list[str],
    retrieved: list[str],
    reranked: list[str] | None = None,
    packed: list[str] | None = None,
) -> AnalyzedQuery:
    record = QueryRecord(
        qid=qid,
        query="q",
        query_type=None,
        difficulty=None,
        is_unanswerable=False,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(relevant),
        retrieved_chunk_ids=tuple(retrieved),
        reranked_chunk_ids=tuple(reranked) if reranked is not None else None,
        packed_chunk_ids=tuple(packed) if packed is not None else None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None,
        answer_metrics=None,
        groundedness=None,
        latency_ms=None,
        trace_id=None,
        trace=None,
    )
    return AnalyzedQuery(record=record, diagnostic=_diag(qid))


def test_fully_missed_chunk_has_miss_rate_1():
    queries = [_aq("q1", relevant=["c1"], retrieved=["c2"])]
    stats = build_chunk_stats(queries)
    c1 = next(s for s in stats if s.chunk_id == "c1")
    assert c1.miss_rate == 1.0
    assert c1.queries_where_relevant == 1
    assert c1.queries_where_retrieved == 0


def test_always_retrieved_chunk_has_miss_rate_0():
    queries = [
        _aq("q1", relevant=["c1"], retrieved=["c1"]),
        _aq("q2", relevant=["c1"], retrieved=["c1"]),
    ]
    stats = build_chunk_stats(queries)
    c1 = next(s for s in stats if s.chunk_id == "c1")
    assert c1.miss_rate == 0.0


def test_rerank_drop_detected():
    queries = [_aq("q1", relevant=["c1"], retrieved=["c1", "c2"], reranked=["c2"])]
    stats = build_chunk_stats(queries)
    c1 = next(s for s in stats if s.chunk_id == "c1")
    assert c1.rerank_drop_rate > 0.0


def test_sorted_by_miss_rate_descending():
    queries = [
        _aq("q1", relevant=["bad"], retrieved=[]),        # miss_rate = 1.0
        _aq("q2", relevant=["good"], retrieved=["good"]),  # miss_rate = 0.0
    ]
    stats = build_chunk_stats(queries)
    assert stats[0].chunk_id == "bad"
    assert stats[-1].chunk_id == "good"


def test_no_queries_returns_empty():
    assert build_chunk_stats([]) == ()


def test_every_stat_has_nonzero_presence():
    queries = [_aq("q1", relevant=["c1"], retrieved=["c2"])]
    stats = build_chunk_stats(queries)
    for s in stats:
        assert s.queries_where_relevant > 0 or s.queries_where_retrieved > 0
```

### Step 2: Run to verify it fails

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_chunk_stats.py -v
```

Expected: `ImportError` — `chunk_stats` module does not exist.

### Step 3: Implement `engine/derived/chunk_stats.py`

```python
# eval/app_v2/engine/derived/chunk_stats.py
from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from eval.app_v2.engine.domain.models import AnalyzedQuery


@dataclass(frozen=True, slots=True)
class ChunkStat:
    chunk_id: str
    queries_where_relevant: int
    queries_where_retrieved: int
    queries_where_reranked: int
    queries_where_packed: int
    miss_rate: float         # (queries_where_relevant - queries_where_retrieved) / queries_where_relevant
    rerank_drop_rate: float  # (queries_where_retrieved - queries_where_reranked) / queries_where_retrieved


def build_chunk_stats(queries: Sequence[AnalyzedQuery]) -> tuple[ChunkStat, ...]:
    """Compute per-chunk aggregate statistics across all queries.

    Useful for identifying chunks that are consistently missed, dropped at rerank,
    or otherwise problematic across the run.
    """
    relevant_counts: dict[str, int] = defaultdict(int)
    retrieved_counts: dict[str, int] = defaultdict(int)
    reranked_counts: dict[str, int] = defaultdict(int)
    packed_counts: dict[str, int] = defaultdict(int)
    has_rerank_trace: set[str] = set()  # chunks seen in at least one query with trace

    for aq in queries:
        r = aq.record
        for cid in r.relevant_chunk_ids:
            relevant_counts[cid] += 1
        for cid in r.retrieved_chunk_ids:
            retrieved_counts[cid] += 1
        if r.reranked_chunk_ids is not None:
            for cid in r.reranked_chunk_ids:
                reranked_counts[cid] += 1
            # All retrieved chunks in this query now have rerank-trace data
            for cid in r.retrieved_chunk_ids:
                has_rerank_trace.add(cid)
        if r.packed_chunk_ids is not None:
            for cid in r.packed_chunk_ids:
                packed_counts[cid] += 1

    all_chunks = set(relevant_counts) | set(retrieved_counts)
    stats: list[ChunkStat] = []
    for cid in all_chunks:
        n_rel = relevant_counts.get(cid, 0)
        n_ret = retrieved_counts.get(cid, 0)
        n_rrk = reranked_counts.get(cid, 0)
        n_pck = packed_counts.get(cid, 0)
        miss_rate = (n_rel - n_ret) / n_rel if n_rel > 0 else 0.0
        rerank_drop_rate = (
            (n_ret - n_rrk) / n_ret
            if (n_ret > 0 and cid in has_rerank_trace)
            else 0.0
        )
        stats.append(ChunkStat(
            chunk_id=cid,
            queries_where_relevant=n_rel,
            queries_where_retrieved=n_ret,
            queries_where_reranked=n_rrk,
            queries_where_packed=n_pck,
            miss_rate=miss_rate,
            rerank_drop_rate=rerank_drop_rate,
        ))

    return tuple(sorted(stats, key=lambda s: s.miss_rate, reverse=True))
```

### Step 4: Run to verify it passes

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_chunk_stats.py -v
```

Expected: all 6 tests PASS.

### Step 5: Commit

```bash
git add eval/app_v2/engine/derived/chunk_stats.py tests/eval/app_v2/engine/test_chunk_stats.py
git commit -m "feat(app-v2): add chunk statistics engine for run-wide chunk-level analysis"
```

---

## Task 5: Chunk stats panel in Triage (chunk-centric views)

**Files:**
- Create: `eval/app_v2/ui/widgets/chunk_stats_panel.py`
- Modify: `eval/app_v2/ui/pages/triage.py`

**Acceptance criteria:**
- The Triage page shows a "Problem chunks" collapsible expander at the bottom.
- The expander renders a table of up to 20 chunks sorted by miss rate descending.
- Columns shown: chunk ID, times relevant, times retrieved, miss rate (%), rerank drop rate (%).
- When the run has no queries, the expander shows an info message instead of a table.
- The widget function has no Streamlit import at module level in `chunk_stats_panel.py` — it imports `streamlit` at function call time via the normal top-level import (keeping the widget stateless and testable; the engine function `build_chunk_stats` remains import-free of Streamlit).

---

### Step 1: Create `ui/widgets/chunk_stats_panel.py`

```python
# eval/app_v2/ui/widgets/chunk_stats_panel.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.derived.chunk_stats import build_chunk_stats
from eval.app_v2.engine.domain.models import RunBundle

_MAX_CHUNKS = 20


def render_chunk_stats_panel(bundle: RunBundle) -> None:
    """Render a 'Problem chunks' expander section for the Triage page."""
    with st.expander("Problem chunks (run-wide)", expanded=False):
        if not bundle.queries:
            st.info("No queries loaded.")
            return

        stats = build_chunk_stats(bundle.queries)
        if not stats:
            st.info("No chunk data available.")
            return

        display = stats[:_MAX_CHUNKS]
        rows = [
            {
                "Chunk ID": s.chunk_id,
                "# Relevant": s.queries_where_relevant,
                "# Retrieved": s.queries_where_retrieved,
                "# Reranked": s.queries_where_reranked,
                "Miss rate": f"{s.miss_rate:.1%}",
                "Rerank drop": f"{s.rerank_drop_rate:.1%}",
            }
            for s in display
        ]
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        if len(stats) > _MAX_CHUNKS:
            st.caption(f"Showing {_MAX_CHUNKS} of {len(stats)} chunks.")
```

### Step 2: Add the panel to `ui/pages/triage.py`

In `eval/app_v2/ui/pages/triage.py`, add the import at the top:

```python
from eval.app_v2.ui.widgets.chunk_stats_panel import render_chunk_stats_panel
```

Then add at the end of the `render` function (after the top-N queries section):

```python
    st.divider()
    render_chunk_stats_panel(bundle)
```

### Step 3: Smoke test

Launch the app (`streamlit run eval/app_v2/app.py`), navigate to **Triage**, scroll to the bottom, and expand "Problem chunks". Verify the table renders and is sorted by miss rate.

### Step 4: Commit

```bash
git add eval/app_v2/ui/widgets/chunk_stats_panel.py eval/app_v2/ui/pages/triage.py
git commit -m "feat(app-v2): add chunk-centric problem-chunks panel to Triage page"
```

---

## Task 6: Latency/quality correlation scatter chart (add to Trends page)

**Files:**
- Modify: `eval/app_v2/ui/pages/trends.py`

**Acceptance criteria:**
- A "Latency vs recall@10 (per query)" section appears in the Trends page below the verdict timeline.
- Each point in the scatter chart represents one query; x-axis is latency (ms), y-axis is recall@10, color encodes severity.
- Queries without `latency_ms` are excluded silently.
- When no query in any selected run has latency data, a caption "No latency data available" appears instead.
- The section does not appear for runs with zero queries.

---

### Step 1: Add correlation section to `ui/pages/trends.py`

In `eval/app_v2/ui/pages/trends.py`, add this function and call it at the end of `render()`:

```python
def _render_correlation(trend: "TrendBundle") -> None:
    """Scatter chart: latency vs recall@10, coloured by severity, across all selected runs."""
    import pandas as pd
    rows = []
    for run in trend.runs:
        for aq in run.queries:
            r = aq.record
            if r.latency_ms is not None:
                rows.append({
                    "latency_ms": r.latency_ms,
                    "recall@10": r.per_query_recall_at_k.get(10, 0.0),
                    "severity": aq.diagnostic.severity.value,
                    "run": run.display_name,
                })

    st.subheader("Latency vs recall@10 (per query)")
    if not rows:
        st.caption("No latency data available.")
        return

    df = pd.DataFrame(rows)
    st.scatter_chart(df, x="latency_ms", y="recall@10", color="severity", use_container_width=True)
```

Then in the `render()` function body, after `_render_run_summary_table(trend)` (or the equivalent inline code), add:

```python
    st.divider()
    _render_correlation(trend)
```

> Note: `st.scatter_chart` was introduced in Streamlit 1.26. Verify the installed version supports it before smoke testing. If unavailable, substitute with `st.altair_chart` as a fallback.

### Step 2: Smoke test

Launch app, go to Trends, select 2+ runs. Scroll to bottom — verify the scatter chart renders (or the "No latency data" caption if your eval runs lack latency).

### Step 3: Commit

```bash
git add eval/app_v2/ui/pages/trends.py
git commit -m "feat(app-v2): add latency vs recall scatter chart to Trends page"
```

---

## Task 7: Cohort analysis page (`ui/pages/cohorts.py`)

**Files:**
- Create: `eval/app_v2/ui/pages/cohorts.py`
- Modify: `eval/app_v2/app.py`

**Acceptance criteria:**
- A `Cohorts` entry appears in the sidebar radio and routes to the cohorts page.
- The user can select 1–3 grouping dimensions from `[query_type, difficulty, requires_synthesis, is_unanswerable]`.
- A dataframe shows cohort label, size, recall@10, and ndcg@10 for each slice.
- The three worst cohorts (by recall@10) are expanded with their top-3 queries shown as `diagnostic_card` widgets.
- Selecting 0 dimensions shows a warning message.
- When `bundle` is `None`, a standard "select a run" info message appears.

---

### Step 1: Create `ui/pages/cohorts.py`

```python
# eval/app_v2/ui/pages/cohorts.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.derived.slices import build_slice_table
from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.engine.services.forensics import list_queries_by_slice
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_card

_AVAILABLE_DIMS = ["query_type", "difficulty", "requires_synthesis", "is_unanswerable"]
_DEFAULT_DIMS = ["query_type", "difficulty"]
_WORST_N = 3
_QUERIES_PER_COHORT = 3


def render(bundle: RunBundle | None) -> None:
    st.header("Cohort Analysis")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    st.markdown(
        "Group queries by shared attributes and compare their aggregate retrieval performance. "
        "Use this to identify which query segments consistently underperform."
    )

    selected_dims = st.multiselect(
        "Group by dimensions",
        _AVAILABLE_DIMS,
        default=_DEFAULT_DIMS,
    )

    if not selected_dims:
        st.warning("Select at least one grouping dimension.")
        return

    table = build_slice_table(bundle.queries, group_by=selected_dims)

    if not table.rows:
        st.info("No data to group — run has no queries.")
        return

    # ── Full cohort table ──────────────────────────────────────────────────────
    st.subheader(f"All cohorts: {' × '.join(selected_dims)}")
    import pandas as pd
    rows = []
    for row in table.rows:
        label = " | ".join(f"{k}={v}" for k, v in row.key.parts)
        recall = row.metrics.get("recall@10")
        ndcg = row.metrics.get("ndcg@10")
        rows.append({
            "Cohort": label,
            "Size": row.size,
            "recall@10": f"{recall:.3f}" if recall is not None else "—",
            "ndcg@10": f"{ndcg:.3f}" if ndcg is not None else "—",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Worst cohorts drill-down ───────────────────────────────────────────────
    st.subheader(f"Worst {_WORST_N} cohorts (by recall@10)")

    def _recall_sort_key(row):
        return row.metrics.get("recall@10") if row.metrics.get("recall@10") is not None else 1.0

    worst_rows = sorted(table.rows, key=_recall_sort_key)[:_WORST_N]

    for row in worst_rows:
        label = " | ".join(f"{k}={v}" for k, v in row.key.parts)
        recall = row.metrics.get("recall@10")
        header = f"**{label}** — n={row.size}"
        if recall is not None:
            header += f", recall@10={recall:.3f}"
        st.markdown(header)

        cohort_queries = list_queries_by_slice(bundle, row.key)
        if not cohort_queries:
            st.caption("No matching queries found.")
            continue
        for aq in list(cohort_queries)[:_QUERIES_PER_COHORT]:
            render_diagnostic_card(aq, show_forensics_link=True)
        st.divider()
```

### Step 2: Wire `Cohorts` into `eval/app_v2/app.py`

Update the imports and page routing in `eval/app_v2/app.py`:

```python
from eval.app_v2.ui.pages.cohorts import render as cohorts_page

SINGLE_RUN_PAGES = {
    "Triage": triage_page,
    "Forensics": forensics_page,
    "Artifacts": artifacts_page,
    "Cohorts": cohorts_page,
}

ALL_PAGE_NAMES = list(SINGLE_RUN_PAGES) + ["Trends"]
```

The `main()` routing logic for `Cohorts` is already handled by the `SINGLE_RUN_PAGES[page](bundle)` branch added in Task 3 — no further changes to `main()` are needed.

### Step 3: Smoke test

Launch app, go to **Cohorts**. Select a run. Change the grouping dimensions — verify the table updates. Expand "Worst 3 cohorts" and verify diagnostic cards render with Forensics links.

### Step 4: Run typecheck

```bash
make typecheck
```

Resolve any errors before committing.

### Step 5: Commit

```bash
git add eval/app_v2/ui/pages/cohorts.py eval/app_v2/app.py
git commit -m "feat(app-v2): add Cohorts page for query clustering and slice-based analysis"
```

---

## Phase 6 completion checklist

After all 7 tasks are done, verify the following:

```bash
# All existing tests still pass
./scripts/py -m pytest tests/eval/app_v2/ -v

# New trend service tests
./scripts/py -m pytest tests/eval/app_v2/engine/test_trend_service.py -v

# New chunk stats tests
./scripts/py -m pytest tests/eval/app_v2/engine/test_chunk_stats.py -v

# No type errors
make typecheck
```

All 6 pages now available in the sidebar:
- **Triage** — run health summary + problem chunks
- **Forensics** — single-query deep inspection
- **Artifacts** — raw artifact viewer
- **Trends** — multi-run time series + config changes + correlation
- **Cohorts** — slice-based query clustering
- *(Compare and Verdicts from Phase 4 are wired in if their pages exist)*

---

## Dependency map

```
Task 1 (detect_config_change_events)
  └─ Task 2 (build_trend_bundle tests) depends on Task 1 implementation
       └─ Task 3 (Trends page) depends on Task 2 passing

Task 4 (build_chunk_stats engine) — independent of Tasks 1-3
  └─ Task 5 (Chunk panel in Triage) depends on Task 4

Task 3 (Trends page) — must exist before Task 6
  └─ Task 6 (Correlation chart) modifies the Trends page from Task 3

Task 7 (Cohorts page) — independent, but wires into app.py touched by Task 3
  └─ Merge app.py changes carefully if Tasks 3 and 7 are executed in parallel
```
