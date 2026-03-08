# Results Analyzer v2 — Phase 3: Minimum Viable UI

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** A working Streamlit app: load run → identify bad queries → drill into pipeline root cause.

**Architecture:** `ui/` imports from `engine/`. No Streamlit in `engine/`. Pages are plain callables `(RunBundle | None) -> None`. Widgets are stateless functions over typed engine objects.

**Tech Stack:** Streamlit, `st.cache_data`, typed engine objects from Phase 1.

**Prerequisite:** Phase 1 complete. `build_bundle()` loads a real run.

**Run the app at any point with:**
```bash
./scripts/py -m streamlit run eval/app_v2/app.py
```

**Parallel execution map:**
```
Phase 1 complete
  └─ Task 16 (ui/app.py shell)
       ├─ Task 17 (metric_cards widget) ─────────┐
       ├─ Task 18 (diagnostic_card widget) ───────┤
       ├─ Task 19 (triage page) ◄── 17, 18 ───────┤
       ├─ Task 20 (forensics page) ◄── 18 ─────────┤
       └─ Task 21 (artifacts page) ────────────────┘
                                   (all independent after Task 16)
```

---

## Task 16: `ui/app.py` — app shell and run selector

**Depends on:** Phase 1 (build_bundle)

**Files:**
- Create: `eval/app_v2/ui/app.py`
- Create: `eval/app_v2/app.py` (entry point)

**Step 1: Check the developing-with-streamlit skill before writing any Streamlit code**

Use `Skill` tool with `developing-with-streamlit`.

**Step 2: Implement `ui/app.py`**

```python
# eval/app_v2/ui/app.py
from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.engine.loaders.bundle import build_bundle

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("eval/runs")
_RUN_DIR_PATTERN = re.compile(r"run_(\d{4}_\d{2}_\d{2}T\d{2}-\d{2})")


def discover_runs(runs_dir: Path) -> list[tuple[str, Path]]:
    """Return [(display_name, run_dir)] sorted newest-first."""
    entries: list[tuple[datetime, str, Path]] = []
    for d in runs_dir.iterdir():
        if not d.is_dir() or not (d / "metrics.json").exists():
            continue
        m = _RUN_DIR_PATTERN.match(d.name)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y_%m_%dT%H-%M").replace(tzinfo=UTC)
                entries.append((ts, d.name, d))
            except ValueError:
                pass
    entries.sort(reverse=True)
    return [(name, path) for _, name, path in entries]


@st.cache_data(show_spinner="Building run bundle...")
def load_bundle(run_id: str, run_dir_str: str) -> RunBundle:
    return build_bundle(Path(run_dir_str))


def run_selector_widget(runs: list[tuple[str, Path]]) -> tuple[str, Path] | None:
    if not runs:
        st.warning("No runs found in eval/runs/")
        return None
    names = [name for name, _ in runs]
    idx = st.selectbox("Select run", range(len(names)), format_func=lambda i: names[i])
    return runs[idx]
```

**Step 3: Implement entry point**

```python
# eval/app_v2/app.py
"""Entry point: streamlit run eval/app_v2/app.py"""
from __future__ import annotations

import streamlit as st

from eval.app_v2.ui.app import DEFAULT_RUNS_DIR, discover_runs, load_bundle, run_selector_widget

# Import pages (stubs until implemented)
def _stub(bundle):
    st.info("Page not yet implemented.")

try:
    from eval.app_v2.ui.pages.triage import render as triage_page
except ImportError:
    triage_page = _stub  # type: ignore

try:
    from eval.app_v2.ui.pages.forensics import render as forensics_page
except ImportError:
    forensics_page = _stub  # type: ignore

try:
    from eval.app_v2.ui.pages.artifacts import render as artifacts_page
except ImportError:
    artifacts_page = _stub  # type: ignore

PAGES = {
    "Triage": triage_page,
    "Forensics": forensics_page,
    "Artifacts": artifacts_page,
}


def main() -> None:
    st.set_page_config(page_title="Results Analyzer v2", layout="wide")
    runs = discover_runs(DEFAULT_RUNS_DIR)

    with st.sidebar:
        st.title("Results Analyzer v2")
        page = st.radio("Page", list(PAGES.keys()))
        selected = run_selector_widget(runs)

    bundle = None
    if selected:
        name, run_dir = selected
        bundle = load_bundle(name, str(run_dir))

    PAGES[page](bundle)


if __name__ == "__main__":
    main()
```

**Step 4: Smoke test — launch the app**

```bash
./scripts/py -m streamlit run eval/app_v2/app.py --server.headless true &
sleep 3
kill %1
```
Expected: starts without error.

**Step 5: Commit**

```bash
git add eval/app_v2/app.py eval/app_v2/ui/app.py
git commit -m "feat(app-v2): add app shell, run selector, cache boundary"
```

**Acceptance criteria:** App starts, sidebar shows run list, selecting a run calls `build_bundle()` without error (may take a few seconds first load), `st.cache_data` serves subsequent loads instantly.

---

## Task 17: `ui/widgets/metric_cards.py`

**Depends on:** Task 16 (imports pattern established)

**Files:**
- Create: `eval/app_v2/ui/widgets/metric_cards.py`

**Note:** Widgets are stateless functions. They do not call services. They only render.

**Step 1: Implement**

```python
# eval/app_v2/ui/widgets/metric_cards.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import RunHealthSummary

_SEVERITY_COLORS = {
    Severity.OK:       "#2ecc71",
    Severity.MINOR:    "#f39c12",
    Severity.MODERATE: "#e67e22",
    Severity.CRITICAL: "#e74c3c",
}


def render_kpi_cards(health: RunHealthSummary) -> None:
    """Render headline KPI metric cards from a RunHealthSummary."""
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recall@10", f"{health.headline_recall_at_10:.1%}")
    c2.metric("NDCG@10",   f"{health.headline_ndcg_at_10:.1%}")
    c3.metric(
        "Avg Quality",
        f"{health.avg_quality_score:.2f}" if health.avg_quality_score is not None else "—",
    )
    c4.metric(
        "Avg Latency",
        f"{health.avg_latency_ms:.0f} ms" if health.avg_latency_ms is not None else "—",
    )


def render_severity_bar(health: RunHealthSummary) -> None:
    """Horizontal breakdown: OK | MINOR | MODERATE | CRITICAL counts."""
    total = sum(health.severity_counts.values()) or 1
    cols = st.columns(4)
    for col, sev in zip(cols, [Severity.OK, Severity.MINOR, Severity.MODERATE, Severity.CRITICAL]):
        n = health.severity_counts.get(sev, 0)
        col.markdown(
            f"<div style='background:{_SEVERITY_COLORS[sev]};padding:8px;border-radius:4px;"
            f"text-align:center'><b>{sev.upper()}</b><br>{n} ({n/total:.0%})</div>",
            unsafe_allow_html=True,
        )


def render_dominant_failure_banner(health: RunHealthSummary) -> None:
    """Show the dominant failure mode as a colored banner."""
    if health.dominant_failure_mode is None:
        st.success("No dominant failure mode — run looks healthy.")
        return
    st.error(
        f"**Dominant failure:** `{health.dominant_failure_mode}` — "
        f"{health.dominant_failure_summary or ''} "
        f"({health.diagnostic_counts.get(health.dominant_failure_mode, 0)} queries)"
    )
```

**Step 2: Commit**

```bash
git add eval/app_v2/ui/widgets/metric_cards.py
git commit -m "feat(app-v2): add metric_cards widget"
```

**Acceptance criteria:** All three render functions accept `RunHealthSummary` and call only `st.*` methods. No imports from `engine/services/` or `engine/loaders/`.

---

## Task 18: `ui/widgets/diagnostic_card.py`

**Depends on:** Task 16

**Files:**
- Create: `eval/app_v2/ui/widgets/diagnostic_card.py`

**Step 1: Implement**

```python
# eval/app_v2/ui/widgets/diagnostic_card.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery

_SEV_BADGE = {
    Severity.OK:       "🟢",
    Severity.MINOR:    "🟡",
    Severity.MODERATE: "🟠",
    Severity.CRITICAL: "🔴",
}


def render_diagnostic_card(aq: AnalyzedQuery, *, show_forensics_link: bool = False) -> None:
    """Render a compact card for a single AnalyzedQuery."""
    d = aq.diagnostic
    r = aq.record
    badge = _SEV_BADGE.get(d.severity, "⚪")

    with st.container(border=True):
        cols = st.columns([0.05, 0.7, 0.25])
        cols[0].markdown(badge)
        cols[1].markdown(f"**`{r.qid}`** — {r.query[:80]}{'…' if len(r.query) > 80 else ''}")
        cols[2].markdown(f"`{d.diagnostic_code}`")

        with st.expander("Details", expanded=False):
            st.markdown(f"**Root cause:** {d.root_cause_summary}")
            if d.suggested_next_check:
                st.markdown(f"**Next check:** {d.suggested_next_check}")
            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.caption(f"Retrieval: `{d.retrieval_status}`")
            sc2.caption(f"Rerank: `{d.rerank_status}`")
            sc3.caption(f"Packing: `{d.packing_status}`")
            sc4.caption(f"Generation: `{d.generation_status}`")

        if show_forensics_link:
            if st.button("Inspect in Forensics →", key=f"forensics_{r.qid}"):
                st.session_state["forensics_qid"] = r.qid


def render_diagnostic_detail(aq: AnalyzedQuery) -> None:
    """Full diagnostic detail panel for the Forensics page."""
    d = aq.diagnostic
    r = aq.record
    badge = _SEV_BADGE.get(d.severity, "⚪")

    st.markdown(f"## {badge} `{d.diagnostic_code}` — {d.severity.upper()}")
    st.markdown(f"**Root cause:** {d.root_cause_summary}")
    if d.suggested_next_check:
        st.info(f"Suggested next check: {d.suggested_next_check}")

    with st.expander("Stage status breakdown", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval", d.retrieval_status)
        c2.metric("Rerank",    d.rerank_status)
        c3.metric("Packing",   d.packing_status)
        c4.metric("Generation", d.generation_status)

    with st.expander("Retrieval sets", expanded=False):
        retrieved_set = frozenset(r.retrieved_chunk_ids)
        matched = r.relevant_chunk_ids & retrieved_set
        missed  = r.relevant_chunk_ids - retrieved_set
        extra   = retrieved_set - r.relevant_chunk_ids
        st.markdown(f"- **Relevant:** {sorted(r.relevant_chunk_ids)}")
        st.markdown(f"- **Retrieved:** {list(r.retrieved_chunk_ids[:10])}")
        st.markdown(f"- **Matched:** {sorted(matched)}")
        st.markdown(f"- **Missed:** {sorted(missed)}")
        st.markdown(f"- **Extra retrieved:** {sorted(extra)}")

    if r.trace:
        with st.expander("Trace — pipeline drill-down", expanded=False):
            import json
            st.json(json.dumps(r.trace.raw_data, indent=2, default=str))
```

**Step 2: Commit**

```bash
git add eval/app_v2/ui/widgets/diagnostic_card.py
git commit -m "feat(app-v2): add diagnostic_card widget"
```

**Acceptance criteria:** `render_diagnostic_card` accepts `AnalyzedQuery`, renders badge + qid + code. Forensics button sets `st.session_state["forensics_qid"]`.

---

## Task 19: `ui/pages/triage.py`

**Depends on:** Tasks 16, 17, 18

**Files:**
- Create: `eval/app_v2/ui/pages/triage.py`

**Step 1: Implement**

```python
# eval/app_v2/ui/pages/triage.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_card
from eval.app_v2.ui.widgets.metric_cards import (
    render_dominant_failure_banner,
    render_kpi_cards,
    render_severity_bar,
)

_TOP_N = 10


def render(bundle: RunBundle | None) -> None:
    st.header("Triage")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    h = bundle.health

    # KPI cards
    render_kpi_cards(h)
    st.divider()

    # Severity bar
    st.subheader("Severity breakdown")
    render_severity_bar(h)
    st.divider()

    # Dominant failure mode
    st.subheader("Dominant failure mode")
    render_dominant_failure_banner(h)
    st.divider()

    # Verdict badge
    if bundle.verdict is not None:
        v = bundle.verdict
        if v.decision == "SHIP":
            st.success(f"**Verdict: SHIP** ✅")
        else:
            st.error(f"**Verdict: BLOCK** 🚫 — Failed: {', '.join(v.failed_check_names)}")
    st.divider()

    # Worst slice
    if h.worst_slice:
        st.subheader("Worst slice")
        parts = dict(h.worst_slice.parts)
        st.markdown(" | ".join(f"**{k}**: `{v}`" for k, v in parts.items()))
    st.divider()

    # Top-N critical/moderate queries
    critical = [
        aq for aq in bundle.queries
        if aq.diagnostic.severity in (Severity.CRITICAL, Severity.MODERATE)
    ]
    critical_sorted = sorted(critical, key=lambda aq: (
        0 if aq.diagnostic.severity == Severity.CRITICAL else 1
    ))

    st.subheader(f"Top {_TOP_N} queries needing attention")
    if not critical_sorted:
        st.success("No critical or moderate queries.")
    for aq in critical_sorted[:_TOP_N]:
        render_diagnostic_card(aq, show_forensics_link=True)
```

**Step 2: Update `eval/app_v2/app.py` to import the triage page (remove the try/except stub)**

Edit `app.py` to do a direct import:

```python
from eval.app_v2.ui.pages.triage import render as triage_page
```

**Step 3: Smoke test**

```bash
./scripts/py -m streamlit run eval/app_v2/app.py --server.headless true &
sleep 4
kill %1
```

**Step 4: Commit**

```bash
git add eval/app_v2/ui/pages/triage.py eval/app_v2/app.py
git commit -m "feat(app-v2): add Triage page"
```

**Acceptance criteria:** Triage page loads, shows KPI cards, severity bar, dominant failure banner, verdict badge (if present), and top-N diagnostic cards with a "Inspect in Forensics →" button that sets `session_state["forensics_qid"]`.

---

## Task 20: `ui/pages/forensics.py`

**Depends on:** Tasks 16, 18

**Files:**
- Create: `eval/app_v2/ui/pages/forensics.py`

**Step 1: Implement**

```python
# eval/app_v2/ui/pages/forensics.py
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.enums import Severity
from eval.app_v2.engine.domain.models import AnalyzedQuery, RunBundle
from eval.app_v2.ui.widgets.diagnostic_card import render_diagnostic_detail


def _find_query(bundle: RunBundle, qid: str) -> AnalyzedQuery | None:
    for aq in bundle.queries:
        if aq.record.qid == qid:
            return aq
    return None


def render(bundle: RunBundle | None) -> None:
    st.header("Forensics")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    # qid selection — prefer session_state push from Triage
    all_qids = [aq.record.qid for aq in bundle.queries]
    default_idx = 0
    pushed_qid = st.session_state.get("forensics_qid")
    if pushed_qid and pushed_qid in all_qids:
        default_idx = all_qids.index(pushed_qid)

    qid = st.selectbox("Query ID", all_qids, index=default_idx)
    aq = _find_query(bundle, qid)

    if aq is None:
        st.error(f"Query `{qid}` not found in bundle.")
        return

    # Query header
    r = aq.record
    with st.container(border=True):
        st.markdown(f"**Query:** {r.query}")
        cols = st.columns(4)
        cols[0].caption(f"Type: `{r.query_type or '—'}`")
        cols[1].caption(f"Difficulty: `{r.difficulty or '—'}`")
        cols[2].caption(f"Unanswerable: `{r.is_unanswerable}`")
        cols[3].caption(f"Trace: `{'✓' if r.trace else '✗'}`")
        if r.tags:
            st.caption(f"Tags: {', '.join(r.tags)}")

    st.divider()

    # Diagnostic detail
    render_diagnostic_detail(aq)

    # Answer section
    if r.answer_text:
        st.divider()
        with st.expander("Generated answer", expanded=False):
            st.markdown(r.answer_text)
            if r.answer_metrics:
                st.caption(f"Answer metrics: {r.answer_metrics}")

    # Per-query metrics
    st.divider()
    with st.expander("Per-query retrieval metrics", expanded=False):
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Recall@10",    f"{r.per_query_recall_at_k.get(10, 0):.1%}")
        mc2.metric("Precision@10", f"{r.per_query_precision_at_k.get(10, 0):.1%}")
        mc3.metric("NDCG@10",      f"{r.per_query_ndcg_at_k.get(10, 0):.1%}")
        mc4.metric("Latency",      f"{r.latency_ms} ms" if r.latency_ms else "—")
```

**Step 2: Update `app.py` to import forensics page directly**

**Step 3: Commit**

```bash
git add eval/app_v2/ui/pages/forensics.py
git commit -m "feat(app-v2): add Forensics page"
```

**Acceptance criteria:** Forensics page renders query header, diagnostic detail, retrieval sets, trace drill-down, and answer section. Navigating from Triage via "Inspect in Forensics →" pre-selects the correct qid.

---

## Task 21: `ui/pages/artifacts.py`

**Depends on:** Task 16

**Files:**
- Create: `eval/app_v2/ui/pages/artifacts.py`

**Step 1: Implement**

```python
# eval/app_v2/ui/pages/artifacts.py
from __future__ import annotations

import json

import streamlit as st

from eval.app_v2.engine.domain.models import RunBundle


def render(bundle: RunBundle | None) -> None:
    st.header("Artifacts")

    if bundle is None:
        st.info("Select a run from the sidebar.")
        return

    # Loader warnings
    if bundle.warnings:
        st.subheader(f"Loader warnings ({len(bundle.warnings)})")
        for w in bundle.warnings:
            st.warning(f"`{w.code}` — {w.message}" + (f" [{w.artifact_name}]" if w.artifact_name else ""))
    else:
        st.success("No loader warnings.")

    st.divider()

    # Raw artifact viewers
    st.subheader("Raw artifacts")
    for name, payload in bundle.raw_artifacts.items():
        with st.expander(f"`{name}`", expanded=False):
            try:
                st.code(json.dumps(payload, indent=2, default=str)[:5000], language="json")
            except Exception:
                st.text(str(payload)[:2000])
```

**Step 2: Update `app.py` to import artifacts page directly**

**Step 3: Commit**

```bash
git add eval/app_v2/ui/pages/artifacts.py
git commit -m "feat(app-v2): add Artifacts page"
```

---

## Phase 3 validation

Run the full app and manually verify:
1. Triage page shows KPI cards and top-N diagnostic cards
2. Clicking "Inspect in Forensics →" navigates to Forensics with the correct query pre-selected
3. Forensics page shows trace drill-down (if traces.jsonl exists for the selected run)
4. Artifacts page lists loader warnings

```bash
./scripts/py -m streamlit run eval/app_v2/app.py
```

**Phase 3 exit criterion:** Load run → identify bad queries → drill into pipeline root cause. All three pages functional.
