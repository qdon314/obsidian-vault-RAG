# Query Change Diagnostics Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add per-query drill-down to the comparison "Query Changes" tab, showing retrieval diffs with relevance labels, rank movement, query metadata, and answer diffs.

**Architecture:** Extract pure diff logic into `eval/app/results/query_diff.py` (no Streamlit dependency, fully testable). The rendering stays in `results_analyzer.py` and calls the diff functions. One new file for logic, one new test file.

**Tech Stack:** Python dataclasses, Streamlit (`st.expander`, `st.dataframe`, `st.metric`, `st.columns`), pandas (for `st.dataframe`).

**Spec:** [docs/specs/06-query-changes-enhancement.md](../specs/06-query-changes-enhancement.md)

---

### Task 1: Create `query_diff.py` with `natural_sort_key`

**Files:**
- Create: `eval/app/results/query_diff.py`
- Test: `tests/eval/test_query_diff.py`

**Step 1: Write the failing test**

Create `tests/eval/test_query_diff.py`:

```python
"""Tests for query diff logic (spec 06)."""

from __future__ import annotations

from eval.app.results.query_diff import natural_sort_key


class TestNaturalSortKey:
    def test_numeric_ordering(self) -> None:
        """q_2 sorts before q_10."""
        qids = ["q_10", "q_2", "q_100", "q_1"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["q_1", "q_2", "q_10", "q_100"]

    def test_mixed_alpha_numeric(self) -> None:
        """Handles mixed prefixes with numeric suffixes."""
        qids = ["a_2", "b_1", "a_10", "a_1"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["a_1", "a_2", "a_10", "b_1"]

    def test_pure_numeric(self) -> None:
        """Handles bare numbers."""
        qids = ["10", "2", "1"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["1", "2", "10"]

    def test_no_numbers(self) -> None:
        """Falls back to lexicographic for non-numeric strings."""
        qids = ["beta", "alpha", "gamma"]
        result = sorted(qids, key=natural_sort_key)
        assert result == ["alpha", "beta", "gamma"]
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/test_query_diff.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.app.results.query_diff'`

**Step 3: Write minimal implementation**

Create `eval/app/results/query_diff.py`:

```python
"""Pure logic for query change diagnostics (spec 06).

No Streamlit imports — all functions are testable standalone.
"""

from __future__ import annotations

import re

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def natural_sort_key(qid: str) -> tuple:
    """Sort key that orders numeric segments numerically.

    "q_2" < "q_10" < "q_100" instead of lexicographic "q_10" < "q_100" < "q_2".
    """
    parts = _NATURAL_SORT_RE.split(qid)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/eval/test_query_diff.py -v`
Expected: All 4 tests PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check eval/app/results/query_diff.py tests/eval/test_query_diff.py`

---

### Task 2: Add `ChunkDiffRow` and `compute_retrieval_diff`

**Files:**
- Modify: `eval/app/results/query_diff.py`
- Modify: `tests/eval/test_query_diff.py`

**Step 1: Write the failing tests**

Append to `tests/eval/test_query_diff.py`:

```python
from rag.eval.models import EvalResult, RetrievalResult
from eval.app.results.query_diff import compute_retrieval_diff


def _make_result(
    qid: str,
    retrieved: list[str],
    relevant: set[str],
) -> EvalResult:
    """Helper to create a minimal EvalResult for testing."""
    return EvalResult(
        qid=qid,
        query="test query",
        retrieval_result=RetrievalResult(
            qid=qid,
            retrieved_chunk_ids=tuple(retrieved),
            relevant_chunk_ids=relevant,
        ),
    )


class TestComputeRetrievalDiff:
    def test_tp_lost(self) -> None:
        """Relevant chunk in A but not B → TP lost."""
        a = _make_result("q1", ["c1", "c2"], {"c1"})
        b = _make_result("q1", ["c2", "c3"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.relevant is True
        assert c1_row.rank_a == 1
        assert c1_row.rank_b is None
        assert c1_row.status == "TP lost"

    def test_tp_gained(self) -> None:
        """Relevant chunk in B but not A → TP gained."""
        a = _make_result("q1", ["c2"], {"c1"})
        b = _make_result("q1", ["c1", "c2"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.relevant is True
        assert c1_row.rank_a is None
        assert c1_row.rank_b == 1
        assert c1_row.status == "TP gained"

    def test_fp_lost(self) -> None:
        """Irrelevant chunk in A but not B → FP lost."""
        a = _make_result("q1", ["c1", "c2"], {"c1"})
        b = _make_result("q1", ["c1"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c2_row = next(r for r in rows if r.chunk_id == "c2")
        assert c2_row.relevant is False
        assert c2_row.status == "FP lost"

    def test_fp_gained(self) -> None:
        """Irrelevant chunk in B but not A → FP gained."""
        a = _make_result("q1", ["c1"], {"c1"})
        b = _make_result("q1", ["c1", "c2"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        c2_row = next(r for r in rows if r.chunk_id == "c2")
        assert c2_row.relevant is False
        assert c2_row.status == "FP gained"

    def test_moved_up(self) -> None:
        """Chunk present in both, lower rank in B → Moved up."""
        a = _make_result("q1", ["c1", "c2", "c3"], set())
        b = _make_result("q1", ["c3", "c1", "c2"], set())
        rows = compute_retrieval_diff(a, b)

        c3_row = next(r for r in rows if r.chunk_id == "c3")
        assert c3_row.rank_a == 3
        assert c3_row.rank_b == 1
        assert c3_row.status == "Moved up"

    def test_moved_down(self) -> None:
        """Chunk present in both, higher rank in B → Moved down."""
        a = _make_result("q1", ["c1", "c2", "c3"], set())
        b = _make_result("q1", ["c3", "c1", "c2"], set())
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.rank_a == 1
        assert c1_row.rank_b == 2
        assert c1_row.status == "Moved down"

    def test_unchanged(self) -> None:
        """Chunk at same rank in both → Unchanged."""
        a = _make_result("q1", ["c1", "c2"], set())
        b = _make_result("q1", ["c1", "c2"], set())
        rows = compute_retrieval_diff(a, b)

        c1_row = next(r for r in rows if r.chunk_id == "c1")
        assert c1_row.rank_a == 1
        assert c1_row.rank_b == 1
        assert c1_row.status == "Unchanged"

    def test_sort_order_tp_lost_first(self) -> None:
        """TP lost rows sort before FP changes."""
        a = _make_result("q1", ["c1", "c2", "c3"], {"c1"})
        b = _make_result("q1", ["c2", "c4"], {"c1"})
        rows = compute_retrieval_diff(a, b)

        statuses = [r.status for r in rows]
        # TP lost should be first
        assert statuses[0] == "TP lost"

    def test_respects_k(self) -> None:
        """Only considers top-k chunks from each run."""
        a = _make_result("q1", ["c1", "c2", "c3"], set())
        b = _make_result("q1", ["c1", "c2", "c3"], set())
        rows = compute_retrieval_diff(a, b, k=2)

        chunk_ids = {r.chunk_id for r in rows}
        assert "c3" not in chunk_ids  # c3 is at rank 3, beyond k=2
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/test_query_diff.py::TestComputeRetrievalDiff -v`
Expected: FAIL — `ImportError: cannot import name 'compute_retrieval_diff'`

**Step 3: Write implementation**

Add to `eval/app/results/query_diff.py`:

```python
from dataclasses import dataclass

from rag.eval.models import EvalResult


@dataclass(frozen=True, slots=True)
class ChunkDiffRow:
    """One row of the retrieval diff table."""

    chunk_id: str
    relevant: bool
    rank_a: int | None  # 1-indexed, None if absent from run
    rank_b: int | None
    status: str  # "TP lost", "TP gained", "FP lost", "FP gained", "Moved up", "Moved down", "Unchanged"


# Sort priority: lower number = higher in table
_STATUS_SORT_ORDER = {
    "TP lost": 0,
    "TP gained": 1,
    "Moved up": 2,
    "Moved down": 3,
    "Unchanged": 4,
    "FP gained": 5,
    "FP lost": 6,
}


def compute_retrieval_diff(
    result_a: EvalResult,
    result_b: EvalResult,
    k: int = 10,
) -> list[ChunkDiffRow]:
    """Compute a unified diff of retrieved chunks between two runs.

    Returns rows sorted by diagnostic priority: TP lost first, then TP gained,
    then rank movers, then the rest.
    """
    ids_a = result_a.retrieval_result.retrieved_chunk_ids[:k]
    ids_b = result_b.retrieval_result.retrieved_chunk_ids[:k]

    rank_a = {cid: i + 1 for i, cid in enumerate(ids_a)}
    rank_b = {cid: i + 1 for i, cid in enumerate(ids_b)}

    # Union of relevant chunks from both results (should be identical, but be safe)
    relevant = result_a.retrieval_result.relevant_chunk_ids | result_b.retrieval_result.relevant_chunk_ids

    all_chunk_ids = list(dict.fromkeys(list(ids_a) + list(ids_b)))  # preserve order, deduplicate

    rows: list[ChunkDiffRow] = []
    for cid in all_chunk_ids:
        ra = rank_a.get(cid)
        rb = rank_b.get(cid)
        is_relevant = cid in relevant

        if ra is not None and rb is None:
            status = "TP lost" if is_relevant else "FP lost"
        elif ra is None and rb is not None:
            status = "TP gained" if is_relevant else "FP gained"
        elif ra is not None and rb is not None:
            if ra == rb:
                status = "Unchanged"
            elif rb < ra:
                status = "Moved up"
            else:
                status = "Moved down"
        else:
            continue  # shouldn't happen

        rows.append(ChunkDiffRow(
            chunk_id=cid,
            relevant=is_relevant,
            rank_a=ra,
            rank_b=rb,
            status=status,
        ))

    rows.sort(key=lambda r: _STATUS_SORT_ORDER.get(r.status, 99))
    return rows
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/eval/test_query_diff.py -v`
Expected: All tests PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check eval/app/results/query_diff.py tests/eval/test_query_diff.py`

---

### Task 3: Rewrite `_render_query_changes` and helpers in `results_analyzer.py`

**Files:**
- Modify: `eval/app/results_analyzer.py` (lines 622–669)

This is the Streamlit rendering task. No new tests — validated manually via `make results`.

**Step 1: Add imports at top of `results_analyzer.py`**

Add after the existing imports:

```python
import pandas as pd

from eval.app.results.query_diff import ChunkDiffRow, compute_retrieval_diff, natural_sort_key
from rag.eval.models import EvalResult
```

Note: `pandas` is already a transitive dependency of Streamlit; no new install needed.

**Step 2: Replace `_render_query_changes` (lines 622–656)**

Replace the entire function with:

```python
def _render_query_changes(comparison, filter_service: FilterService) -> None:
    """Render detailed query-level changes with drill-down diagnostics."""
    # Build lookup dicts once to avoid O(n) scans per query
    results_a: dict[str, EvalResult] = {r.qid: r for r in comparison.run_a.results}
    results_b: dict[str, EvalResult] = {r.qid: r for r in comparison.run_b.results}

    sorted_improved = sorted(comparison.improved_queries, key=natural_sort_key)
    sorted_regressed = sorted(comparison.regressed_queries, key=natural_sort_key)

    _render_query_category(
        "Improved", sorted_improved, results_a, results_b, "success",
    )
    _render_query_category(
        "Regressed", sorted_regressed, results_a, results_b, "error",
    )
```

**Step 3: Add `_render_query_category` helper**

```python
def _render_query_category(
    label: str,
    qids: list[str],
    results_a: dict[str, EvalResult],
    results_b: dict[str, EvalResult],
    style: str,
) -> None:
    """Render a category (improved/regressed) of changed queries."""
    if not qids:
        st.info(f"No queries {label.lower()} significantly")
        return

    st.subheader(f"{label} Queries ({len(qids)})")

    for qid in qids[:20]:
        result_a = results_a.get(qid)
        result_b = results_b.get(qid)
        if not result_a or not result_b:
            continue

        recall_a = _compute_recall(result_a)
        recall_b = _compute_recall(result_b)

        # Summary line for expander label
        query_type = result_b.query_type.value if result_b.query_type else "—"
        difficulty = result_b.difficulty.value if result_b.difficulty else "—"
        delta = recall_b - recall_a
        sign = "+" if delta >= 0 else ""
        summary = (
            f"[{qid}] {result_b.query[:60]}... | "
            f"{query_type} · {difficulty} | "
            f"Recall: {recall_a:.2f} → {recall_b:.2f} ({sign}{delta:.2f})"
        )

        with st.expander(summary, expanded=False):
            _render_query_detail(result_a, result_b)

    if len(qids) > 20:
        st.caption(f"...and {len(qids) - 20} more")
```

**Step 4: Add `_render_query_detail` helper**

```python
def _render_query_detail(result_a: EvalResult, result_b: EvalResult) -> None:
    """Render drill-down detail for a single changed query."""
    # --- Query header ---
    st.markdown(f"**Query:** {result_b.query}")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"**Type:** {result_b.query_type.value if result_b.query_type else '—'}")
    with col2:
        st.markdown(f"**Difficulty:** {result_b.difficulty.value if result_b.difficulty else '—'}")
    with col3:
        st.markdown(f"**Unanswerable:** {'Yes' if result_b.is_unanswerable else 'No'}")

    st.divider()

    # --- Retrieval diff table ---
    st.markdown("**Retrieval Diff**")
    diff_rows = compute_retrieval_diff(result_a, result_b)

    if diff_rows:
        df = pd.DataFrame(
            [
                {
                    "Chunk ID": r.chunk_id,
                    "Relevant": "Yes" if r.relevant else "No",
                    "Rank A": r.rank_a if r.rank_a is not None else "—",
                    "Rank B": r.rank_b if r.rank_b is not None else "—",
                    "Status": r.status,
                }
                for r in diff_rows
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No chunks retrieved in either run")

    # --- Answer diff (conditional) ---
    if result_a.answer and result_b.answer:
        st.divider()
        st.markdown("**Answer Diff**")

        col_a, col_b = st.columns(2)
        qid = result_a.qid

        with col_a:
            st.markdown("*Run A*")
            st.text_area(
                "Answer A",
                value=result_a.answer.text,
                height=150,
                disabled=True,
                key=f"ans_a_{qid}",
                label_visibility="collapsed",
            )
            _render_answer_metrics(result_a, prefix=f"a_{qid}")

        with col_b:
            st.markdown("*Run B*")
            st.text_area(
                "Answer B",
                value=result_b.answer.text,
                height=150,
                disabled=True,
                key=f"ans_b_{qid}",
                label_visibility="collapsed",
            )
            _render_answer_metrics(result_b, prefix=f"b_{qid}", baseline=result_a)
```

**Step 5: Add `_render_answer_metrics` helper**

```python
def _render_answer_metrics(
    result: EvalResult,
    *,
    prefix: str,
    baseline: EvalResult | None = None,
) -> None:
    """Render answer quality metrics, optionally with deltas against a baseline."""
    m = result.answer_metrics
    if not m:
        st.caption("No answer metrics")
        return

    bm = baseline.answer_metrics if baseline else None

    quality_delta = None
    correctness_delta = None
    halluc_delta = None

    if bm:
        if m.quality_score is not None and bm.quality_score is not None:
            quality_delta = f"{m.quality_score - bm.quality_score:.2f}"
        if m.correctness is not None and bm.correctness is not None:
            correctness_delta = f"{m.correctness - bm.correctness:.1f}"
        if m.hallucination_severity is not None and bm.hallucination_severity is not None:
            halluc_delta = f"{m.hallucination_severity - bm.hallucination_severity:.1f}"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Quality",
            f"{m.quality_score:.2f}" if m.quality_score is not None else "—",
            delta=quality_delta,
        )
    with c2:
        st.metric(
            "Correctness",
            f"{m.correctness:.1f}/5" if m.correctness is not None else "—",
            delta=correctness_delta,
        )
    with c3:
        st.metric(
            "Hallucination",
            f"{m.hallucination_severity:.1f}/5" if m.hallucination_severity is not None else "—",
            delta=halluc_delta,
            delta_color="inverse",  # lower hallucination is better
        )
```

**Step 6: Replace `_get_recall` (lines 659–669) with `_compute_recall`**

Replace with:

```python
def _compute_recall(result: EvalResult, k: int = 10) -> float:
    """Compute recall@k for a single result."""
    retrieved = set(result.retrieval_result.retrieved_chunk_ids[:k])
    relevant = result.retrieval_result.relevant_chunk_ids
    if not relevant:
        return 0.0
    return len(retrieved & relevant) / len(relevant)
```

This is the same logic but takes an `EvalResult` directly (no run lookup needed since we use dicts now). The old `_get_recall` took a `LoadedRun` and scanned linearly.

**Step 7: Lint and type check**

Run: `./scripts/py -m ruff check eval/app/results_analyzer.py eval/app/results/query_diff.py`
Run: `./scripts/py -m ruff format eval/app/results_analyzer.py eval/app/results/query_diff.py`

---

### Task 4: Run full test suite

**Step 1: Run all tests**

Run: `./scripts/py -m pytest -v`
Expected: All existing tests still pass, new `test_query_diff.py` tests pass.

**Step 2: Lint everything**

Run: `make lint`

---

### Task 5: Manual validation

**Step 1: Run the Streamlit app**

Run: `make results`

Validate against a comparison of two runs:
- Queries sorted naturally (q_2 before q_10)
- Summary row shows query type and difficulty
- Expanding a query shows full text, type, difficulty, unanswerable
- Retrieval diff table renders with correct ranks, relevance, and status
- TP lost rows appear first in the table
- Answer diff appears when both runs have answers (side-by-side text + metrics)
- No crashes when query type/difficulty/answer is None

This step cannot be automated — requires visual inspection.

---

## Suggested Commits

After all tasks pass:

**Commit 1:** `feat(eval): add query change diff logic`
- `eval/app/results/query_diff.py` (new)
- `tests/eval/test_query_diff.py` (new)

**Commit 2:** `feat(eval): add query change diagnostics to comparison view`
- `eval/app/results_analyzer.py` (modified)

---

## Deviation from Spec

The spec says "No new files." This plan creates `eval/app/results/query_diff.py` to keep the pure diff logic (dataclass + computation) separate from Streamlit rendering. This is necessary because:
1. `results_analyzer.py` imports `streamlit`, which is an optional `[ui]` dependency
2. Testing functions that live in that module would require Streamlit in the test environment
3. The diff logic is pure data transformation — it belongs in its own module

The alternative (putting everything in `results_analyzer.py`) works but makes the logic untestable without Streamlit installed.
