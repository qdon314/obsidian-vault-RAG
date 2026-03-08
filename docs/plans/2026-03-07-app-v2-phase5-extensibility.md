# Results Analyzer v2 — Phase 5: Extensibility

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Declarative facet system so adding a new filter is one list entry. No hardcoded by-type/by-difficulty sections.

**Architecture:** `engine/facets/registry.py` is pure Python. `ui/widgets/facet_panel.py` reads `FACETS` and renders the correct widget per `value_type`. `engine/services/filter.py` applies facet predicates to an `AnalyzedQuery` list.

**Prerequisite:** Phase 3 complete.

**Parallel execution map:**
```
Phase 3 complete
  ├─ Task 27 (facets registry) ─────────────────────────┐
  │    ├─ Task 29 (facet panel widget) ◄── 16, 27        │
  │    └─ Task 30 (filter service) ◄── 4, 27             │
  └─ (all three independent of each other after Task 27) ┘
```

---

## Task 27: `engine/facets/registry.py`

**Depends on:** Phase 1 domain models

**Files:**
- Create: `eval/app_v2/engine/facets/registry.py`
- Create: `tests/eval/app_v2/engine/test_facets.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_facets.py
from eval.app_v2.engine.facets.registry import FACETS, FacetDef
from eval.app_v2.engine.domain.models import QueryRecord


def _record(query_type="factual", difficulty="easy", is_unanswerable=False):
    return QueryRecord(
        qid="q1", query="test", query_type=query_type, difficulty=difficulty,
        is_unanswerable=is_unanswerable, requires_synthesis=False, tags=(),
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


def test_facets_list_is_nonempty():
    assert len(FACETS) >= 4


def test_facet_def_has_required_fields():
    for f in FACETS:
        assert isinstance(f, FacetDef)
        assert f.key
        assert f.label
        assert f.value_type in ("enum", "bool", "numeric_bucket")
        assert callable(f.extract)


def test_query_type_facet_extracts_correctly():
    facet = next(f for f in FACETS if f.key == "query_type")
    r = _record(query_type="factual")
    assert facet.extract(r) == "factual"


def test_bool_facet_extracts_correctly():
    facet = next(f for f in FACETS if f.key == "is_unanswerable")
    r = _record(is_unanswerable=True)
    assert facet.extract(r) is True
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_facets.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/facets/registry.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

from eval.app_v2.engine.domain.models import QueryRecord


@dataclass(frozen=True)
class FacetDef:
    key: str
    label: str
    value_type: Literal["enum", "bool", "numeric_bucket"]
    extract: Callable[[QueryRecord], Any]
    higher_is_better: bool = True


FACETS: list[FacetDef] = [
    FacetDef(
        key="query_type",
        label="Query Type",
        value_type="enum",
        extract=lambda r: r.query_type,
    ),
    FacetDef(
        key="difficulty",
        label="Difficulty",
        value_type="enum",
        extract=lambda r: r.difficulty,
    ),
    FacetDef(
        key="requires_synthesis",
        label="Requires Synthesis",
        value_type="bool",
        extract=lambda r: r.requires_synthesis,
    ),
    FacetDef(
        key="is_unanswerable",
        label="Unanswerable",
        value_type="bool",
        extract=lambda r: r.is_unanswerable,
        higher_is_better=False,
    ),
    # Severity is on the diagnostic, not the record — use a wrapper
    FacetDef(
        key="severity",
        label="Severity",
        value_type="enum",
        extract=lambda r: None,  # overridden by filter.py which receives AnalyzedQuery
    ),
]


def get_facet(key: str) -> FacetDef | None:
    return next((f for f in FACETS if f.key == key), None)
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_facets.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/facets/registry.py tests/eval/app_v2/engine/test_facets.py
git commit -m "feat(app-v2): add declarative facet registry"
```

**Acceptance criteria:** At least 4 `FacetDef` entries. All have `key`, `label`, `value_type`, `extract`. Adding a new facet requires only one list entry.

---

## Task 28 (renumbered from design Task 29): `ui/widgets/facet_panel.py`

**Depends on:** Tasks 27, 16 (ui shell established)

**Files:**
- Create: `eval/app_v2/ui/widgets/facet_panel.py`

**Step 1: Implement**

```python
# eval/app_v2/ui/widgets/facet_panel.py
"""
Stateless facet filter panel. Returns a dict[facet_key -> selected_value].
Reads FACETS and renders the correct widget per value_type automatically.
"""
from __future__ import annotations

import streamlit as st

from eval.app_v2.engine.domain.models import AnalyzedQuery
from eval.app_v2.engine.facets.registry import FACETS, FacetDef


def _collect_values(queries: list[AnalyzedQuery], facet: FacetDef) -> list[str]:
    """Collect unique non-None values for a facet across all queries."""
    values: set[str] = set()
    for aq in queries:
        if facet.key == "severity":
            values.add(str(aq.diagnostic.severity))
        else:
            v = facet.extract(aq.record)
            if v is not None:
                values.add(str(v))
    return sorted(values)


def render_facet_panel(queries: list[AnalyzedQuery]) -> dict[str, str | bool | None]:
    """
    Render a sidebar filter panel. Returns selected values keyed by facet.key.
    Returns None for a facet if no filter is applied (show all).
    """
    st.subheader("Filters")
    selections: dict[str, str | bool | None] = {}

    for facet in FACETS:
        if facet.value_type == "enum":
            values = _collect_values(queries, facet)
            if not values:
                continue
            opts = ["(all)"] + values
            choice = st.selectbox(facet.label, opts, key=f"facet_{facet.key}")
            selections[facet.key] = None if choice == "(all)" else choice

        elif facet.value_type == "bool":
            opts = ["(all)", "True", "False"]
            choice = st.radio(facet.label, opts, horizontal=True, key=f"facet_{facet.key}")
            if choice == "True":
                selections[facet.key] = True
            elif choice == "False":
                selections[facet.key] = False
            else:
                selections[facet.key] = None

        # numeric_bucket: extend when a numeric facet is added to FACETS

    return selections
```

**Step 2: Commit**

```bash
git add eval/app_v2/ui/widgets/facet_panel.py
git commit -m "feat(app-v2): add facet_panel widget"
```

**Acceptance criteria:** `render_facet_panel` renders one widget per `FacetDef` in `FACETS` automatically. Adding a new `FacetDef` to `FACETS` adds a new filter without touching `facet_panel.py`.

---

## Task 29 (renumbered from design Task 27): `engine/services/filter.py`

**Depends on:** Tasks 27, Phase 1 models

**Files:**
- Create: `eval/app_v2/engine/services/filter.py`
- Create: `tests/eval/app_v2/engine/test_filter.py`

**Step 1: Write the failing test**

```python
# tests/eval/app_v2/engine/test_filter.py
from eval.app_v2.engine.services.filter import apply_facet_filters
from eval.app_v2.engine.derived.diagnostics import analyze_queries
from eval.app_v2.engine.domain.models import QueryRecord


def _r(qid, qtype, difficulty):
    return QueryRecord(
        qid=qid, query="q", query_type=qtype, difficulty=difficulty,
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


def test_filter_by_query_type():
    analyzed = analyze_queries([
        _r("q1", "factual", "easy"),
        _r("q2", "conceptual", "hard"),
        _r("q3", "factual", "hard"),
    ])
    filtered = apply_facet_filters(analyzed, {"query_type": "factual"})
    assert len(filtered) == 2
    assert all(a.record.query_type == "factual" for a in filtered)


def test_no_filter_returns_all():
    analyzed = analyze_queries([_r("q1", "factual", "easy"), _r("q2", "conceptual", "hard")])
    filtered = apply_facet_filters(analyzed, {})
    assert len(filtered) == 2


def test_none_value_is_no_filter():
    analyzed = analyze_queries([_r("q1", "factual", "easy"), _r("q2", "conceptual", "hard")])
    filtered = apply_facet_filters(analyzed, {"query_type": None})
    assert len(filtered) == 2
```

**Step 2: Run — verify fails**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_filter.py -v
```

**Step 3: Implement**

```python
# eval/app_v2/engine/services/filter.py
"""
Facet-based filtering of AnalyzedQuery lists.
Filters are applied conjunctively (AND).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from eval.app_v2.engine.domain.models import AnalyzedQuery
from eval.app_v2.engine.facets.registry import FACETS, get_facet


def _matches(aq: AnalyzedQuery, key: str, value: Any) -> bool:
    if value is None:
        return True
    if key == "severity":
        return str(aq.diagnostic.severity) == str(value)
    facet = get_facet(key)
    if facet is None:
        return True
    actual = facet.extract(aq.record)
    if isinstance(value, bool):
        return actual == value
    return str(actual) == str(value)


def apply_facet_filters(
    queries: Sequence[AnalyzedQuery],
    selections: dict[str, Any],
) -> tuple[AnalyzedQuery, ...]:
    """Apply facet selections (AND-conjunctive). None values = no filter."""
    result = []
    for aq in queries:
        if all(_matches(aq, k, v) for k, v in selections.items()):
            result.append(aq)
    return tuple(result)
```

**Step 4: Run — verify passes**

```bash
./scripts/py -m pytest tests/eval/app_v2/engine/test_filter.py -v
```

**Step 5: Commit**

```bash
git add eval/app_v2/engine/services/filter.py tests/eval/app_v2/engine/test_filter.py
git commit -m "feat(app-v2): add facet-based filter service"
```

**Acceptance criteria:** `apply_facet_filters` with empty selections returns all queries. `None` values are treated as no filter. Multiple selections are AND-conjunctive.

---

## Phase 5 validation

```bash
./scripts/py -m pytest tests/eval/app_v2/ -v
```

All tests pass. Then verify in the UI that the facet panel integrates cleanly by wiring `render_facet_panel` into the Triage or Forensics page:

```python
# In triage.py, inside the sidebar or before the query list:
from eval.app_v2.ui.widgets.facet_panel import render_facet_panel
from eval.app_v2.engine.services.filter import apply_facet_filters

selections = render_facet_panel(list(bundle.queries))
filtered_queries = apply_facet_filters(bundle.queries, selections)
# then use filtered_queries for the diagnostic card list
```

**Phase 5 exit criterion:** Adding a new `FacetDef` to `FACETS` list automatically adds a filter widget in the UI and the filter service handles it without any other code change.
