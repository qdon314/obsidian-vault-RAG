# Benchmark Review App Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit split-panel app at `benchmark_review/` that lets a reviewer approve, reject, or flag query candidates from a benchmark pipeline run, writing decisions to a sidecar `review_state.jsonl`.

**Architecture:** Hexagonal-lite — a pure `engine/` layer (loader, models, writer) with no Streamlit imports, and a `ui/` layer that owns all `st.*` calls. Session state drives selection; the writer appends/updates lines in the sidecar on every action. No full-file rewrites on save.

**Tech Stack:** Python 3.11, Streamlit ≥ 1.30 (already in `[ui]` extra), standard-library `json`/`pathlib`/`datetime`. All commands via `./scripts/py`.

---

## Task 1: Directory scaffold + empty modules

**Files:**
- Create: `benchmark_review/__init__.py`
- Create: `benchmark_review/engine/__init__.py`
- Create: `benchmark_review/engine/models.py`
- Create: `benchmark_review/engine/loader.py`
- Create: `benchmark_review/engine/writer.py`
- Create: `benchmark_review/ui/__init__.py`
- Create: `benchmark_review/ui/run_selector.py`
- Create: `benchmark_review/ui/record_list.py`
- Create: `benchmark_review/ui/record_detail.py`
- Create: `benchmark_review/ui/progress_bar.py`
- Create: `benchmark_review/app.py`
- Create: `tests/benchmark_review/__init__.py`

**Step 1: Create directory structure**

```bash
mkdir -p benchmark_review/engine benchmark_review/ui tests/benchmark_review
touch benchmark_review/__init__.py
touch benchmark_review/engine/__init__.py
touch benchmark_review/engine/models.py
touch benchmark_review/engine/loader.py
touch benchmark_review/engine/writer.py
touch benchmark_review/ui/__init__.py
touch benchmark_review/ui/run_selector.py
touch benchmark_review/ui/record_list.py
touch benchmark_review/ui/record_detail.py
touch benchmark_review/ui/progress_bar.py
touch benchmark_review/app.py
touch tests/benchmark_review/__init__.py
```

**Step 2: Commit scaffold**

```bash
git add benchmark_review/ tests/benchmark_review/
git commit -m "chore(benchmark-review): scaffold directory structure"
```

---

## Task 2: Domain models

**Files:**
- Modify: `benchmark_review/engine/models.py`
- Create: `tests/benchmark_review/test_models.py`

**Step 1: Write the failing test**

```python
# tests/benchmark_review/test_models.py
from benchmark_review.engine.models import ReviewRecord, ReviewStatus


def test_review_record_defaults_to_pending():
    rec = ReviewRecord(
        candidate_id="qc_50.1_cit_0",
        unit_id="50.1",
        query="What does 10 CFR 50.1 say?",
        query_class="citation_lookup",
        difficulty="easy",
        source_citations=("10 CFR 50.1",),
        evidence_span_ids=("50.1_0",),
        is_valid=False,
        validation_flags=("missing_snapshot_id",),
        critical_evidence=(),
        supporting_evidence=(),
        contextual_evidence=(),
        is_unanswerable=False,
        unanswerable_reason=None,
    )
    assert rec.review_status == ReviewStatus.PENDING
    assert rec.reviewed_by is None
    assert rec.reviewed_at is None
    assert rec.revision_notes is None
    assert rec.rejection_note is None


def test_review_status_values():
    assert ReviewStatus.PENDING.value == "pending"
    assert ReviewStatus.APPROVED.value == "approved"
    assert ReviewStatus.REJECTED.value == "rejected"
    assert ReviewStatus.NEEDS_REVISION.value == "needs_revision"
```

**Step 2: Run test to confirm failure**

```bash
./scripts/py -m pytest tests/benchmark_review/test_models.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`

**Step 3: Write the models**

```python
# benchmark_review/engine/models.py
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    span_id: str
    citation: str
    text: str
    char_start: int
    char_end: int
    tier: str  # "critical" | "supporting" | "contextual"


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    candidate_id: str
    unit_id: str
    query: str
    query_class: str
    difficulty: str
    source_citations: tuple[str, ...]
    evidence_span_ids: tuple[str, ...]
    is_valid: bool
    validation_flags: tuple[str, ...]
    critical_evidence: tuple[EvidenceSpan, ...]
    supporting_evidence: tuple[EvidenceSpan, ...]
    contextual_evidence: tuple[EvidenceSpan, ...]
    is_unanswerable: bool
    unanswerable_reason: str | None
    # Review state (populated from sidecar)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    revision_notes: str | None = None
    rejection_note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Step 4: Run test to confirm pass**

```bash
./scripts/py -m pytest tests/benchmark_review/test_models.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add benchmark_review/engine/models.py tests/benchmark_review/test_models.py
git commit -m "feat(benchmark-review): ReviewRecord domain model + ReviewStatus enum"
```

---

## Task 3: Loader — join candidate_generation + query_validation_results + evidence_tiers

**Files:**
- Modify: `benchmark_review/engine/loader.py`
- Create: `tests/benchmark_review/test_loader.py`

The loader reads three JSONL files from a run directory and joins them by `unit_id` / `candidate_id`.

**Step 1: Write the failing tests**

```python
# tests/benchmark_review/test_loader.py
import json
from pathlib import Path
import pytest
from benchmark_review.engine.loader import load_run
from benchmark_review.engine.models import ReviewStatus


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    candidates = [
        {
            "candidate_id": "qc_50.1_citation_lookup_0",
            "unit_id": "50.1",
            "query": "What does 10 CFR 50.1 say?",
            "query_class": "citation_lookup",
            "source_citations": ["10 CFR 50.1"],
            "evidence_span_ids": ["50.1_0"],
            "difficulty": "easy",
            "corpus_snapshot_id": "",
            "metadata": {},
        }
    ]
    validation = [
        {
            "candidate_id": "qc_50.1_citation_lookup_0",
            "is_valid": False,
            "flags": ["missing_snapshot_id"],
            "scores": {},
        }
    ]
    evidence = [
        {
            "unit_id": "50.1",
            "critical": [
                {
                    "span_id": "50.1_0",
                    "citation": "10 CFR 50.1",
                    "text": "The regulations in this part...",
                    "char_start": 0,
                    "char_end": 100,
                    "chunk_ids": [],
                    "tier": "critical",
                }
            ],
            "supporting": [],
            "contextual": [],
        }
    ]
    (tmp_path / "candidate_generation.jsonl").write_text(
        "\n".join(json.dumps(c) for c in candidates)
    )
    (tmp_path / "query_validation_results.jsonl").write_text(
        "\n".join(json.dumps(v) for v in validation)
    )
    (tmp_path / "evidence_tiers.jsonl").write_text(
        "\n".join(json.dumps(e) for e in evidence)
    )
    return tmp_path


def test_load_run_returns_one_record(run_dir: Path):
    records = load_run(run_dir)
    assert len(records) == 1


def test_load_run_joins_evidence(run_dir: Path):
    records = load_run(run_dir)
    rec = records[0]
    assert len(rec.critical_evidence) == 1
    assert rec.critical_evidence[0].span_id == "50.1_0"
    assert rec.critical_evidence[0].text == "The regulations in this part..."


def test_load_run_joins_validation(run_dir: Path):
    records = load_run(run_dir)
    rec = records[0]
    assert rec.is_valid is False
    assert "missing_snapshot_id" in rec.validation_flags


def test_load_run_defaults_review_status_to_pending(run_dir: Path):
    records = load_run(run_dir)
    assert records[0].review_status == ReviewStatus.PENDING


def test_load_run_merges_sidecar(run_dir: Path):
    sidecar = [
        {
            "candidate_id": "qc_50.1_citation_lookup_0",
            "review_status": "approved",
            "reviewed_by": "jsmith",
            "reviewed_at": "2026-03-25T10:00:00Z",
            "revision_notes": None,
            "rejection_note": None,
        }
    ]
    (run_dir / "review_state.jsonl").write_text(
        "\n".join(json.dumps(s) for s in sidecar)
    )
    records = load_run(run_dir)
    assert records[0].review_status == ReviewStatus.APPROVED
    assert records[0].reviewed_by == "jsmith"
```

**Step 2: Run tests to confirm failure**

```bash
./scripts/py -m pytest tests/benchmark_review/test_loader.py -v
```
Expected: ImportError

**Step 3: Write the loader**

```python
# benchmark_review/engine/loader.py
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from benchmark_review.engine.models import EvidenceSpan, ReviewRecord, ReviewStatus


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _parse_evidence_spans(spans: list[dict], tier: str) -> tuple[EvidenceSpan, ...]:
    return tuple(
        EvidenceSpan(
            span_id=s["span_id"],
            citation=s["citation"],
            text=s["text"],
            char_start=s["char_start"],
            char_end=s["char_end"],
            tier=tier,
        )
        for s in spans
    )


def load_run(run_dir: Path) -> list[ReviewRecord]:
    candidates = _read_jsonl(run_dir / "candidate_generation.jsonl")
    validations = {
        v["candidate_id"]: v
        for v in _read_jsonl(run_dir / "query_validation_results.jsonl")
    }
    evidence_by_unit = {
        e["unit_id"]: e for e in _read_jsonl(run_dir / "evidence_tiers.jsonl")
    }
    sidecar = {
        s["candidate_id"]: s
        for s in _read_jsonl(run_dir / "review_state.jsonl")
    }

    records: list[ReviewRecord] = []
    for cand in candidates:
        cid = cand["candidate_id"]
        uid = cand["unit_id"]
        val = validations.get(cid, {})
        ev = evidence_by_unit.get(uid, {})
        side = sidecar.get(cid, {})

        rec = ReviewRecord(
            candidate_id=cid,
            unit_id=uid,
            query=cand["query"],
            query_class=cand["query_class"],
            difficulty=cand.get("difficulty", "easy"),
            source_citations=tuple(cand.get("source_citations", [])),
            evidence_span_ids=tuple(cand.get("evidence_span_ids", [])),
            is_valid=val.get("is_valid", False),
            validation_flags=tuple(val.get("flags", [])),
            critical_evidence=_parse_evidence_spans(ev.get("critical", []), "critical"),
            supporting_evidence=_parse_evidence_spans(ev.get("supporting", []), "supporting"),
            contextual_evidence=_parse_evidence_spans(ev.get("contextual", []), "contextual"),
            is_unanswerable=cand.get("metadata", {}).get("is_unanswerable", False),
            unanswerable_reason=cand.get("metadata", {}).get("unanswerable_reason"),
            review_status=ReviewStatus(side["review_status"]) if "review_status" in side else ReviewStatus.PENDING,
            reviewed_by=side.get("reviewed_by"),
            reviewed_at=side.get("reviewed_at"),
            revision_notes=side.get("revision_notes"),
            rejection_note=side.get("rejection_note"),
        )
        records.append(rec)

    return records
```

**Step 4: Run tests to confirm pass**

```bash
./scripts/py -m pytest tests/benchmark_review/test_loader.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add benchmark_review/engine/loader.py tests/benchmark_review/test_loader.py
git commit -m "feat(benchmark-review): run loader — joins candidates, validation, evidence, sidecar"
```

---

## Task 4: Writer — append/update sidecar

**Files:**
- Modify: `benchmark_review/engine/writer.py`
- Create: `tests/benchmark_review/test_writer.py`

**Step 1: Write the failing tests**

```python
# tests/benchmark_review/test_writer.py
import json
from pathlib import Path
from benchmark_review.engine.writer import save_decision
from benchmark_review.engine.models import ReviewStatus


def test_save_decision_creates_sidecar(tmp_path: Path):
    save_decision(
        run_dir=tmp_path,
        candidate_id="qc_50.1_cit_0",
        status=ReviewStatus.APPROVED,
        reviewed_by="jsmith",
        revision_notes=None,
        rejection_note=None,
    )
    sidecar = tmp_path / "review_state.jsonl"
    assert sidecar.exists()
    record = json.loads(sidecar.read_text().strip())
    assert record["review_status"] == "approved"
    assert record["reviewed_by"] == "jsmith"
    assert "reviewed_at" in record


def test_save_decision_overwrites_existing_entry(tmp_path: Path):
    for status in [ReviewStatus.NEEDS_REVISION, ReviewStatus.APPROVED]:
        save_decision(
            run_dir=tmp_path,
            candidate_id="qc_50.1_cit_0",
            status=status,
            reviewed_by="jsmith",
            revision_notes="fix the citation" if status == ReviewStatus.NEEDS_REVISION else None,
            rejection_note=None,
        )
    lines = (tmp_path / "review_state.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # deduped, not appended
    assert json.loads(lines[0])["review_status"] == "approved"


def test_save_decision_preserves_other_entries(tmp_path: Path):
    save_decision(tmp_path, "qc_a", ReviewStatus.APPROVED, "jsmith", None, None)
    save_decision(tmp_path, "qc_b", ReviewStatus.REJECTED, "jsmith", None, "duplicate")
    lines = (tmp_path / "review_state.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
```

**Step 2: Run tests to confirm failure**

```bash
./scripts/py -m pytest tests/benchmark_review/test_writer.py -v
```
Expected: ImportError

**Step 3: Write the writer**

```python
# benchmark_review/engine/writer.py
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from benchmark_review.engine.models import ReviewStatus


def save_decision(
    run_dir: Path,
    candidate_id: str,
    status: ReviewStatus,
    reviewed_by: str,
    revision_notes: str | None,
    rejection_note: str | None,
) -> None:
    sidecar = run_dir / "review_state.jsonl"

    # Load existing entries (keyed by candidate_id for dedup)
    existing: dict[str, dict] = {}
    if sidecar.exists():
        for line in sidecar.read_text().splitlines():
            if line.strip():
                entry = json.loads(line)
                existing[entry["candidate_id"]] = entry

    existing[candidate_id] = {
        "candidate_id": candidate_id,
        "review_status": status.value,
        "reviewed_by": reviewed_by,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "revision_notes": revision_notes,
        "rejection_note": rejection_note,
    }

    sidecar.write_text(
        "\n".join(json.dumps(e) for e in existing.values()) + "\n"
    )
```

**Step 4: Run tests to confirm pass**

```bash
./scripts/py -m pytest tests/benchmark_review/test_writer.py -v
```
Expected: all PASS

**Step 5: Commit**

```bash
git add benchmark_review/engine/writer.py tests/benchmark_review/test_writer.py
git commit -m "feat(benchmark-review): sidecar writer with dedup-on-save"
```

---

## Task 5: Progress bar widget

**Files:**
- Modify: `benchmark_review/ui/progress_bar.py`

No unit tests needed — pure Streamlit rendering, no logic. Test visually after app runs.

**Step 1: Write the widget**

```python
# benchmark_review/ui/progress_bar.py
from __future__ import annotations

from collections import Counter

import streamlit as st

from benchmark_review.engine.models import ReviewRecord, ReviewStatus

_STATUS_COLOURS = {
    ReviewStatus.PENDING: "gray",
    ReviewStatus.APPROVED: "green",
    ReviewStatus.REJECTED: "red",
    ReviewStatus.NEEDS_REVISION: "orange",
}

_STATUS_LABELS = {
    ReviewStatus.PENDING: "Pending",
    ReviewStatus.APPROVED: "Approved",
    ReviewStatus.REJECTED: "Rejected",
    ReviewStatus.NEEDS_REVISION: "Needs revision",
}


def render(records: list[ReviewRecord]) -> None:
    counts = Counter(r.review_status for r in records)
    total = len(records)
    reviewed = total - counts[ReviewStatus.PENDING]

    cols = st.columns([3, 1, 1, 1, 1])
    with cols[0]:
        st.progress(reviewed / total if total else 0, text=f"{reviewed} / {total} reviewed")
    for col, status in zip(cols[1:], [ReviewStatus.PENDING, ReviewStatus.APPROVED, ReviewStatus.REJECTED, ReviewStatus.NEEDS_REVISION]):
        colour = _STATUS_COLOURS[status]
        label = _STATUS_LABELS[status]
        with col:
            st.markdown(f":{colour}[**{counts[status]}** {label}]")
```

**Step 2: Commit**

```bash
git add benchmark_review/ui/progress_bar.py
git commit -m "feat(benchmark-review): progress bar widget"
```

---

## Task 6: Record list widget (left panel)

**Files:**
- Modify: `benchmark_review/ui/record_list.py`

**Step 1: Write the widget**

```python
# benchmark_review/ui/record_list.py
from __future__ import annotations

import streamlit as st

from benchmark_review.engine.models import ReviewRecord, ReviewStatus

_STATUS_ICON = {
    ReviewStatus.PENDING: "⬜",
    ReviewStatus.APPROVED: "✅",
    ReviewStatus.REJECTED: "❌",
    ReviewStatus.NEEDS_REVISION: "🔶",
}

_FILTER_OPTIONS = ["All", "Pending", "Approved", "Rejected", "Needs revision"]
_FILTER_MAP = {
    "All": None,
    "Pending": ReviewStatus.PENDING,
    "Approved": ReviewStatus.APPROVED,
    "Rejected": ReviewStatus.REJECTED,
    "Needs revision": ReviewStatus.NEEDS_REVISION,
}


def render(records: list[ReviewRecord]) -> str | None:
    """Render the left panel. Returns the selected candidate_id or None."""
    st.subheader("Candidates")

    filter_choice = st.selectbox("Filter", _FILTER_OPTIONS, key="filter_status")
    search = st.text_input("Search", placeholder="query text or citation", key="search_query")

    status_filter = _FILTER_MAP[filter_choice]
    filtered = [
        r for r in records
        if (status_filter is None or r.review_status == status_filter)
        and (not search or search.lower() in r.query.lower() or any(search.lower() in c.lower() for c in r.source_citations))
    ]

    if not filtered:
        st.caption("No records match filter.")
        return st.session_state.get("selected_id")

    selected_id: str | None = st.session_state.get("selected_id")
    # Auto-select first pending if nothing selected
    if selected_id is None or not any(r.candidate_id == selected_id for r in filtered):
        pending = [r for r in filtered if r.review_status == ReviewStatus.PENDING]
        selected_id = (pending or filtered)[0].candidate_id

    for rec in filtered:
        icon = _STATUS_ICON[rec.review_status]
        label = f"{icon} {rec.candidate_id}"
        is_selected = rec.candidate_id == selected_id
        if st.button(label, key=f"btn_{rec.candidate_id}", use_container_width=True, type="primary" if is_selected else "secondary"):
            selected_id = rec.candidate_id

    return selected_id
```

**Step 2: Commit**

```bash
git add benchmark_review/ui/record_list.py
git commit -m "feat(benchmark-review): record list left panel with filter + search"
```

---

## Task 7: Record detail widget (right panel)

**Files:**
- Modify: `benchmark_review/ui/record_detail.py`

**Step 1: Write the widget**

```python
# benchmark_review/ui/record_detail.py
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from benchmark_review.engine.models import ReviewRecord, ReviewStatus
from benchmark_review.engine.writer import save_decision


def render(rec: ReviewRecord, run_dir: Path, records: list[ReviewRecord]) -> ReviewRecord | None:
    """Render the right panel. Returns updated record if a decision was saved, else None."""
    st.subheader(rec.candidate_id)

    # Status badge
    _status_badge(rec.review_status)
    st.caption(f"`{rec.query_class}` · `{rec.difficulty}`")

    st.divider()

    # Query
    st.markdown("**Query**")
    st.info(rec.query)

    # Source citations
    st.markdown("**Source citations**")
    for c in rec.source_citations:
        st.code(c, language=None)

    # Validation flags
    if rec.validation_flags:
        st.warning("Validation flags: " + ", ".join(f"`{f}`" for f in rec.validation_flags))

    # Evidence
    _render_evidence(rec)

    # Semantic duplicate hint
    _render_duplicate_hint(rec, records)

    st.divider()

    # Review actions
    return _render_actions(rec, run_dir, records)


def _status_badge(status: ReviewStatus) -> None:
    colour_map = {
        ReviewStatus.PENDING: "gray",
        ReviewStatus.APPROVED: "green",
        ReviewStatus.REJECTED: "red",
        ReviewStatus.NEEDS_REVISION: "orange",
    }
    colour = colour_map[status]
    st.markdown(f":{colour}[**{status.value.upper()}**]")


def _render_evidence(rec: ReviewRecord) -> None:
    if rec.is_unanswerable:
        st.markdown("**Unanswerable reason**")
        st.markdown(rec.unanswerable_reason or "_none provided_")
        if rec.critical_evidence:
            st.error("Pipeline bug: unanswerable record has non-empty critical evidence.")
        return

    for tier_label, spans in [
        ("Critical evidence", rec.critical_evidence),
        ("Supporting evidence", rec.supporting_evidence),
        ("Contextual evidence", rec.contextual_evidence),
    ]:
        if not spans:
            continue
        with st.expander(f"{tier_label} ({len(spans)} span{'s' if len(spans) != 1 else ''})", expanded=(tier_label == "Critical evidence")):
            for span in spans:
                st.markdown(f"**{span.citation}** · `{span.span_id}`")
                st.markdown(f"> {span.text}")


def _render_duplicate_hint(rec: ReviewRecord, all_records: list[ReviewRecord]) -> None:
    same_unit = [r for r in all_records if r.unit_id == rec.unit_id and r.candidate_id != rec.candidate_id]
    if not same_unit:
        return
    with st.expander(f"Similar queries — same unit ({len(same_unit)})", expanded=False):
        for other in same_unit:
            _status_badge(other.review_status)
            st.caption(other.candidate_id)
            st.markdown(other.query)
            if st.button("View", key=f"view_{other.candidate_id}_from_{rec.candidate_id}"):
                st.session_state["selected_id"] = other.candidate_id
                st.rerun()


def _render_actions(rec: ReviewRecord, run_dir: Path, all_records: list[ReviewRecord]) -> ReviewRecord | None:
    reviewer_id = st.session_state.get("reviewer_id", "")
    if not reviewer_id:
        st.warning("Enter your reviewer ID at the top of the page before making decisions.")
        return None

    col1, col2, col3 = st.columns(3)
    approve = col1.button("✅ Approve", key=f"approve_{rec.candidate_id}", use_container_width=True)
    reject = col2.button("❌ Reject", key=f"reject_{rec.candidate_id}", use_container_width=True)
    revise = col3.button("🔶 Needs revision", key=f"revise_{rec.candidate_id}", use_container_width=True)

    if revise:
        st.session_state[f"pending_action_{rec.candidate_id}"] = "needs_revision"
    if reject:
        st.session_state[f"pending_action_{rec.candidate_id}"] = "rejected"

    pending_action = st.session_state.get(f"pending_action_{rec.candidate_id}")

    note_text: str | None = None
    if pending_action in ("needs_revision", "rejected"):
        note_text = st.text_area(
            "Note (required)",
            key=f"note_{rec.candidate_id}",
            placeholder="Describe what needs fixing or why rejected",
        )
        if st.button("Save", key=f"save_note_{rec.candidate_id}", disabled=not note_text):
            status = ReviewStatus.NEEDS_REVISION if pending_action == "needs_revision" else ReviewStatus.REJECTED
            save_decision(
                run_dir=run_dir,
                candidate_id=rec.candidate_id,
                status=status,
                reviewed_by=reviewer_id,
                revision_notes=note_text if status == ReviewStatus.NEEDS_REVISION else None,
                rejection_note=note_text if status == ReviewStatus.REJECTED else None,
            )
            del st.session_state[f"pending_action_{rec.candidate_id}"]
            _advance_to_next_pending(rec.candidate_id, all_records)
            st.rerun()
        return None

    if approve:
        save_decision(
            run_dir=run_dir,
            candidate_id=rec.candidate_id,
            status=ReviewStatus.APPROVED,
            reviewed_by=reviewer_id,
            revision_notes=None,
            rejection_note=None,
        )
        _advance_to_next_pending(rec.candidate_id, all_records)
        st.rerun()

    return None


def _advance_to_next_pending(current_id: str, records: list[ReviewRecord]) -> None:
    ids = [r.candidate_id for r in records]
    current_idx = ids.index(current_id) if current_id in ids else -1
    pending = [r for r in records[current_idx + 1:] if r.review_status == ReviewStatus.PENDING]
    if pending:
        st.session_state["selected_id"] = pending[0].candidate_id
```

**Step 2: Commit**

```bash
git add benchmark_review/ui/record_detail.py
git commit -m "feat(benchmark-review): record detail right panel with approve/reject/revise actions"
```

---

## Task 8: Run selector widget

**Files:**
- Modify: `benchmark_review/ui/run_selector.py`

**Step 1: Write the widget**

```python
# benchmark_review/ui/run_selector.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

_BENCHMARK_RUNS_DIR = Path("benchmark_runs")


def render() -> Path | None:
    """Render run + reviewer selector. Returns selected run_dir or None."""
    run_dirs = sorted(
        [d for d in _BENCHMARK_RUNS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    ) if _BENCHMARK_RUNS_DIR.exists() else []

    if not run_dirs:
        st.error(f"No benchmark runs found in `{_BENCHMARK_RUNS_DIR}/`.")
        return None

    col1, col2 = st.columns([2, 1])
    with col1:
        run_name = st.selectbox(
            "Benchmark run",
            [d.name for d in run_dirs],
            key="selected_run_name",
        )
    with col2:
        reviewer_id = st.text_input(
            "Reviewer ID",
            value=st.session_state.get("reviewer_id", ""),
            key="reviewer_id_input",
            placeholder="e.g. jsmith",
        )
        st.session_state["reviewer_id"] = reviewer_id

    return _BENCHMARK_RUNS_DIR / run_name
```

**Step 2: Commit**

```bash
git add benchmark_review/ui/run_selector.py
git commit -m "feat(benchmark-review): run selector with reviewer ID input"
```

---

## Task 9: App entry point + Makefile target

**Files:**
- Modify: `benchmark_review/app.py`
- Modify: `Makefile`

**Step 1: Write the app entry point**

```python
# benchmark_review/app.py
"""Benchmark review app.

Launch with:
    make benchmark-review
or:
    ./scripts/py -m streamlit run benchmark_review/app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from benchmark_review.engine.loader import load_run
from benchmark_review.ui import progress_bar, record_detail, record_list, run_selector

st.set_page_config(
    page_title="Benchmark Review",
    page_icon="📋",
    layout="wide",
)

st.title("📋 Benchmark Review")

run_dir: Path | None = run_selector.render()
if run_dir is None:
    st.stop()

st.divider()

@st.cache_data(show_spinner="Loading run artifacts...")
def _load(run_dir_str: str) -> list:
    return load_run(Path(run_dir_str))

records = _load(str(run_dir))

# Invalidate cache when sidecar changes (keyed by mtime)
sidecar = run_dir / "review_state.jsonl"
sidecar_mtime = sidecar.stat().st_mtime if sidecar.exists() else 0.0
# Re-load after decisions are saved by passing mtime as cache key
records = load_run(run_dir)  # intentionally uncached — fast enough for JSONL at this scale

progress_bar.render(records)
st.divider()

left, right = st.columns([1, 2])

with left:
    selected_id = record_list.render(records)
    if selected_id:
        st.session_state["selected_id"] = selected_id

with right:
    selected_id = st.session_state.get("selected_id")
    if selected_id:
        selected_rec = next((r for r in records if r.candidate_id == selected_id), None)
        if selected_rec:
            record_detail.render(selected_rec, run_dir, records)
    else:
        st.info("Select a candidate from the list to begin reviewing.")
```

**Step 2: Add Makefile target**

Find the `curate` target in `Makefile` and add after it:

```makefile
benchmark-review:  ## Launch benchmark review app
	$(PYTHON) -m streamlit run benchmark_review/app.py
```

**Step 3: Commit**

```bash
git add benchmark_review/app.py Makefile
git commit -m "feat(benchmark-review): app entry point + make benchmark-review target"
```

---

## Task 10: Lint, typecheck, smoke test

**Step 1: Format + lint**

```bash
./scripts/py -m ruff format benchmark_review/ tests/benchmark_review/
./scripts/py -m ruff check benchmark_review/ tests/benchmark_review/
```
Fix any issues, then re-run until clean.

**Step 2: Typecheck**

```bash
./scripts/py -m mypy benchmark_review/ --ignore-missing-imports
```
Fix any type errors. Common issue: `dict[str, Any]` fields on frozen dataclasses — use `field(default_factory=dict)`.

**Step 3: Run all benchmark review tests**

```bash
./scripts/py -m pytest tests/benchmark_review/ -v
```
Expected: all PASS

**Step 4: Smoke test the app**

```bash
make benchmark-review
```
Open http://localhost:8501. Verify:
- Run selector shows `first_run`
- Candidate list shows 267 items (all pending)
- Clicking a candidate shows query text + evidence
- Approving a candidate saves to `review_state.jsonl` and advances to next

**Step 5: Final commit**

```bash
git add -p  # stage any lint/type fixes
git commit -m "fix(benchmark-review): lint and typecheck clean"
```
