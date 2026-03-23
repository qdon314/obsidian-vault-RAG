# NRC Benchmark M1: Stage 0 + Stage 1a Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Design doc:** `docs/plans/2026-03-21-nrc-benchmark-generation-design.md`

**Goal:** Build the deterministic foundation of the benchmark pipeline — domain models, corpus snapshot, Stage 0 source view, and Stage 1a structural segmentation — with no LLM involvement.

**Architecture:** New `src/benchmark/` package with hexagonal structure mirroring `src/rag/`. Imports `rag.domain` types (frozen dataclasses) and `rag.adapters.ingestion.regulatory` parser types, but never touches `rag.adapters` or `rag.ports`. All domain models are frozen dataclasses with `slots=True`.

**Tech Stack:** Python 3.11+, dataclasses, hashlib, `ecfr_parser.ParsedSection`/`ParsedParagraph`/`CrossRef`

**Prerequisites:** M0 is complete — `ecfr_parser` exports `CrossRef`, `SectionAmendment`, `ParsedParagraph`, `ParsedSection` with cross-reference support.

---

### Task 1: Benchmark domain enums

**Files:**
- Create: `src/benchmark/__init__.py`
- Create: `src/benchmark/domain/__init__.py`
- Create: `src/benchmark/domain/enums.py`
- Test: `tests/benchmark/__init__.py`
- Test: `tests/benchmark/domain/__init__.py`
- Test: `tests/benchmark/domain/test_enums.py`

**Step 1: Create package structure**

Create the `__init__.py` files. These are empty except for the top-level one:

```python
# src/benchmark/__init__.py
"""NRC benchmark generation pipeline."""
```

```python
# src/benchmark/domain/__init__.py
```

```python
# tests/benchmark/__init__.py
```

```python
# tests/benchmark/domain/__init__.py
```

**Step 2: Write the failing test**

```python
# tests/benchmark/domain/test_enums.py
"""Tests for benchmark domain enums."""

from __future__ import annotations

from enum import StrEnum

from benchmark.domain.enums import EvidenceTier, QueryClass, ReviewStatus, UnitKind


class TestUnitKind:
    def test_is_str_enum(self) -> None:
        assert issubclass(UnitKind, StrEnum)

    def test_values(self) -> None:
        expected = {
            "obligation",
            "prohibition",
            "threshold",
            "exception",
            "condition",
            "definition",
            "process",
            "cross_reference",
        }
        assert {e.value for e in UnitKind} == expected


class TestQueryClass:
    def test_is_str_enum(self) -> None:
        assert issubclass(QueryClass, StrEnum)

    def test_snake_case_values(self) -> None:
        """Design doc specifies snake_case for QueryClass values."""
        for member in QueryClass:
            assert "_" in member.value or member.value.isalpha(), (
                f"{member.name} should be snake_case"
            )

    def test_values(self) -> None:
        expected = {
            "citation_lookup",
            "narrow_factual",
            "rule_explanation",
            "cross_reference",
            "scenario_application",
            "unanswerable",
            "robustness_variant",
        }
        assert {e.value for e in QueryClass} == expected


class TestEvidenceTier:
    def test_is_str_enum(self) -> None:
        assert issubclass(EvidenceTier, StrEnum)

    def test_values(self) -> None:
        expected = {"critical", "supporting", "contextual"}
        assert {e.value for e in EvidenceTier} == expected

    def test_ordering(self) -> None:
        """Critical < supporting < contextual by string sort matches importance."""
        assert EvidenceTier.CONTEXTUAL < EvidenceTier.CRITICAL < EvidenceTier.SUPPORTING


class TestReviewStatus:
    def test_is_str_enum(self) -> None:
        assert issubclass(ReviewStatus, StrEnum)

    def test_values(self) -> None:
        expected = {"pending", "approved", "rejected", "needs_revision"}
        assert {e.value for e in ReviewStatus} == expected
```

**Step 3: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/benchmark/domain/test_enums.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'benchmark'`

**Step 4: Write minimal implementation**

```python
# src/benchmark/domain/enums.py
"""Enums for the NRC benchmark generation pipeline.

These use snake_case string values, intentionally distinct from
``rag.eval.schema.QueryType``. The exporter handles translation.
"""

from __future__ import annotations

from enum import StrEnum


class UnitKind(StrEnum):
    """Classification of a regulatory unit's normative function."""

    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    THRESHOLD = "threshold"
    EXCEPTION = "exception"
    CONDITION = "condition"
    DEFINITION = "definition"
    PROCESS = "process"
    CROSS_REFERENCE = "cross_reference"


class QueryClass(StrEnum):
    """Benchmark query class — drives scoring expectations."""

    CITATION_LOOKUP = "citation_lookup"
    NARROW_FACTUAL = "narrow_factual"
    RULE_EXPLANATION = "rule_explanation"
    CROSS_REFERENCE = "cross_reference"
    SCENARIO_APPLICATION = "scenario_application"
    UNANSWERABLE = "unanswerable"
    ROBUSTNESS_VARIANT = "robustness_variant"


class EvidenceTier(StrEnum):
    """Tier of evidence relevance (unit-relative, not query-relative)."""

    CRITICAL = "critical"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"


class ReviewStatus(StrEnum):
    """Human review state machine for benchmark queries."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
```

**Step 5: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/benchmark/domain/test_enums.py -v`
Expected: PASS (all 9 tests)

**Step 6: Commit**

```bash
git add src/benchmark/__init__.py src/benchmark/domain/__init__.py src/benchmark/domain/enums.py \
  tests/benchmark/__init__.py tests/benchmark/domain/__init__.py tests/benchmark/domain/test_enums.py
git commit -m "feat(benchmark): add domain enums — UnitKind, QueryClass, EvidenceTier, ReviewStatus (M1)"
```

---

### Task 2: Benchmark domain models — BenchmarkSourceSpan, RegulatoryUnit, StageConfig

**Files:**
- Create: `src/benchmark/domain/models.py`
- Test: `tests/benchmark/domain/test_models.py`

**Context:** These are the core domain objects for Stages 0 and 1a. All are frozen dataclasses.

Key design decisions:
- `BenchmarkSourceSpan` is the Stage 0 output — a benchmark-friendly view over the existing corpus.
- `RegulatoryUnit` is the Stage 1a output — a stable unit with an immutable `unit_id`.
- `unit_id` is minted from the subsection chain in Stage 1a and never changes afterward.
- `StageConfig` controls LLM parameters per stage (used in later milestones but defined here).

**Step 1: Write the failing test**

```python
# tests/benchmark/domain/test_models.py
"""Tests for benchmark domain models."""

from __future__ import annotations

import dataclasses

from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit, StageConfig


class TestBenchmarkSourceSpan:
    def test_frozen(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="10_cfr_50.46_b_1",
            section_title="Acceptance criteria for ECCS",
            text="Peak cladding temperature shall not exceed 2200°F.",
            char_start=0,
            char_end=51,
            chunk_ids_overlapping_span=("chunk_17",),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="abc123",
        )
        assert dataclasses.is_dataclass(span)
        with_error = False
        try:
            span.text = "mutated"  # type: ignore[misc]
        except dataclasses.FrozenInstanceError:
            with_error = True
        assert with_error

    def test_defaults(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="10_cfr_50.46_b_1",
            section_title="Acceptance criteria",
            text="Some text.",
            char_start=0,
            char_end=10,
            chunk_ids_overlapping_span=(),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="abc123",
        )
        assert span.metadata == {}


class TestRegulatoryUnit:
    def test_frozen(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="10_cfr_50.46_b_1",
            section_title="ECCS criteria",
            text="Temperature limit.",
            char_start=0,
            char_end=18,
            chunk_ids_overlapping_span=("c1",),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="snap1",
        )
        unit = RegulatoryUnit(
            unit_id="50.46_b_1_peak_cladding_temp",
            kind=UnitKind.THRESHOLD,
            spans=(span,),
            citation="10 CFR 50.46(b)(1)",
            subsection_chain=("b", "1"),
            parent_section_id="50.46",
            corpus_snapshot_id="snap1",
        )
        assert dataclasses.is_dataclass(unit)
        assert unit.unit_id == "50.46_b_1_peak_cladding_temp"

    def test_defaults(self) -> None:
        span = BenchmarkSourceSpan(
            source_doc_id="doc_1",
            citation="10 CFR 50.46(b)(1)",
            citation_key="key",
            section_title="Title",
            text="Text.",
            char_start=0,
            char_end=5,
            chunk_ids_overlapping_span=(),
            parent_section_id="50.46",
            effective_date="2026-01-01",
            corpus_snapshot_id="snap1",
        )
        unit = RegulatoryUnit(
            unit_id="50.46_b_1",
            kind=UnitKind.OBLIGATION,
            spans=(span,),
            citation="10 CFR 50.46(b)(1)",
            subsection_chain=("b", "1"),
            parent_section_id="50.46",
            corpus_snapshot_id="snap1",
        )
        assert unit.cross_references == ()
        assert unit.canonical_statement is None
        assert unit.entities == ()
        assert unit.conditions == ()
        assert unit.metadata == {}


class TestStageConfig:
    def test_defaults(self) -> None:
        cfg = StageConfig(model="gpt-4o")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 4096
        assert cfg.max_retries == 3
        assert cfg.timeout_s == 60.0
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/benchmark/domain/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/benchmark/domain/models.py
"""Domain models for the NRC benchmark generation pipeline.

All models are frozen dataclasses. Identity (``unit_id``) is structurally
derived in Stage 1a and immutable from that point forward.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.domain.enums import UnitKind


@dataclass(frozen=True, slots=True)
class BenchmarkSourceSpan:
    """Stage 0 output: a benchmark-friendly view over a corpus span.

    Maps one ``(section, subsection_chain)`` pair from ``ecfr_parser``
    to the benchmark's evidence coordinate system.
    """

    source_doc_id: str
    citation: str  # e.g. "10 CFR 50.46(b)(1)"
    citation_key: str  # stable key for dedup/linking
    section_title: str
    text: str
    char_start: int
    char_end: int
    chunk_ids_overlapping_span: tuple[str, ...]
    parent_section_id: str  # ParsedSection.section_number, e.g. "50.46"
    effective_date: str  # ISO date string
    corpus_snapshot_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegulatoryUnit:
    """Stage 1a output: a stable regulatory unit with immutable identity.

    ``unit_id`` is minted from the subsection chain (e.g.
    ``50.46_b_1_peak_cladding_temp``) and never changes after Stage 1a.
    """

    unit_id: str
    kind: UnitKind
    spans: tuple[BenchmarkSourceSpan, ...]
    citation: str
    subsection_chain: tuple[str, ...]
    parent_section_id: str
    corpus_snapshot_id: str

    # Populated by Stage 1b (LLM classification) — left as defaults in M1.
    canonical_statement: str | None = None
    entities: tuple[str, ...] = ()
    value: str | None = None  # numeric threshold if present
    conditions: tuple[str, ...] = ()

    # Populated by Stage 1a when paragraphs contain cross-references.
    cross_references: tuple[str, ...] = ()  # target_citation values

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Per-stage configuration for the benchmark pipeline.

    Deterministic by default (``temperature=0.0``) for reproducibility.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_s: float = 60.0
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/benchmark/domain/test_models.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/benchmark/domain/models.py tests/benchmark/domain/test_models.py
git commit -m "feat(benchmark): add domain models — BenchmarkSourceSpan, RegulatoryUnit, StageConfig (M1)"
```

---

### Task 3: Corpus snapshot utilities

**Files:**
- Create: `src/benchmark/domain/snapshot.py`
- Test: `tests/benchmark/domain/test_snapshot.py`

**Context:** The design doc specifies `compute_snapshot_id()` using `SHA-256 of sorted (doc_id, content_hash) pairs`. However, the `Document` model (`src/rag/domain/models.py:15-28`) has no `content_hash` field. We compute SHA-256 of `doc.text` at snapshot time instead, which achieves the same content-addressable property.

**Step 1: Write the failing test**

```python
# tests/benchmark/domain/test_snapshot.py
"""Tests for corpus snapshot utilities."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.domain.snapshot import compute_snapshot_id, verify_snapshot


@dataclass(frozen=True)
class _FakeDoc:
    """Minimal stand-in for rag.domain.models.Document."""

    doc_id: str
    text: str


class TestComputeSnapshotId:
    def test_deterministic(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="hello"), _FakeDoc(doc_id="d2", text="world")]
        assert compute_snapshot_id(docs) == compute_snapshot_id(docs)

    def test_order_independent(self) -> None:
        """Sorting ensures the same corpus in different order gives the same ID."""
        docs_a = [_FakeDoc(doc_id="d1", text="a"), _FakeDoc(doc_id="d2", text="b")]
        docs_b = [_FakeDoc(doc_id="d2", text="b"), _FakeDoc(doc_id="d1", text="a")]
        assert compute_snapshot_id(docs_a) == compute_snapshot_id(docs_b)

    def test_content_sensitive(self) -> None:
        docs_a = [_FakeDoc(doc_id="d1", text="version_1")]
        docs_b = [_FakeDoc(doc_id="d1", text="version_2")]
        assert compute_snapshot_id(docs_a) != compute_snapshot_id(docs_b)

    def test_empty_corpus(self) -> None:
        result = compute_snapshot_id([])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_returns_hex_string(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="x")]
        result = compute_snapshot_id(docs)
        assert len(result) == 64
        int(result, 16)  # should not raise


class TestVerifySnapshot:
    def test_match(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="hello")]
        snap_id = compute_snapshot_id(docs)
        assert verify_snapshot(docs, snap_id) is True

    def test_mismatch(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="hello")]
        assert verify_snapshot(docs, "0" * 64) is False
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/benchmark/domain/test_snapshot.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/benchmark/domain/snapshot.py
"""Corpus snapshot identity — content-addressable hash of the full corpus.

The snapshot ID is a SHA-256 of sorted ``(doc_id, content_hash)`` pairs.
Since ``rag.domain.models.Document`` has no ``content_hash`` field, we
compute SHA-256 of each document's text at snapshot time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol


class _HasDocIdAndText(Protocol):
    """Structural type for anything with doc_id and text."""

    @property
    def doc_id(self) -> str: ...
    @property
    def text(self) -> str: ...


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compute_snapshot_id(corpus: Sequence[_HasDocIdAndText]) -> str:
    """SHA-256 of sorted (doc_id, content_hash) pairs."""
    pairs = sorted(
        (doc.doc_id, _content_hash(doc.text)) for doc in corpus
    )
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def verify_snapshot(corpus: Sequence[_HasDocIdAndText], expected_id: str) -> bool:
    """Confirm the current corpus matches the claimed snapshot."""
    return compute_snapshot_id(corpus) == expected_id
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/benchmark/domain/test_snapshot.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/benchmark/domain/snapshot.py tests/benchmark/domain/test_snapshot.py
git commit -m "feat(benchmark): add corpus snapshot utilities — compute_snapshot_id, verify_snapshot (M1)"
```

---

### Task 4: Stage 0 — Corpus normalization (BenchmarkSourceSpan builder)

**Files:**
- Create: `src/benchmark/stages/__init__.py`
- Create: `src/benchmark/stages/stage_0_source_view.py`
- Test: `tests/benchmark/stages/__init__.py`
- Test: `tests/benchmark/stages/test_stage_0_source_view.py`

**Context:** Stage 0 reads `ParsedSection` objects from `ecfr_parser` and the existing `Chunk` index, then builds `BenchmarkSourceSpan` records. This is a plain builder function — not a swappable port. The key logic is:

1. Each `(section, paragraph)` pair becomes a `BenchmarkSourceSpan`.
2. `citation` and `citation_key` are structurally derived from section number + subsection chain.
3. `chunk_ids_overlapping_span` maps paragraph char ranges to chunk IDs from the existing index.
4. `corpus_snapshot_id` comes from the snapshot utility.

The chunk overlap resolution needs the existing chunk index. We accept a simple mapping of `doc_id -> list[Chunk]` to resolve overlaps.

**Step 1: Write the failing test**

```python
# tests/benchmark/stages/test_stage_0_source_view.py
"""Tests for Stage 0 corpus normalization."""

from __future__ import annotations

from dataclasses import dataclass, field

from benchmark.stages.stage_0_source_view import build_source_spans

from rag.adapters.ingestion.regulatory.ecfr_parser import (
    CrossRef,
    ParsedParagraph,
    ParsedSection,
)
from rag.domain.models import Chunk


def _make_chunk(
    chunk_id: str,
    doc_id: str,
    text: str,
    start_char: int,
    end_char: int,
) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        text=text,
        chunk_index=0,
        start_char=start_char,
        end_char=end_char,
    )


class TestBuildSourceSpans:
    def test_basic_span_creation(self) -> None:
        """A single section with one paragraph produces one span."""
        section = ParsedSection(
            section_number="50.46",
            title="Acceptance criteria for ECCS",
            part_number="50",
            paragraphs=(
                ParsedParagraph(
                    text="(b)(1) Peak cladding temperature shall not exceed 2200°F.",
                    level=2,
                    prefix="1",
                    subsection_tokens=("b", "1"),
                ),
            ),
        )
        chunks = [
            _make_chunk("c1", "doc_50", "Peak cladding temperature", 0, 100),
        ]
        spans = build_source_spans(
            sections=[section],
            doc_id="doc_50",
            chunk_index=chunks,
            corpus_snapshot_id="snap1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 1
        span = spans[0]
        assert span.source_doc_id == "doc_50"
        assert span.parent_section_id == "50.46"
        assert span.section_title == "Acceptance criteria for ECCS"
        assert span.corpus_snapshot_id == "snap1"
        assert span.effective_date == "2026-01-01"
        assert "50.46" in span.citation
        assert span.citation_key  # non-empty

    def test_citation_includes_subsection(self) -> None:
        """Citation includes the subsection chain."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS criteria",
            part_number="50",
            paragraphs=(
                ParsedParagraph(
                    text="(b)(1) Limit text.",
                    level=2,
                    prefix="1",
                    subsection_tokens=("b", "1"),
                ),
            ),
        )
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert "(b)(1)" in spans[0].citation

    def test_paragraph_without_subsection(self) -> None:
        """A paragraph with no subsection chain still produces a span."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS criteria",
            part_number="50",
            paragraphs=(
                ParsedParagraph(text="General intro text.", level=0, prefix=None),
            ),
        )
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 1
        assert spans[0].citation == "10 CFR 50.46"

    def test_chunk_overlap_resolution(self) -> None:
        """Spans include chunk IDs whose char ranges overlap the paragraph."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS",
            part_number="50",
            paragraphs=(
                ParsedParagraph(
                    text="(a) First paragraph text.",
                    level=1,
                    prefix="a",
                    subsection_tokens=("a",),
                ),
            ),
        )
        # Paragraph is the first one, char_start=0.  Chunk c1 overlaps, c2 doesn't.
        chunks = [
            _make_chunk("c1", "d1", "First paragraph", 0, 50),
            _make_chunk("c2", "d1", "Later text", 200, 300),
        ]
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=chunks,
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert "c1" in spans[0].chunk_ids_overlapping_span
        assert "c2" not in spans[0].chunk_ids_overlapping_span

    def test_multiple_paragraphs_multiple_spans(self) -> None:
        """Each paragraph becomes its own span."""
        section = ParsedSection(
            section_number="50.46",
            title="ECCS",
            part_number="50",
            paragraphs=(
                ParsedParagraph(text="(a) Para A.", level=1, prefix="a", subsection_tokens=("a",)),
                ParsedParagraph(text="(b) Para B.", level=1, prefix="b", subsection_tokens=("b",)),
            ),
        )
        spans = build_source_spans(
            sections=[section],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 2
        assert spans[0].citation != spans[1].citation

    def test_multiple_sections(self) -> None:
        """Spans from multiple sections are all returned."""
        sections = [
            ParsedSection(
                section_number="50.46",
                title="ECCS",
                part_number="50",
                paragraphs=(
                    ParsedParagraph(text="(a) A.", level=1, prefix="a", subsection_tokens=("a",)),
                ),
            ),
            ParsedSection(
                section_number="50.47",
                title="Emergency plans",
                part_number="50",
                paragraphs=(
                    ParsedParagraph(text="(a) B.", level=1, prefix="a", subsection_tokens=("a",)),
                ),
            ),
        ]
        spans = build_source_spans(
            sections=sections,
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert len(spans) == 2
        assert spans[0].parent_section_id == "50.46"
        assert spans[1].parent_section_id == "50.47"

    def test_empty_sections(self) -> None:
        """An empty section list produces no spans."""
        spans = build_source_spans(
            sections=[],
            doc_id="d1",
            chunk_index=[],
            corpus_snapshot_id="s1",
            effective_date="2026-01-01",
        )
        assert spans == []
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/benchmark/stages/test_stage_0_source_view.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/benchmark/stages/__init__.py
```

```python
# tests/benchmark/stages/__init__.py
```

```python
# src/benchmark/stages/stage_0_source_view.py
"""Stage 0: Corpus normalization — build benchmark-friendly source spans.

Reads ``ParsedSection`` objects from the eCFR parser and the existing chunk
index, producing ``BenchmarkSourceSpan`` records.  This is a plain builder
function (not a swappable port) because there is one sensible implementation.
"""

from __future__ import annotations

from collections.abc import Sequence

from benchmark.domain.models import BenchmarkSourceSpan
from rag.adapters.ingestion.regulatory.ecfr_parser import ParsedParagraph, ParsedSection
from rag.domain.models import Chunk


def _build_citation(section_number: str, paragraph: ParsedParagraph) -> str:
    """Build a citation string like ``10 CFR 50.46(b)(1)``."""
    base = f"10 CFR {section_number}"
    if not paragraph.subsection_tokens:
        return base
    suffix = "".join(f"({t})" for t in paragraph.subsection_tokens)
    return f"{base}{suffix}"


def _build_citation_key(section_number: str, paragraph: ParsedParagraph) -> str:
    """Build a stable citation key like ``10_cfr_50.46_b_1``."""
    parts = ["10_cfr", section_number.replace(".", "_")]
    parts.extend(paragraph.subsection_tokens)
    return "_".join(parts)


def _find_overlapping_chunks(
    chunks: Sequence[Chunk],
    para_start: int,
    para_end: int,
) -> tuple[str, ...]:
    """Return chunk IDs whose char range overlaps [para_start, para_end)."""
    result: list[str] = []
    for chunk in chunks:
        if chunk.start_char is None or chunk.end_char is None:
            continue
        # Overlap: chunk starts before para ends AND chunk ends after para starts
        if chunk.start_char < para_end and chunk.end_char > para_start:
            result.append(chunk.chunk_id)
    return tuple(result)


def build_source_spans(
    *,
    sections: Sequence[ParsedSection],
    doc_id: str,
    chunk_index: Sequence[Chunk],
    corpus_snapshot_id: str,
    effective_date: str,
) -> list[BenchmarkSourceSpan]:
    """Build ``BenchmarkSourceSpan`` records from parsed eCFR sections.

    Each ``(section, paragraph)`` pair produces one span.  Character offsets
    are computed cumulatively within each section.

    Args:
        sections: Parsed eCFR sections from ``parse_ecfr_xml()``.
        doc_id: The document ID for the source document.
        chunk_index: Existing chunks for overlap resolution.
        corpus_snapshot_id: Snapshot hash from ``compute_snapshot_id()``.
        effective_date: ISO date string for the corpus effective date.
    """
    spans: list[BenchmarkSourceSpan] = []
    for section in sections:
        char_offset = 0
        for para in section.paragraphs:
            para_start = char_offset
            para_end = char_offset + len(para.text)
            overlapping = _find_overlapping_chunks(chunk_index, para_start, para_end)

            spans.append(
                BenchmarkSourceSpan(
                    source_doc_id=doc_id,
                    citation=_build_citation(section.section_number, para),
                    citation_key=_build_citation_key(section.section_number, para),
                    section_title=section.title,
                    text=para.text,
                    char_start=para_start,
                    char_end=para_end,
                    chunk_ids_overlapping_span=overlapping,
                    parent_section_id=section.section_number,
                    effective_date=effective_date,
                    corpus_snapshot_id=corpus_snapshot_id,
                )
            )
            char_offset = para_end

    return spans
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/benchmark/stages/test_stage_0_source_view.py -v`
Expected: PASS (all 7 tests)

**Step 5: Commit**

```bash
git add src/benchmark/stages/__init__.py src/benchmark/stages/stage_0_source_view.py \
  tests/benchmark/stages/__init__.py tests/benchmark/stages/test_stage_0_source_view.py
git commit -m "feat(benchmark): add Stage 0 corpus normalization — build_source_spans (M1)"
```

---

### Task 5: Stage 1a — Structural segmentation (RulesExtractor)

**Files:**
- Create: `src/benchmark/ports/__init__.py`
- Create: `src/benchmark/ports/unit_extractor.py`
- Create: `src/benchmark/adapters/__init__.py`
- Create: `src/benchmark/adapters/extraction/__init__.py`
- Create: `src/benchmark/adapters/extraction/rules_extractor.py`
- Test: `tests/benchmark/adapters/__init__.py`
- Test: `tests/benchmark/adapters/extraction/__init__.py`
- Test: `tests/benchmark/adapters/extraction/test_rules_extractor.py`

**Context:** Stage 1a is the deterministic structural segmentation. The `RulesExtractor` adapter:

1. Consumes `BenchmarkSourceSpan` records from Stage 0.
2. Groups spans by `parent_section_id` (one section = one or more units).
3. Mints a stable `unit_id` from `section_number + subsection_chain`.
4. Detects cross-references from `ParsedParagraph.cross_references`.
5. Assigns a preliminary `UnitKind` based on structural cues (default `OBLIGATION`; cross-reference paragraphs get `CROSS_REFERENCE`). Full semantic classification happens in Stage 1b (M2).

The `UnitExtractor` port protocol defines the contract:

```python
class UnitExtractor(Protocol):
    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]: ...
```

**Step 1: Write the port protocol**

```python
# src/benchmark/ports/__init__.py
```

```python
# src/benchmark/ports/unit_extractor.py
"""Port protocol for Stage 1 regulatory unit extraction."""

from __future__ import annotations

from typing import Protocol

from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit


class UnitExtractor(Protocol):
    """Extract regulatory units from benchmark source spans."""

    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]: ...
```

**Step 2: Write the failing test**

```python
# tests/benchmark/adapters/__init__.py
```

```python
# tests/benchmark/adapters/extraction/__init__.py
```

```python
# tests/benchmark/adapters/extraction/test_rules_extractor.py
"""Tests for Stage 1a deterministic rules extractor."""

from __future__ import annotations

from benchmark.adapters.extraction.rules_extractor import RulesExtractor
from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan


def _span(
    section_id: str = "50.46",
    subsection: tuple[str, ...] = ("b", "1"),
    text: str = "Peak cladding temperature.",
    cross_refs: tuple[str, ...] = (),
    citation: str | None = None,
) -> BenchmarkSourceSpan:
    suffix = "".join(f"({t})" for t in subsection) if subsection else ""
    cit = citation or f"10 CFR {section_id}{suffix}"
    key = "_".join(["10_cfr", section_id.replace(".", "_"), *subsection])
    return BenchmarkSourceSpan(
        source_doc_id="doc_1",
        citation=cit,
        citation_key=key,
        section_title="Title",
        text=text,
        char_start=0,
        char_end=len(text),
        chunk_ids_overlapping_span=(),
        parent_section_id=section_id,
        effective_date="2026-01-01",
        corpus_snapshot_id="snap1",
        metadata={"subsection_tokens": subsection, "cross_references": cross_refs},
    )


class TestRulesExtractor:
    def test_single_span_produces_unit(self) -> None:
        extractor = RulesExtractor()
        spans = [_span()]
        units = extractor.extract(spans)
        assert len(units) == 1
        unit = units[0]
        assert unit.parent_section_id == "50.46"
        assert unit.subsection_chain == ("b", "1")
        assert unit.corpus_snapshot_id == "snap1"

    def test_unit_id_from_subsection_chain(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(section_id="50.46", subsection=("b", "1"))]
        units = extractor.extract(spans)
        assert units[0].unit_id == "50.46_b_1"

    def test_unit_id_no_subsection(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(section_id="50.46", subsection=())]
        units = extractor.extract(spans)
        assert units[0].unit_id == "50.46"

    def test_cross_reference_detection(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(cross_refs=("10 CFR §50.55a",))]
        units = extractor.extract(spans)
        assert units[0].cross_references == ("10 CFR §50.55a",)
        assert units[0].kind == UnitKind.CROSS_REFERENCE

    def test_default_kind_is_obligation(self) -> None:
        extractor = RulesExtractor()
        spans = [_span(cross_refs=())]
        units = extractor.extract(spans)
        assert units[0].kind == UnitKind.OBLIGATION

    def test_multiple_spans_same_section_grouped(self) -> None:
        """Spans with same section_id + subsection_chain are grouped into one unit."""
        extractor = RulesExtractor()
        spans = [
            _span(section_id="50.46", subsection=("b", "1"), text="First."),
            _span(section_id="50.46", subsection=("b", "1"), text="Second."),
        ]
        units = extractor.extract(spans)
        assert len(units) == 1
        assert len(units[0].spans) == 2

    def test_different_subsections_different_units(self) -> None:
        extractor = RulesExtractor()
        spans = [
            _span(section_id="50.46", subsection=("b", "1")),
            _span(section_id="50.46", subsection=("b", "2")),
        ]
        units = extractor.extract(spans)
        assert len(units) == 2
        assert {u.unit_id for u in units} == {"50.46_b_1", "50.46_b_2"}

    def test_satisfies_unit_extractor_protocol(self) -> None:
        """RulesExtractor structurally satisfies the UnitExtractor protocol."""
        from benchmark.ports.unit_extractor import UnitExtractor

        extractor: UnitExtractor = RulesExtractor()
        assert hasattr(extractor, "extract")

    def test_empty_input(self) -> None:
        extractor = RulesExtractor()
        assert extractor.extract([]) == []
```

**Step 3: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/benchmark/adapters/extraction/test_rules_extractor.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 4: Write minimal implementation**

```python
# src/benchmark/adapters/__init__.py
```

```python
# src/benchmark/adapters/extraction/__init__.py
```

```python
# src/benchmark/adapters/extraction/rules_extractor.py
"""Stage 1a: Deterministic structural segmentation.

Groups ``BenchmarkSourceSpan`` records by ``(parent_section_id, subsection_chain)``
and mints stable ``RegulatoryUnit`` records.  No LLM involvement — semantic
classification is deferred to Stage 1b.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.domain.enums import UnitKind
from benchmark.domain.models import BenchmarkSourceSpan, RegulatoryUnit


def _mint_unit_id(section_id: str, subsection_chain: tuple[str, ...]) -> str:
    """Mint a stable unit ID from section number and subsection chain."""
    parts = [section_id]
    parts.extend(subsection_chain)
    return "_".join(parts)


class RulesExtractor:
    """Deterministic regulatory unit extractor (Stage 1a).

    Groups spans by ``(parent_section_id, subsection_chain)`` and assigns
    a preliminary ``UnitKind`` based on structural cues:

    - Spans with cross-references → ``CROSS_REFERENCE``
    - All others → ``OBLIGATION`` (refined by LLM in Stage 1b)
    """

    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]:
        # Group spans by (section_id, subsection_chain).
        groups: dict[tuple[str, tuple[str, ...]], list[BenchmarkSourceSpan]] = defaultdict(list)
        for span in spans:
            subsection = tuple(span.metadata.get("subsection_tokens", ()))
            key = (span.parent_section_id, subsection)
            groups[key] = groups.get(key, [])
            groups[key].append(span)

        units: list[RegulatoryUnit] = []
        for (section_id, subsection_chain), group_spans in groups.items():
            # Collect cross-references from span metadata.
            cross_refs: list[str] = []
            for s in group_spans:
                cross_refs.extend(s.metadata.get("cross_references", ()))
            cross_refs_deduped = tuple(dict.fromkeys(cross_refs))

            kind = UnitKind.CROSS_REFERENCE if cross_refs_deduped else UnitKind.OBLIGATION

            units.append(
                RegulatoryUnit(
                    unit_id=_mint_unit_id(section_id, subsection_chain),
                    kind=kind,
                    spans=tuple(group_spans),
                    citation=group_spans[0].citation,
                    subsection_chain=subsection_chain,
                    parent_section_id=section_id,
                    corpus_snapshot_id=group_spans[0].corpus_snapshot_id,
                    cross_references=cross_refs_deduped,
                )
            )

        return units
```

**Step 5: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/benchmark/adapters/extraction/test_rules_extractor.py -v`
Expected: PASS (all 9 tests)

**Step 6: Commit**

```bash
git add src/benchmark/ports/__init__.py src/benchmark/ports/unit_extractor.py \
  src/benchmark/adapters/__init__.py src/benchmark/adapters/extraction/__init__.py \
  src/benchmark/adapters/extraction/rules_extractor.py \
  tests/benchmark/adapters/__init__.py tests/benchmark/adapters/extraction/__init__.py \
  tests/benchmark/adapters/extraction/test_rules_extractor.py
git commit -m "feat(benchmark): add Stage 1a RulesExtractor — structural segmentation (M1)"
```

---

### Task 6: Lint, typecheck, and full test pass

**Files:**
- Modify: any files with lint/type issues

**Step 1: Run linter**

Run: `make lint`
Expected: PASS (or fix any issues)

**Step 2: Run type checker**

Run: `./scripts/py -m mypy --config-file pyproject.toml src/benchmark`
Expected: PASS (or fix any issues)

**Step 3: Run full test suite**

Run: `make test`
Expected: PASS

**Step 4: Commit any fixes**

```bash
git add -u
git commit -m "chore(benchmark): fix lint/type issues (M1)"
```

---

### Task 7: Write ADR — Benchmark–Eval schema boundary

**Files:**
- Create: `docs/decisions/adr-benchmark-eval-schema-boundary.md`

**Context:** The design doc calls out this ADR at the "Eval framework integration" section. It formalizes the relationship between the benchmark domain (`src/benchmark/domain/`) and the eval framework (`src/rag/eval/schema.py`).

**Step 1: Write the ADR**

```markdown
# ADR: Benchmark–Eval Schema Boundary

**Status:** Accepted
**Date:** 2026-03-22
**Context:** NRC Benchmark Generation Pipeline, M1

## Context

The benchmark pipeline maintains a richer domain schema (tiered evidence,
rubrics, contamination probes, regulatory unit provenance) than the eval
framework's `EvalQuery` from `src/rag/eval/schema.py`.

We need to decide how these two schemas relate.

## Decision

The benchmark pipeline does **not** extend or modify `EvalQuery`.

Instead, `BenchmarkExporter` (the exporter port) is responsible for emitting
`EvalQuery`-compatible JSONL as its primary output format.

The mapping is:

| Benchmark field | EvalQuery field |
|---|---|
| `qid` | `qid` |
| `query` | `query` |
| `critical_evidence[*].chunk_ids` | `relevant_chunk_ids` / `critical_chunk_ids` |
| `source_citations` | `relevant_citations` / `critical_citations` |
| `query_class` | `query_type` (mapped via enum translation) |
| `difficulty` | `difficulty` |

Fields without an `EvalQuery` counterpart (tiered evidence detail, rubric,
contamination flags) are preserved only in the full benchmark JSONL export.

## Consequences

- The existing eval harness, metrics, judges, and Streamlit app work unchanged.
- The benchmark domain retains full fidelity for benchmark-specific analysis.
- No cross-layer coupling between the benchmark package and eval internals.
- The exporter is the single point of translation — changes to either schema
  require only exporter updates.
```

**Step 2: Commit**

```bash
git add docs/decisions/adr-benchmark-eval-schema-boundary.md
git commit -m "docs(decisions): add ADR — benchmark–eval schema boundary (M1)"
```

---

### Task 8: Write ADR — Benchmark schema versioning

**Files:**
- Create: `docs/decisions/adr-benchmark-schema-versioning.md`

**Context:** The design doc specifies a versioning policy for the benchmark dataset schema: additive minor versions, breaking major versions with migration scripts.

**Step 1: Write the ADR**

```markdown
# ADR: Benchmark Schema Versioning

**Status:** Accepted
**Date:** 2026-03-22
**Context:** NRC Benchmark Generation Pipeline, M1

## Context

The benchmark dataset JSONL schema will evolve as new query classes, fields,
and stages are added. We need a compatibility policy so consumers know what
to expect.

## Decision

The benchmark dataset schema follows semantic versioning:

- **Minor versions** (1.0 → 1.1): additive fields only. Backward compatible.
  Consumers must tolerate missing optional fields.
- **Major versions** (1.x → 2.0): breaking changes. A migration script at
  `src/benchmark/scripts/migrate_schema.py` is required for each major bump.

The `schema_version` field is mandatory on every benchmark record.

Schema version is set by the pipeline runner at export time, not by
individual stages.

## Consequences

- Consumers can safely ignore unknown fields (forward compatibility for minor bumps).
- Major bumps are rare and require explicit migration tooling.
- The `schema_version` field is the single source of truth for compatibility checks.
- The exporter validates that all output records have `schema_version` set.
```

**Step 2: Commit**

```bash
git add docs/decisions/adr-benchmark-schema-versioning.md
git commit -m "docs(decisions): add ADR — benchmark schema versioning policy (M1)"
```

---

## Summary

| Task | Scope | LLM? | Tests |
|---|---|---|---|
| 1 | Domain enums | No | 9 |
| 2 | Domain models (BenchmarkSourceSpan, RegulatoryUnit, StageConfig) | No | 5 |
| 3 | Snapshot utilities | No | 7 |
| 4 | Stage 0 source view builder | No | 7 |
| 5 | Stage 1a RulesExtractor + UnitExtractor port | No | 9 |
| 6 | Lint/typecheck/full test pass | No | — |
| 7 | ADR: benchmark–eval schema boundary | No | — |
| 8 | ADR: benchmark schema versioning | No | — |

Total: 8 tasks, ~37 tests, 0 LLM calls.
