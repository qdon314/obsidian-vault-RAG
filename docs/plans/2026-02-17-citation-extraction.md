# Citation Extraction & Normalization Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extract regulatory citations (CFR sections, dockets, ADAMS accessions, NUREGs, RIS/GL/IN) from NRC ADAMS document text and attach them as structured `CitationSpan` metadata to documents and frontmatter.

**Architecture:** A composable citation pipeline that runs during ingestion normalization. Text is normalized for OCR quirks, then run through per-kind extractors that produce `CitationSpan` frozen dataclasses with stable canonical keys. Results are scored, deduped, and attached to the markdown frontmatter. The domain model lives in `src/rag/domain/`, extractors in `src/rag/adapters/ingestion/case/`, integration via the existing normalizer.

**Tech Stack:** Python 3.11+ dataclasses, `re` module (compiled patterns), existing Hexagonal Architecture conventions.

---

## Existing Code Context

Before implementing, understand these existing pieces:

- **[normalizer.py](src/rag/adapters/ingestion/case/normalizer.py)** — already has `_PART_RE` and `_SECTION_RE` that extract regulation parts/sections into `CaseMetadata`. The citation extractor will produce richer structured spans; the normalizer will delegate to it.
- **[cross_references.py](src/rag/adapters/ingestion/regulatory/cross_references.py)** — handles wikilink rewriting (`§ 50.36` → `[[10 CFR §50.36]]`). This stays as-is. Different purpose: it rewrites text for Obsidian linking. The citation extractor mines structured data.
- **[case_documents.py](src/rag/domain/case_documents.py)** — `CaseMetadata` currently has `regulation_parts` and `regulation_sections` as flat tuples. We'll add a `citation_keys` field here.
- **Sample data** in `data/adams_samples/` — 38 real ADAMS documents (mix of modern ML-prefixed and legacy numeric accessions). Content quality varies from clean to heavily OCR-degraded.

---

### Task 1: `CitationSpan` Domain Model

**Files:**
- Create: `src/rag/domain/citations.py`
- Test: `tests/domain/test_citations.py`

**Step 1: Write the failing test**

```python
# tests/domain/test_citations.py
"""Tests for the CitationSpan domain model."""

from __future__ import annotations

from rag.domain.citations import CitationSpan


class TestCitationSpan:
    def test_frozen(self) -> None:
        span = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46",
            key="cfr:10:50.46",
            start=0,
            end=12,
            confidence=0.95,
            source_field="content",
        )
        assert span.kind == "cfr"
        assert span.key == "cfr:10:50.46"
        # Frozen — assignment raises
        try:
            span.kind = "other"  # type: ignore[misc]
            raise AssertionError("Expected FrozenInstanceError")
        except AttributeError:
            pass

    def test_defaults(self) -> None:
        span = CitationSpan(
            kind="docket",
            raw="50-247",
            key="docket:50-247",
            start=10,
            end=16,
            confidence=0.85,
            source_field="content",
        )
        assert span.context is None
        assert span.attrs == {}

    def test_with_attrs(self) -> None:
        span = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46(b)(1)",
            key="cfr:10:50.46(b)(1)",
            start=0,
            end=18,
            confidence=0.95,
            source_field="content",
            attrs={"title": 10, "part": 50, "section": "46", "subsections": ["b", "1"]},
        )
        assert span.attrs["part"] == 50
        assert span.attrs["subsections"] == ["b", "1"]

    def test_equality_by_value(self) -> None:
        a = CitationSpan(kind="cfr", raw="10 CFR 50.46", key="cfr:10:50.46", start=0, end=12, confidence=0.95, source_field="content")
        b = CitationSpan(kind="cfr", raw="10 CFR 50.46", key="cfr:10:50.46", start=0, end=12, confidence=0.95, source_field="content")
        assert a == b

    def test_context_window(self) -> None:
        span = CitationSpan(
            kind="cfr",
            raw="10 CFR 50.46",
            key="cfr:10:50.46",
            start=100,
            end=112,
            confidence=0.95,
            source_field="content",
            context="...in accordance with 10 CFR 50.46 requirements...",
        )
        assert "in accordance with" in span.context  # type: ignore[operator]
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/domain/test_citations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'rag.domain.citations'`

**Step 3: Write minimal implementation**

```python
# src/rag/domain/citations.py
"""Domain model for extracted citation spans.

A ``CitationSpan`` represents a single citation found in document text,
carrying the raw match, a stable canonical key for dedup/linking, its
location in the normalized text, and a deterministic confidence score.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class CitationSpan:
    """A single citation extracted from document text."""

    kind: str  # "cfr" | "cfrpart" | "cfrapp" | "docket" | "adams" | "nureg" | "ris" | "gl" | "in"
    raw: str  # exact matched text
    key: str  # canonical key (stable, for dedup/linking)
    start: int  # span start in normalized text
    end: int  # span end in normalized text
    confidence: float  # 0.0–1.0 (deterministic scoring)
    source_field: str  # "title" | "content" | "metadata"
    context: str | None = None  # short context window for debugging/UI
    attrs: dict[str, object] = field(default_factory=dict)  # parsed structure
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/domain/test_citations.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/rag/domain/citations.py tests/domain/test_citations.py
git commit -m "domain: add CitationSpan frozen dataclass for citation extraction"
```

---

### Task 2: Text Normalizer for OCR Cleanup

Real ADAMS documents have OCR artifacts: split characters ("C F R"), em-dashes as `--`, curly quotes, inconsistent whitespace, "C.F.R." variants. This stage improves regex recall.

**Files:**
- Create: `src/rag/adapters/ingestion/case/text_normalizer.py`
- Test: `tests/adapters/ingestion/case/test_text_normalizer.py`

**Step 1: Write the failing test**

```python
# tests/adapters/ingestion/case/test_text_normalizer.py
"""Tests for citation-oriented text normalization."""

from __future__ import annotations

from rag.adapters.ingestion.case.text_normalizer import normalize_for_citation_extraction


class TestNormalizeForCitationExtraction:
    def test_collapse_whitespace(self) -> None:
        assert normalize_for_citation_extraction("10  CFR   50.46") == "10 CFR 50.46"

    def test_fix_ocr_split_cfr(self) -> None:
        assert "10 CFR" in normalize_for_citation_extraction("10 C F R 50.46")

    def test_normalize_cfr_variants(self) -> None:
        result = normalize_for_citation_extraction("10 C.F.R. 50.46")
        assert "10 CFR" in result

    def test_unicode_dashes(self) -> None:
        result = normalize_for_citation_extraction("NUREG\u20130800")
        assert "NUREG-0800" in result

    def test_unicode_quotes_and_section_signs(self) -> None:
        result = normalize_for_citation_extraction("\u201c10 CFR 50.46\u201d")
        assert '"10 CFR 50.46"' in result

    def test_hard_line_breaks_collapsed(self) -> None:
        """Single newlines within a paragraph become spaces."""
        result = normalize_for_citation_extraction("10 CFR\n50.46")
        assert "10 CFR 50.46" in result

    def test_paragraph_breaks_preserved(self) -> None:
        """Double newlines (paragraph boundaries) are preserved."""
        result = normalize_for_citation_extraction("paragraph one\n\nparagraph two")
        assert "\n\n" in result

    def test_empty_input(self) -> None:
        assert normalize_for_citation_extraction("") == ""

    def test_no_mutations_on_clean_text(self) -> None:
        clean = "In accordance with 10 CFR 50.46(b)(1), the licensee shall..."
        assert normalize_for_citation_extraction(clean) == clean

    def test_code_of_federal_regulations_expanded(self) -> None:
        result = normalize_for_citation_extraction("Title 10, Code of Federal Regulations, Section 50.46")
        assert "10 CFR" in result
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_text_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/rag/adapters/ingestion/case/text_normalizer.py
"""Pre-processing for citation extraction: OCR cleanup and CFR normalization.

Improves regex recall by standardizing common surface variations of
regulatory references before span extraction runs.
"""

from __future__ import annotations

import re
import unicodedata


def normalize_for_citation_extraction(text: str) -> str:
    """Normalize *text* for citation extraction.

    Transformations (in order):
    1. Unicode normalization (NFC)
    2. Replace smart quotes, em-dashes, en-dashes with ASCII equivalents
    3. Normalize "C.F.R." / "C F R" → "CFR"
    4. Normalize "Code of Federal Regulations" → "CFR"
    5. Collapse single newlines to spaces (preserve paragraph breaks)
    6. Collapse runs of spaces to single space
    """
    if not text:
        return ""

    s = unicodedata.normalize("NFC", text)

    # Smart quotes → ASCII
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2018", "'").replace("\u2019", "'")

    # Em-dash / en-dash → hyphen
    s = s.replace("\u2013", "-").replace("\u2014", "-")

    # "Code of Federal Regulations" → CFR  (with optional Title N prefix)
    s = re.sub(
        r"Title\s+(\d+)\s*,?\s*Code\s+of\s+Federal\s+Regulations\s*,?\s*(?:Section\s+)?",
        r"\1 CFR ",
        s,
        flags=re.IGNORECASE,
    )

    # "C.F.R." → "CFR"
    s = re.sub(r"C\s*\.\s*F\s*\.\s*R\s*\.?", "CFR", s)

    # "C F R" (OCR split) → "CFR"
    s = re.sub(r"\bC\s+F\s+R\b", "CFR", s)

    # Collapse single newlines to spaces (preserve double-newline paragraph breaks)
    s = re.sub(r"(?<!\n)\n(?!\n)", " ", s)

    # Collapse runs of whitespace (except newlines) to single space
    s = re.sub(r"[^\S\n]+", " ", s)

    return s.strip()
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_text_normalizer.py -v`
Expected: PASS (all 10 tests)

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/case/text_normalizer.py tests/adapters/ingestion/case/test_text_normalizer.py
git commit -m "ingestion(case): add text normalizer for OCR cleanup before citation extraction"
```

---

### Task 3: CFR Section Extractor (Strong Refs)

The highest-value extractor. Handles `10 CFR 50.46(b)(1)`, `10CFR50.46`, `10 CFR §50.46`, and subsection variations.

**Files:**
- Create: `src/rag/adapters/ingestion/case/citation_extractor.py`
- Test: `tests/adapters/ingestion/case/test_citation_extractor.py`

**Step 1: Write the failing test**

```python
# tests/adapters/ingestion/case/test_citation_extractor.py
"""Tests for citation extraction."""

from __future__ import annotations

import pytest

from rag.domain.citations import CitationSpan
from rag.adapters.ingestion.case.citation_extractor import extract_cfr_sections


class TestExtractCfrSections:
    """Strong CFR section references with explicit '10 CFR' anchor."""

    def test_basic_section(self) -> None:
        spans = extract_cfr_sections("See 10 CFR 50.46 for requirements.")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46"
        assert spans[0].kind == "cfr"
        assert spans[0].raw == "10 CFR 50.46"
        assert spans[0].confidence == 0.95

    def test_section_with_subsections(self) -> None:
        spans = extract_cfr_sections("per 10 CFR 50.46(b)(1)(ii)")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46(b)(1)(ii)"
        assert spans[0].attrs["subsections"] == ["b", "1", "ii"]

    def test_section_sign_variant(self) -> None:
        spans = extract_cfr_sections("10 CFR §50.46")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46"

    def test_no_space_variant(self) -> None:
        """10CFR50.46 (no spaces) — rare but exists in ADAMS."""
        spans = extract_cfr_sections("10CFR50.46")
        assert len(spans) == 1
        assert spans[0].key == "cfr:10:50.46"

    def test_multiple_sections(self) -> None:
        text = "10 CFR 50.46 and 10 CFR 50.55a(g)(4) are applicable."
        spans = extract_cfr_sections(text)
        keys = {s.key for s in spans}
        assert "cfr:10:50.46" in keys
        assert "cfr:10:50.55a(g)(4)" in keys

    def test_title_not_10(self) -> None:
        """Handle non-Title-10 CFR refs (e.g., 40 CFR)."""
        spans = extract_cfr_sections("40 CFR 190.10")
        assert len(spans) == 1
        assert spans[0].key == "cfr:40:190.10"

    def test_span_offsets(self) -> None:
        text = "xxx 10 CFR 50.46 yyy"
        spans = extract_cfr_sections(text)
        assert text[spans[0].start : spans[0].end] == "10 CFR 50.46"

    def test_no_false_positive_on_plain_numbers(self) -> None:
        spans = extract_cfr_sections("The value was 50.46 percent.")
        assert len(spans) == 0

    def test_source_field_default(self) -> None:
        spans = extract_cfr_sections("10 CFR 50.46")
        assert spans[0].source_field == "content"

    def test_source_field_override(self) -> None:
        spans = extract_cfr_sections("10 CFR 50.46", source_field="title")
        assert spans[0].source_field == "title"

    def test_letter_suffix_section(self) -> None:
        """Sections like 50.55a — letter suffix on section number."""
        spans = extract_cfr_sections("10 CFR 50.55a")
        assert spans[0].key == "cfr:10:50.55a"

    def test_subsection_with_roman_numerals(self) -> None:
        spans = extract_cfr_sections("10 CFR 50.46(b)(5)(iii)")
        assert spans[0].attrs["subsections"] == ["b", "5", "iii"]
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractCfrSections -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# src/rag/adapters/ingestion/case/citation_extractor.py
"""Composable citation span extractors for NRC case documents.

Each ``extract_*`` function takes normalized text and returns a list of
:class:`~rag.domain.citations.CitationSpan` objects.  The functions are
designed to be composed into a pipeline::

    spans = (
        extract_cfr_sections(text)
        + extract_cfr_parts(text)
        + extract_dockets(text)
        + extract_adams_accessions(text)
        + extract_nuregs(text)
        + extract_generic_communications(text)
    )
"""

from __future__ import annotations

import re

from rag.adapters.ingestion.case.text_normalizer import normalize_for_citation_extraction
from rag.domain.citations import CitationSpan

# ---------------------------------------------------------------------------
# CFR section extraction
# ---------------------------------------------------------------------------

# Matches: "10 CFR 50.46(b)(1)(ii)", "10 CFR §50.46", "10CFR50.46"
# Groups: title, section (with letter suffix), subsections
_CFR_SECTION_RE = re.compile(
    r"(?P<title>\d{1,2})\s*CFR\s*§?\s*"
    r"(?P<section>\d+\.\d+[A-Za-z]?)"
    r"(?P<subs>(?:\([A-Za-z0-9]+\))*)"
)

_SUBSECTION_RE = re.compile(r"\(([A-Za-z0-9]+)\)")


def _parse_subsections(subs_raw: str) -> list[str]:
    """Parse '(b)(1)(ii)' into ['b', '1', 'ii']."""
    return _SUBSECTION_RE.findall(subs_raw)


def extract_cfr_sections(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract strong CFR section references with explicit title anchor.

    Returns spans for patterns like ``10 CFR 50.46(b)(1)`` — requires the
    title number (e.g. ``10``) to be present to avoid false positives on
    bare decimal numbers.
    """
    spans: list[CitationSpan] = []
    for m in _CFR_SECTION_RE.finditer(text):
        title = m.group("title")
        section = m.group("section")
        subs_raw = m.group("subs")
        subs = _parse_subsections(subs_raw)

        key = f"cfr:{title}:{section}"
        if subs_raw:
            key += subs_raw.lower()

        spans.append(
            CitationSpan(
                kind="cfr",
                raw=m.group(0),
                key=key,
                start=m.start(),
                end=m.end(),
                confidence=0.95,
                source_field=source_field,
                attrs={
                    "title": int(title),
                    "part": int(section.split(".")[0]),
                    "section": section.split(".")[1].rstrip("abcdefghijklmnopqrstuvwxyz"),
                    "section_full": section,
                    "subsections": subs,
                },
            )
        )
    return spans
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractCfrSections -v`
Expected: PASS (all 12 tests)

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_citation_extractor.py
git commit -m "ingestion(case): add CFR section citation extractor with strong-ref regex"
```

---

### Task 4: CFR Part and Appendix Extractors

**Files:**
- Modify: `src/rag/adapters/ingestion/case/citation_extractor.py`
- Modify: `tests/adapters/ingestion/case/test_citation_extractor.py`

**Step 1: Write the failing tests**

Append to `tests/adapters/ingestion/case/test_citation_extractor.py`:

```python
from rag.adapters.ingestion.case.citation_extractor import (
    extract_cfr_parts,
    extract_cfr_appendices,
)


class TestExtractCfrParts:
    def test_basic_part(self) -> None:
        spans = extract_cfr_parts("10 CFR Part 50")
        assert len(spans) == 1
        assert spans[0].key == "cfrpart:10:50"
        assert spans[0].kind == "cfrpart"
        assert spans[0].confidence == 0.90

    def test_part_without_title(self) -> None:
        """'Part 50' without '10 CFR' prefix — lower confidence."""
        spans = extract_cfr_parts("Part 50 requires...")
        assert len(spans) == 1
        assert spans[0].key == "cfrpart:10:50"
        assert spans[0].confidence == 0.70

    def test_multiple_parts(self) -> None:
        spans = extract_cfr_parts("10 CFR Part 50 and 10 CFR Part 21")
        keys = {s.key for s in spans}
        assert keys == {"cfrpart:10:50", "cfrpart:10:21"}

    def test_non_title_10_part(self) -> None:
        spans = extract_cfr_parts("40 CFR Part 190")
        assert spans[0].key == "cfrpart:40:190"


class TestExtractCfrAppendices:
    def test_appendix_b_to_part_50(self) -> None:
        spans = extract_cfr_appendices("10 CFR Part 50, Appendix B")
        assert len(spans) == 1
        assert spans[0].key == "cfrapp:10:50:appendix-b"
        assert spans[0].kind == "cfrapp"

    def test_appendix_a_to_part_100(self) -> None:
        spans = extract_cfr_appendices("Appendix A to 10 CFR Part 100")
        assert len(spans) == 1
        assert spans[0].key == "cfrapp:10:100:appendix-a"

    def test_no_false_positive_on_bare_appendix(self) -> None:
        """'Appendix B' alone without a part reference is not extracted."""
        spans = extract_cfr_appendices("See Appendix B for details.")
        assert len(spans) == 0
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractCfrParts tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractCfrAppendices -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

Append to `src/rag/adapters/ingestion/case/citation_extractor.py`:

```python
# ---------------------------------------------------------------------------
# CFR part extraction
# ---------------------------------------------------------------------------

# "10 CFR Part 50" — with explicit title
_CFR_PART_STRONG_RE = re.compile(
    r"(?P<title>\d{1,2})\s*CFR\s+Part\s+(?P<part>\d+)",
    re.IGNORECASE,
)

# "Part 50" — without title (assumes Title 10)
# No lookbehind needed: the `covered` set in extract_cfr_parts() prevents
# double-matching "Part 50" inside an already-matched "10 CFR Part 50".
_CFR_PART_WEAK_RE = re.compile(
    r"Part\s+(?P<part>\d+)",
    re.IGNORECASE,
)


def extract_cfr_parts(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract CFR part references like '10 CFR Part 50' or bare 'Part 50'."""
    spans: list[CitationSpan] = []
    # Track which offsets are already covered by strong matches
    covered: set[int] = set()

    for m in _CFR_PART_STRONG_RE.finditer(text):
        title = m.group("title")
        part = m.group("part")
        spans.append(
            CitationSpan(
                kind="cfrpart",
                raw=m.group(0),
                key=f"cfrpart:{title}:{part}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"title": int(title), "part": int(part)},
            )
        )
        covered.update(range(m.start(), m.end()))

    for m in _CFR_PART_WEAK_RE.finditer(text):
        if m.start() in covered:
            continue
        part = m.group("part")
        spans.append(
            CitationSpan(
                kind="cfrpart",
                raw=m.group(0),
                key=f"cfrpart:10:{part}",
                start=m.start(),
                end=m.end(),
                confidence=0.70,
                source_field=source_field,
                attrs={"title": 10, "part": int(part)},
            )
        )

    return spans


# ---------------------------------------------------------------------------
# CFR appendix extraction
# ---------------------------------------------------------------------------

# "10 CFR Part 50, Appendix B" or "Appendix B to 10 CFR Part 100"
_CFR_APPENDIX_RE = re.compile(
    r"(?:"
    r"(?P<title1>\d{1,2})\s*CFR\s+Part\s+(?P<part1>\d+)\s*,?\s+Appendix\s+(?P<letter1>[A-Z])"
    r"|"
    r"Appendix\s+(?P<letter2>[A-Z])\s+to\s+(?P<title2>\d{1,2})\s*CFR\s+Part\s+(?P<part2>\d+)"
    r")",
    re.IGNORECASE,
)


def extract_cfr_appendices(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract CFR appendix references like '10 CFR Part 50, Appendix B'."""
    spans: list[CitationSpan] = []
    for m in _CFR_APPENDIX_RE.finditer(text):
        title = m.group("title1") or m.group("title2")
        part = m.group("part1") or m.group("part2")
        letter = (m.group("letter1") or m.group("letter2")).lower()

        spans.append(
            CitationSpan(
                kind="cfrapp",
                raw=m.group(0),
                key=f"cfrapp:{title}:{part}:appendix-{letter}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"title": int(title), "part": int(part), "appendix": letter},
            )
        )
    return spans
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py -v`
Expected: PASS (all tests including Task 3 tests)

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_citation_extractor.py
git commit -m "ingestion(case): add CFR part and appendix citation extractors"
```

---

### Task 5: Docket Number Extractor

Docket numbers appear as `05000247`, `50-247`, and `Docket No. 50-247`. The fixed-width 8-digit form and the hyphenated form must canonicalize to the same key.

**Files:**
- Modify: `src/rag/adapters/ingestion/case/citation_extractor.py`
- Modify: `tests/adapters/ingestion/case/test_citation_extractor.py`

**Step 1: Write the failing tests**

```python
from rag.adapters.ingestion.case.citation_extractor import extract_dockets


class TestExtractDockets:
    def test_docket_no_form(self) -> None:
        spans = extract_dockets("Docket No. 50-247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"
        assert spans[0].kind == "docket"
        assert spans[0].confidence == 0.90

    def test_docket_number_form(self) -> None:
        spans = extract_dockets("Docket Number 50-247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"

    def test_fixed_width_form(self) -> None:
        """ADAMS metadata uses 8-digit form like 05000247."""
        spans = extract_dockets("05000247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"

    def test_fixed_width_70_series(self) -> None:
        spans = extract_dockets("07007002")
        assert len(spans) == 1
        assert spans[0].key == "docket:70-7002"

    def test_hyphenated_form(self) -> None:
        spans = extract_dockets("50-247")
        assert len(spans) == 1
        assert spans[0].key == "docket:50-247"
        assert spans[0].confidence == 0.75  # lower — more ambiguous

    def test_docket_nos_plural(self) -> None:
        spans = extract_dockets("Docket Nos. 50-247 and 50-286")
        keys = {s.key for s in spans}
        assert keys == {"docket:50-247", "docket:50-286"}

    def test_no_false_positive_on_dates(self) -> None:
        """Dates like '95-07' should not match as dockets."""
        spans = extract_dockets("dated 95-07 and filed on 97-10")
        assert len(spans) == 0

    def test_no_false_positive_on_short_numbers(self) -> None:
        spans = extract_dockets("page 247 of the report")
        assert len(spans) == 0
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractDockets -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

```python
# ---------------------------------------------------------------------------
# Docket number extraction
# ---------------------------------------------------------------------------

# "Docket No. 50-247" or "Docket Nos. 50-247 and 50-286"
_DOCKET_EXPLICIT_RE = re.compile(
    r"Docket\s+(?:Nos?\.?|Numbers?)\s+"
    r"(?P<docket>\d{1,2}-\d+)",
    re.IGNORECASE,
)

# Fixed-width 8-digit NRC docket: 05000247 → 50-247
# Valid facility-type prefixes: 050, 070, 072, 030, 040
_DOCKET_FIXED_RE = re.compile(
    r"\b(?P<docket>0[3457][02]\d{5})\b"
)

# Bare hyphenated form: 50-247, 70-7002
# Only match NRC facility-type prefixes to reduce false positives
_DOCKET_HYPHEN_RE = re.compile(
    r"\b(?P<docket>(?:50|70|72|30|40)-\d{3,5})\b"
)


def _normalize_fixed_docket(digits: str) -> str:
    """Convert 8-digit docket '05000247' to hyphenated '50-247'."""
    # Strip leading zero, split at position 3
    facility_type = str(int(digits[:3]))
    number = str(int(digits[3:]))
    return f"{facility_type}-{number}"


def extract_dockets(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract NRC docket number references."""
    spans: list[CitationSpan] = []
    covered: set[int] = set()

    # Explicit "Docket No." form (highest confidence)
    for m in _DOCKET_EXPLICIT_RE.finditer(text):
        docket = m.group("docket")
        spans.append(
            CitationSpan(
                kind="docket",
                raw=m.group(0),
                key=f"docket:{docket}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"docket_number": docket},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Fixed-width 8-digit form
    for m in _DOCKET_FIXED_RE.finditer(text):
        if m.start() in covered:
            continue
        docket_hyp = _normalize_fixed_docket(m.group("docket"))
        spans.append(
            CitationSpan(
                kind="docket",
                raw=m.group(0),
                key=f"docket:{docket_hyp}",
                start=m.start(),
                end=m.end(),
                confidence=0.85,
                source_field=source_field,
                attrs={"docket_number": docket_hyp, "raw_fixed": m.group("docket")},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Bare hyphenated form (lower confidence — more ambiguous)
    for m in _DOCKET_HYPHEN_RE.finditer(text):
        if m.start() in covered:
            continue
        docket = m.group("docket")
        spans.append(
            CitationSpan(
                kind="docket",
                raw=m.group(0),
                key=f"docket:{docket}",
                start=m.start(),
                end=m.end(),
                confidence=0.75,
                source_field=source_field,
                attrs={"docket_number": docket},
            )
        )

    return spans
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractDockets -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_citation_extractor.py
git commit -m "ingestion(case): add docket number citation extractor"
```

---

### Task 6: ADAMS Accession Number Extractor

Modern ADAMS accessions look like `ML021910673`. Legacy ones are 10-digit numeric strings.

**Files:**
- Modify: `src/rag/adapters/ingestion/case/citation_extractor.py`
- Modify: `tests/adapters/ingestion/case/test_citation_extractor.py`

**Step 1: Write the failing tests**

```python
from rag.adapters.ingestion.case.citation_extractor import extract_adams_accessions


class TestExtractAdamsAccessions:
    def test_modern_accession(self) -> None:
        spans = extract_adams_accessions("See ML021910673 for details.")
        assert len(spans) == 1
        assert spans[0].key == "adams:ML021910673"
        assert spans[0].kind == "adams"
        assert spans[0].confidence == 0.90

    def test_modern_accession_various_prefixes(self) -> None:
        """ML is most common but other prefixes exist."""
        for acc in ["ML021910673", "ML20108D163"]:
            spans = extract_adams_accessions(acc)
            assert len(spans) == 1
            assert spans[0].key == f"adams:{acc}"

    def test_multiple_accessions(self) -> None:
        text = "Documents ML021910673 and ML20108D163 were reviewed."
        spans = extract_adams_accessions(text)
        assert len(spans) == 2

    def test_legacy_numeric_accession(self) -> None:
        """10-digit numeric legacy accession like 8111110271."""
        spans = extract_adams_accessions("document 8111110271 was filed")
        assert len(spans) == 1
        assert spans[0].key == "adamslegacy:8111110271"
        assert spans[0].confidence == 0.60

    def test_no_false_positive_on_phone_numbers(self) -> None:
        """Phone numbers should not match."""
        spans = extract_adams_accessions("call (301) 564-3309")
        assert len(spans) == 0

    def test_no_false_positive_on_dates(self) -> None:
        spans = extract_adams_accessions("on 20060412 the report was filed")
        # This is tricky — 8 digits, not 10
        assert len(spans) == 0

    def test_accession_case_preserved(self) -> None:
        spans = extract_adams_accessions("ml021910673")
        assert len(spans) == 1
        assert spans[0].key == "adams:ML021910673"  # uppercased
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractAdamsAccessions -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# ---------------------------------------------------------------------------
# ADAMS accession number extraction
# ---------------------------------------------------------------------------

# Modern ADAMS: ML + exactly 9 alphanumeric chars
# Format: ML + 2-digit year + 3-digit Julian day + 1 alpha + 3-digit sequence
# e.g. ML021910673, ML20108D163, ML051600165
_ADAMS_MODERN_RE = re.compile(
    r"\b(?P<acc>[Mm][Ll][0-9A-Za-z]{9})\b"
)

# Legacy ADAMS: exactly 10 digits, first digit typically 7-9 (older era)
_ADAMS_LEGACY_RE = re.compile(
    r"\b(?P<acc>[789]\d{9})\b"
)


def extract_adams_accessions(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract ADAMS accession number references."""
    spans: list[CitationSpan] = []
    covered: set[int] = set()

    # Modern ML-prefixed accessions
    for m in _ADAMS_MODERN_RE.finditer(text):
        acc = m.group("acc").upper()
        spans.append(
            CitationSpan(
                kind="adams",
                raw=m.group(0),
                key=f"adams:{acc}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"accession_number": acc},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Legacy 10-digit numeric accessions
    for m in _ADAMS_LEGACY_RE.finditer(text):
        if m.start() in covered:
            continue
        acc = m.group("acc")
        spans.append(
            CitationSpan(
                kind="adams",
                raw=m.group(0),
                key=f"adamslegacy:{acc}",
                start=m.start(),
                end=m.end(),
                confidence=0.60,
                source_field=source_field,
                attrs={"accession_number": acc, "is_legacy": True},
            )
        )

    return spans
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractAdamsAccessions -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_citation_extractor.py
git commit -m "ingestion(case): add ADAMS accession number citation extractor"
```

---

### Task 7: NUREG and Generic Communication Extractors (RIS/GL/IN)

**Files:**
- Modify: `src/rag/adapters/ingestion/case/citation_extractor.py`
- Modify: `tests/adapters/ingestion/case/test_citation_extractor.py`

**Step 1: Write the failing tests**

```python
from rag.adapters.ingestion.case.citation_extractor import (
    extract_nuregs,
    extract_generic_communications,
)


class TestExtractNuregs:
    def test_nureg_basic(self) -> None:
        spans = extract_nuregs("NUREG-0800")
        assert len(spans) == 1
        assert spans[0].key == "nureg:0800"
        assert spans[0].kind == "nureg"

    def test_nureg_with_series(self) -> None:
        spans = extract_nuregs("NUREG/BR-0073")
        assert len(spans) == 1
        assert spans[0].key == "nureg:BR-0073"

    def test_nureg_cr(self) -> None:
        spans = extract_nuregs("NUREG/CR-6850")
        assert len(spans) == 1
        assert spans[0].key == "nureg:CR-6850"

    def test_reg_guide(self) -> None:
        spans = extract_nuregs("Regulatory Guide 1.174")
        assert len(spans) == 1
        assert spans[0].key == "rg:1.174"

    def test_reg_guide_abbreviated(self) -> None:
        spans = extract_nuregs("RG 1.174")
        assert len(spans) == 1
        assert spans[0].key == "rg:1.174"


class TestExtractGenericCommunications:
    def test_ris(self) -> None:
        spans = extract_generic_communications("RIS 2001-05")
        assert len(spans) == 1
        assert spans[0].key == "ris:2001-05"
        assert spans[0].kind == "ris"

    def test_ris_full_name(self) -> None:
        spans = extract_generic_communications("Regulatory Issue Summary 2001-05")
        assert len(spans) == 1
        assert spans[0].key == "ris:2001-05"

    def test_generic_letter(self) -> None:
        spans = extract_generic_communications("Generic Letter 95-07")
        assert len(spans) == 1
        assert spans[0].key == "gl:95-07"

    def test_gl_abbreviated(self) -> None:
        spans = extract_generic_communications("GL 95-07")
        assert len(spans) == 1
        assert spans[0].key == "gl:95-07"

    def test_information_notice(self) -> None:
        spans = extract_generic_communications("Information Notice 97-10")
        assert len(spans) == 1
        assert spans[0].key == "in:97-10"

    def test_in_abbreviated(self) -> None:
        spans = extract_generic_communications("IN 97-10")
        assert len(spans) == 1
        assert spans[0].key == "in:97-10"
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractNuregs tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractGenericCommunications -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# ---------------------------------------------------------------------------
# NUREG and Regulatory Guide extraction
# ---------------------------------------------------------------------------

# NUREG-0800, NUREG/BR-0073, NUREG/CR-6850
_NUREG_RE = re.compile(
    r"\bNUREG(?:/(?P<series>[A-Z]{1,3}))?-(?P<number>\d{4})\b"
)

# "Regulatory Guide 1.174" or "Reg Guide 1.174" or "RG 1.174"
_REG_GUIDE_RE = re.compile(
    r"(?:Reg(?:ulatory)?\s+Guide|RG)\s+(?P<number>\d+\.\d+)",
    re.IGNORECASE,
)


def extract_nuregs(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract NUREG and Regulatory Guide references."""
    spans: list[CitationSpan] = []

    for m in _NUREG_RE.finditer(text):
        series = m.group("series")
        number = m.group("number")
        key = f"nureg:{series}-{number}" if series else f"nureg:{number}"
        spans.append(
            CitationSpan(
                kind="nureg",
                raw=m.group(0),
                key=key,
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"series": series, "number": number},
            )
        )

    for m in _REG_GUIDE_RE.finditer(text):
        number = m.group("number")
        spans.append(
            CitationSpan(
                kind="rg",
                raw=m.group(0),
                key=f"rg:{number}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"number": number},
            )
        )

    return spans


# ---------------------------------------------------------------------------
# Generic communication extraction (RIS, GL, IN)
# ---------------------------------------------------------------------------

_GENERIC_COMM_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ris",
        re.compile(
            r"(?:Regulatory\s+Issue\s+Summary|RIS)\s+(?P<number>\d{4}-\d{2})",
            re.IGNORECASE,
        ),
    ),
    (
        "gl",
        re.compile(
            r"(?:Generic\s+Letter|GL)\s+(?P<number>\d{2}-\d{2})",
            re.IGNORECASE,
        ),
    ),
    (
        "in",
        re.compile(
            r"(?:Information\s+Notice|IN)\s+(?P<number>\d{2}-\d{2})",
            re.IGNORECASE,
        ),
    ),
]


def extract_generic_communications(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract RIS, Generic Letter, and Information Notice references."""
    spans: list[CitationSpan] = []
    for kind, pattern in _GENERIC_COMM_PATTERNS:
        for m in pattern.finditer(text):
            number = m.group("number")
            spans.append(
                CitationSpan(
                    kind=kind,
                    raw=m.group(0),
                    key=f"{kind}:{number}",
                    start=m.start(),
                    end=m.end(),
                    confidence=0.85,
                    source_field=source_field,
                    attrs={"number": number},
                )
            )
    return spans
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py -v`
Expected: PASS (all tests across all test classes)

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_citation_extractor.py
git commit -m "ingestion(case): add NUREG, Reg Guide, and generic communication citation extractors"
```

---

### Task 8: Citation Pipeline — Compose, Score, Dedupe

The pipeline function composes all extractors, runs text normalization, deduplicates by canonical key (keeping the highest-confidence span), and optionally adds context windows.

**Files:**
- Modify: `src/rag/adapters/ingestion/case/citation_extractor.py`
- Modify: `tests/adapters/ingestion/case/test_citation_extractor.py`

**Step 1: Write the failing tests**

```python
from rag.adapters.ingestion.case.citation_extractor import extract_all_citations


class TestExtractAllCitations:
    """Integration tests for the full pipeline."""

    def test_mixed_citation_types(self) -> None:
        text = (
            "Per 10 CFR 50.46(b)(1), Docket No. 50-247, "
            "accession ML021910673, and NUREG-0800."
        )
        spans = extract_all_citations(text)
        kinds = {s.kind for s in spans}
        assert "cfr" in kinds
        assert "docket" in kinds
        assert "adams" in kinds
        assert "nureg" in kinds

    def test_dedup_by_key(self) -> None:
        """Same citation appearing twice — keep only the highest confidence."""
        text = "10 CFR 50.46 requires ... per 10 CFR 50.46(b)."
        spans = extract_all_citations(text)
        cfr_keys = [s.key for s in spans if s.kind == "cfr"]
        # Both matches have different keys (one has subsection)
        # so both should be present
        assert "cfr:10:50.46" in cfr_keys
        assert "cfr:10:50.46(b)" in cfr_keys

    def test_exact_duplicate_deduped(self) -> None:
        text = "10 CFR 50.46 then again 10 CFR 50.46"
        spans = extract_all_citations(text)
        cfr_spans = [s for s in spans if s.key == "cfr:10:50.46"]
        assert len(cfr_spans) == 1  # deduped

    def test_context_window_populated(self) -> None:
        text = "In accordance with 10 CFR 50.46 requirements for ECCS."
        spans = extract_all_citations(text)
        assert spans[0].context is not None
        assert "accordance" in spans[0].context

    def test_empty_text(self) -> None:
        assert extract_all_citations("") == []

    def test_no_citations(self) -> None:
        assert extract_all_citations("This is a plain sentence with no references.") == []

    def test_text_normalization_applied(self) -> None:
        """OCR-degraded text should still yield citations."""
        text = "per 10 C.F.R. 50.46"
        spans = extract_all_citations(text)
        assert len(spans) >= 1
        assert spans[0].kind == "cfr"

    def test_source_field_propagated(self) -> None:
        spans = extract_all_citations("10 CFR 50.46", source_field="title")
        assert all(s.source_field == "title" for s in spans)

    def test_sorted_by_start_offset(self) -> None:
        text = "NUREG-0800 and 10 CFR 50.46 and Docket No. 50-247"
        spans = extract_all_citations(text)
        starts = [s.start for s in spans]
        assert starts == sorted(starts)
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py::TestExtractAllCitations -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# ---------------------------------------------------------------------------
# Pipeline: compose, dedupe, context
# ---------------------------------------------------------------------------

_CONTEXT_WINDOW = 60  # chars on each side of the match


def _add_context(span: CitationSpan, text: str) -> CitationSpan:
    """Return a copy of *span* with a context window from *text*."""
    ctx_start = max(0, span.start - _CONTEXT_WINDOW)
    ctx_end = min(len(text), span.end + _CONTEXT_WINDOW)
    context = text[ctx_start:ctx_end]
    if ctx_start > 0:
        context = "..." + context
    if ctx_end < len(text):
        context = context + "..."
    return CitationSpan(
        kind=span.kind,
        raw=span.raw,
        key=span.key,
        start=span.start,
        end=span.end,
        confidence=span.confidence,
        source_field=span.source_field,
        context=context,
        attrs=span.attrs,
    )


def _dedupe_spans(spans: list[CitationSpan]) -> list[CitationSpan]:
    """Deduplicate by canonical key, keeping the highest-confidence span."""
    best: dict[str, CitationSpan] = {}
    for span in spans:
        existing = best.get(span.key)
        if existing is None or span.confidence > existing.confidence:
            best[span.key] = span
    return sorted(best.values(), key=lambda s: s.start)


def extract_all_citations(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Run all extractors on *text* and return deduped, sorted spans.

    Applies text normalization before extraction.  The returned spans
    have offsets into the *normalized* text, not the original.
    """
    if not text.strip():
        return []

    text_norm = normalize_for_citation_extraction(text)

    all_spans: list[CitationSpan] = []
    all_spans.extend(extract_cfr_sections(text_norm, source_field=source_field))
    all_spans.extend(extract_cfr_parts(text_norm, source_field=source_field))
    all_spans.extend(extract_cfr_appendices(text_norm, source_field=source_field))
    all_spans.extend(extract_dockets(text_norm, source_field=source_field))
    all_spans.extend(extract_adams_accessions(text_norm, source_field=source_field))
    all_spans.extend(extract_nuregs(text_norm, source_field=source_field))
    all_spans.extend(extract_generic_communications(text_norm, source_field=source_field))

    deduped = _dedupe_spans(all_spans)
    return [_add_context(s, text_norm) for s in deduped]
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_extractor.py -v`
Expected: PASS (all tests)

**Step 5: Run full test suite to check for regressions**

Run: `./scripts/py -m pytest tests/ -v --tb=short`
Expected: PASS

**Step 6: Commit**

```bash
git add src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_citation_extractor.py
git commit -m "ingestion(case): add citation pipeline with dedup, context windows, and text normalization"
```

---

### Task 9: Add `citation_keys` to CaseMetadata and Integration with Normalizer

Wire the citation extractor into the existing normalizer. The normalizer already calls `extract_case_metadata()` — we add citation extraction there and surface `citation_keys` in the frontmatter.

**Files:**
- Modify: `src/rag/domain/case_documents.py` (add `citation_keys` field to `CaseMetadata`)
- Modify: `src/rag/adapters/ingestion/case/normalizer.py` (use `extract_all_citations`)
- Modify: `tests/adapters/ingestion/case/test_normalizer.py` (verify citations in frontmatter)

**Step 1: Write the failing test**

Append to `tests/adapters/ingestion/case/test_normalizer.py`:

```python
class TestCitationIntegration:
    """Citation extraction is integrated into the normalizer pipeline."""

    def test_frontmatter_has_citation_keys(self) -> None:
        from rag.domain.case_documents import CaseDocument
        from rag.adapters.ingestion.case.normalizer import (
            CaseNormalizationConfig,
            normalize_case_document_to_markdown,
        )
        import yaml

        doc = CaseDocument(
            accession_number="ML99999A001",
            title="Test - 10 CFR 50.46 Compliance",
            document_type="Inspection Report",
            content="Per 10 CFR 50.46(b)(1) and NUREG-0800, Docket No. 50-247.",
        )
        md = normalize_case_document_to_markdown(doc, CaseNormalizationConfig())
        # Parse frontmatter
        fm_text = md.split("---")[1]
        fm = yaml.safe_load(fm_text)
        assert "citation_keys" in fm
        keys = fm["citation_keys"]
        assert any("cfr:10:50.46" in k for k in keys)
        assert any("nureg:0800" in k for k in keys)
        assert any("docket:50-247" in k for k in keys)

    def test_metadata_has_citation_keys(self) -> None:
        from rag.domain.case_documents import CaseDocument
        from rag.adapters.ingestion.case.normalizer import extract_case_metadata

        doc = CaseDocument(
            accession_number="ML99999A001",
            title="Test Document",
            document_type="Letter",
            content="10 CFR 50.46 and 10 CFR Part 50, Appendix B.",
        )
        meta = extract_case_metadata(doc)
        assert "cfr:10:50.46" in meta.citation_keys
        assert "cfrapp:10:50:appendix-b" in meta.citation_keys
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_normalizer.py::TestCitationIntegration -v`
Expected: FAIL (`AttributeError: 'CaseMetadata' has no attribute 'citation_keys'`)

**Step 3a: Add `citation_keys` to `CaseMetadata`**

In [case_documents.py:157](src/rag/domain/case_documents.py#L157), add the field after `dockets`:

```python
    dockets: tuple[str, ...] = ()
    citation_keys: tuple[str, ...] = ()  # canonical keys from CitationSpan extraction
```

Update `to_dict()` to include it:

```python
    def to_dict(self) -> dict[str, object]:
        self.validate()
        return {
            # ... existing fields ...
            "dockets": list(self.dockets),
            "citation_keys": list(self.citation_keys),
        }
```

**Step 3b: Integrate into normalizer**

In [normalizer.py](src/rag/adapters/ingestion/case/normalizer.py), modify `extract_case_metadata()`:

```python
def extract_case_metadata(doc: CaseDocument) -> CaseMetadata:
    classification = classify_case_document(
        title=doc.title,
        doc_type=doc.document_type,
    )

    search_text = doc.title + (" " + doc.content if doc.content else "")

    parts: set[str] = set()
    for m in _PART_RE.finditer(search_text):
        parts.add(m.group(1))

    sections: set[str] = set()
    for m in _SECTION_RE.finditer(search_text):
        sections.add(m.group(2))

    # Citation extraction
    from rag.adapters.ingestion.case.citation_extractor import extract_all_citations

    title_citations = extract_all_citations(doc.title, source_field="title")
    content_citations = extract_all_citations(doc.content or "", source_field="content")
    all_citation_keys = sorted({s.key for s in title_citations + content_citations})

    return CaseMetadata(
        case_category=classification.category,
        case_subcategory=classification.subcategory,
        case_category_method=classification.method,
        case_category_confidence=classification.confidence,
        case_category_reasons=classification.reasons,
        case_signals=(),
        regulation_parts=tuple(sorted(parts)),
        regulation_sections=tuple(sorted(sections)),
        dockets=doc.docket_numbers,
        citation_keys=tuple(all_citation_keys),
    )
```

In the frontmatter builder (`_build_frontmatter`), add `citation_keys` to the list fields section:

```python
    for key, items in [
        ("dockets", list(meta.dockets)),
        ("regulation_parts", list(meta.regulation_parts)),
        ("regulation_sections", list(meta.regulation_sections)),
        ("citation_keys", list(meta.citation_keys)),
        ("cross_references", cross_references),
    ]:
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_normalizer.py -v`
Expected: PASS (all tests including new ones — existing tests should still pass since `citation_keys` defaults to `()`)

**Step 5: Run broader validation**

Run: `./scripts/py -m pytest tests/ -v --tb=short`
Expected: PASS

Run: `./scripts/py -m mypy src/rag`
Expected: PASS (or no new errors)

**Step 6: Commit**

```bash
git add src/rag/domain/case_documents.py src/rag/adapters/ingestion/case/normalizer.py src/rag/adapters/ingestion/case/citation_extractor.py tests/adapters/ingestion/case/test_normalizer.py
git commit -m "ingestion(case): integrate citation extraction into normalizer pipeline and CaseMetadata"
```

---

### Task 10: Golden File Tests Against Real ADAMS Samples

Run the citation extractor against actual fetched sample documents to validate recall and spot false positives. This task creates a snapshot-style test.

**Files:**
- Create: `tests/adapters/ingestion/case/test_citation_golden.py`

**Step 1: Write the test**

```python
# tests/adapters/ingestion/case/test_citation_golden.py
"""Golden-file tests: run citation extraction against real ADAMS samples.

These tests validate extraction quality against known documents.
They don't snapshot exact output (too brittle) but assert minimum
expected citations and absence of known false positives.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag.adapters.ingestion.case.citation_extractor import extract_all_citations
from rag.adapters.ingestion.case.document_adapter import adams_document_to_case_document
from rag.adapters.ingestion.case.normalizer import extract_case_metadata
from rag.ports.nrc_adams_client import AdamsDocument

SAMPLES_DIR = Path("data/adams_samples")


def _load_sample(accession: str) -> AdamsDocument:
    path = SAMPLES_DIR / f"{accession}.json"
    if not path.exists():
        pytest.skip(f"Sample {accession} not available")
    with path.open() as f:
        data = json.load(f)
    doc_data = data["document"]
    return AdamsDocument(**doc_data)


@pytest.mark.skipif(
    not SAMPLES_DIR.exists(),
    reason="Sample data directory not present",
)
class TestCitationGoldenFiles:
    def test_modern_doc_with_cfr_refs(self) -> None:
        """ML20137W401 — certification letter that cites 10 CFR 76.68."""
        adams_doc = _load_sample("ML20137W401")
        case_doc = adams_document_to_case_document(adams_doc)
        meta = extract_case_metadata(case_doc)
        # This doc references 10 CFR 76.68, 10 CFR 2.790, 10 CFR 76.33, etc.
        assert len(meta.citation_keys) >= 3
        assert any("cfr:" in k for k in meta.citation_keys)

    def test_legacy_doc_no_content(self) -> None:
        """771870188 — legacy doc with no content. Should not crash."""
        adams_doc = _load_sample("771870188")
        case_doc = adams_document_to_case_document(adams_doc)
        meta = extract_case_metadata(case_doc)
        # No content → no citations (or just from title)
        assert isinstance(meta.citation_keys, tuple)

    def test_docket_in_metadata(self) -> None:
        """ML20137W401 — docket 07007002 in metadata."""
        adams_doc = _load_sample("ML20137W401")
        case_doc = adams_document_to_case_document(adams_doc)
        content = case_doc.content or ""
        spans = extract_all_citations(content)
        # Should pick up the docket from content if it appears there
        docket_spans = [s for s in spans if s.kind == "docket"]
        # The doc_number field has 07007002 — check the extractor finds it
        if "07007002" in content or "70-7002" in content:
            assert len(docket_spans) >= 1

    def test_citation_count_summary(self) -> None:
        """Aggregate: run on all available samples, report stats."""
        docs_with_citations = 0
        total_citations = 0
        by_kind: dict[str, int] = {}

        for path in sorted(SAMPLES_DIR.glob("ML*.json")):
            with path.open() as f:
                data = json.load(f)
            adams_doc = AdamsDocument(**data["document"])
            case_doc = adams_document_to_case_document(adams_doc)
            content = case_doc.title + " " + (case_doc.content or "")
            spans = extract_all_citations(content)
            if spans:
                docs_with_citations += 1
            total_citations += len(spans)
            for s in spans:
                by_kind[s.kind] = by_kind.get(s.kind, 0) + 1

        # Soft assertions — these thresholds can be tuned as we iterate
        assert docs_with_citations >= 1, "Expected at least some docs with citations"
        # Print stats for manual review (visible in pytest -v output)
        print(f"\n--- Citation Stats ---")
        print(f"Samples scanned: {len(list(SAMPLES_DIR.glob('ML*.json')))}")
        print(f"Docs with citations: {docs_with_citations}")
        print(f"Total citations: {total_citations}")
        for kind, count in sorted(by_kind.items()):
            print(f"  {kind}: {count}")
```

**Step 2: Run tests**

Run: `./scripts/py -m pytest tests/adapters/ingestion/case/test_citation_golden.py -v -s`
Expected: PASS (the `-s` flag shows the stats summary)

**Step 3: Review output**

Inspect the citation stats printout. Look for:
- Are CFR citations being found in docs that reference them?
- Any surprising false positive kinds?
- Any docs that should have citations but don't?

**Step 4: Commit**

```bash
git add tests/adapters/ingestion/case/test_citation_golden.py
git commit -m "test(ingestion): add golden-file citation extraction tests against ADAMS samples"
```

---

### Task 11: Lint, Typecheck, Full Validation

**Step 1: Run ruff**

Run: `./scripts/py -m ruff check src/rag/domain/citations.py src/rag/adapters/ingestion/case/citation_extractor.py src/rag/adapters/ingestion/case/text_normalizer.py`

Fix any issues.

**Step 2: Run ruff format**

Run: `./scripts/py -m ruff format src/rag/domain/citations.py src/rag/adapters/ingestion/case/citation_extractor.py src/rag/adapters/ingestion/case/text_normalizer.py`

**Step 3: Run mypy**

Run: `./scripts/py -m mypy src/rag`

Fix any type errors.

**Step 4: Run full test suite**

Run: `make test`

All tests must pass.

**Step 5: Commit fixes if any**

```bash
git add -u
git commit -m "chore: fix lint and type errors in citation extraction"
```

---

## File Summary

| Action | File |
|--------|------|
| Create | `src/rag/domain/citations.py` |
| Create | `src/rag/adapters/ingestion/case/text_normalizer.py` |
| Create | `src/rag/adapters/ingestion/case/citation_extractor.py` |
| Create | `tests/domain/test_citations.py` |
| Create | `tests/adapters/ingestion/case/test_text_normalizer.py` |
| Create | `tests/adapters/ingestion/case/test_citation_golden.py` |
| Modify | `src/rag/domain/case_documents.py` (add `citation_keys` to `CaseMetadata`) |
| Modify | `src/rag/adapters/ingestion/case/normalizer.py` (integrate extractor) |
| Modify | `tests/adapters/ingestion/case/test_citation_extractor.py` (all extractor tests) |
| Modify | `tests/adapters/ingestion/case/test_normalizer.py` (integration tests) |

## Design Decisions

1. **Offsets are into normalized text** — not the original. This is simpler and sufficient for v1. The spec notes raw-to-norm mapping is optional.
2. **No weak CFR refs in v1** — bare `50.46` without a `10 CFR` anchor is skipped to avoid false positives. Can be added later with anchor-window logic.
3. **Docket false-positive mitigation** — only match NRC facility-type prefixes (50, 70, 72, 30, 40) for the bare hyphenated form. Note: "Docket Nos. X and Y" only captures X at high confidence; Y falls through to the bare-hyphenated pattern at 0.75. Acceptable for v1.
4. **Legacy ADAMS accessions** — restricted to first digit 7-9 and exactly 10 digits to reduce false positives.
5. **Modern ADAMS accessions** — `ML` + exactly 9 alphanumeric chars. Empirically validated against all 33 ML-prefixed samples in `data/adams_samples/`.
6. **`extract_all_citations` applies text normalization** — callers don't need to pre-normalize.
7. **Dedup by canonical key** — keeps highest confidence span when the same reference appears multiple times.
8. **CFR part weak-match overlap prevention** — uses a `covered` offset set (not lookbehinds, which Python `re` doesn't support at variable width) to avoid double-matching "Part 50" inside "10 CFR Part 50".
