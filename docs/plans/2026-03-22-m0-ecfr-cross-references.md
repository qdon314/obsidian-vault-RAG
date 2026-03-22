# M0: ecfr_parser Cross-Reference Tag Support — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend `ecfr_parser.py` to expose cross-reference data (XREF amendment metadata + textual § citations + incorporated standards) so the benchmark pipeline can detect cross-references structurally.

**Architecture:** Add a `CrossRef` frozen dataclass and a `SectionAmendment` frozen dataclass. Extend `ParsedParagraph` with a `cross_references` field for textual citations found in paragraph text. Extend `ParsedSection` with an `amendments` field for section-level XREF tags. Reuse existing regex from `cross_references.py` for textual detection. Add incorporated standard detection (IEEE, ASME, ASTM, ANS, ANSI).

**Tech Stack:** Python stdlib `xml.etree.ElementTree`, `re`, `dataclasses`. No new dependencies.

**GitHub Issue:** #3

---

## Design Decision: XREF vs Textual Cross-References

The design doc assumed XREF/AREF XML tags would carry section-level cross-references. Audit of real data (`data/ecfr/title-10-part-50.xml`, `title-10-part-72.xml`) reveals:

- **XREF tags** are amendment links to the Federal Register (e.g., "Link to an amendment published at 89 FR 106251"), not section cross-references. Only 1 per part.
- **AREF tags**: zero instances in the corpus.
- **Textual cross-references** (`§ 50.55a`, `10 CFR 50.46`) live in paragraph text — already detected by `cross_references.py`.
- **Incorporated standards** (ASME, IEEE, etc.) appear in ~196 paragraphs and are not currently detected.

Therefore M0 delivers three things:
1. `SectionAmendment` — captures XREF amendment metadata at section level
2. `CrossRef` — captures textual § citations per paragraph (reusing existing regex)
3. Incorporated standard detection via new regex

---

### Task 1: Define `CrossRef` and `SectionAmendment` Dataclasses

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/ecfr_parser.py:45-66`
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

Add to `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`:

```python
from rag.adapters.ingestion.regulatory.ecfr_parser import CrossRef, SectionAmendment


def test_cross_ref_dataclass_is_frozen() -> None:
    ref = CrossRef(target_citation="10 CFR §50.55a", kind="cfr")
    assert ref.target_citation == "10 CFR §50.55a"
    assert ref.kind == "cfr"


def test_section_amendment_dataclass_is_frozen() -> None:
    amend = SectionAmendment(
        amendment_id="20241230",
        ref_id="14",
        text="Link to an amendment published at 89 FR 106251, Dec. 30, 2024.",
    )
    assert amend.amendment_id == "20241230"
    assert amend.ref_id == "14"
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_cross_ref_dataclass_is_frozen -v`
Expected: FAIL with `ImportError: cannot import name 'CrossRef'`

**Step 3: Write minimal implementation**

In `ecfr_parser.py`, after line 15 (`from dataclasses import dataclass`), before the `ParsedParagraph` class, add:

```python
@dataclass(frozen=True, slots=True)
class CrossRef:
    """A cross-reference found in paragraph text."""

    target_citation: str  # canonical form, e.g. "10 CFR §50.55a" or "ASME BPV III"
    kind: str  # "cfr" | "incorporated_standard"


@dataclass(frozen=True, slots=True)
class SectionAmendment:
    """Amendment metadata from an XREF element at section level."""

    amendment_id: str  # XREF ID attribute (date-like, e.g. "20241230")
    ref_id: str  # XREF REFID attribute
    text: str  # human-readable amendment description
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_cross_ref_dataclass_is_frozen tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_section_amendment_dataclass_is_frozen -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/ecfr_parser.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): add CrossRef and SectionAmendment dataclasses (M0)"
```

---

### Task 2: Add `cross_references` Field to `ParsedParagraph`

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/ecfr_parser.py:48-55`
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

```python
from rag.adapters.ingestion.regulatory.ecfr_parser import ParsedParagraph


def test_parsed_paragraph_has_cross_references_field() -> None:
    p = ParsedParagraph(text="See § 50.55a.", level=0, prefix=None)
    assert p.cross_references == ()  # default empty tuple
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_parsed_paragraph_has_cross_references_field -v`
Expected: FAIL with `TypeError: ...unexpected keyword argument` or missing field

**Step 3: Write minimal implementation**

Modify `ParsedParagraph` in `ecfr_parser.py`:

```python
@dataclass(frozen=True, slots=True)
class ParsedParagraph:
    """One paragraph extracted from an eCFR section."""

    text: str  # full paragraph text with whitespace normalized
    level: int  # nesting depth (0 = no subsection prefix)
    prefix: str | None  # the raw prefix value, e.g. "a", "1", "iv"
    subsection_tokens: tuple[str, ...] = ()  # full leading chain, e.g. ("a", "1", "i")
    cross_references: tuple[CrossRef, ...] = ()  # cross-refs detected in text
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_parsed_paragraph_has_cross_references_field -v`
Expected: PASS

**Step 5: Run all existing tests to confirm no regression**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py -v`
Expected: All PASS (default `()` is backward-compatible)

**Step 6: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/ecfr_parser.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): add cross_references field to ParsedParagraph (M0)"
```

---

### Task 3: Add `amendments` Field to `ParsedSection`

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/ecfr_parser.py:58-65`
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

```python
from rag.adapters.ingestion.regulatory.ecfr_parser import ParsedSection


def test_parsed_section_has_amendments_field() -> None:
    s = ParsedSection(section_number="50.71", title="Maintenance of records", part_number="50")
    assert s.amendments == ()  # default empty tuple
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_parsed_section_has_amendments_field -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Modify `ParsedSection` in `ecfr_parser.py`:

```python
@dataclass(frozen=True, slots=True)
class ParsedSection:
    """One CFR section (e.g. § 50.36) with its paragraphs."""

    section_number: str  # e.g. "50.36"
    title: str  # human-readable title
    part_number: str  # e.g. "50"
    paragraphs: tuple[ParsedParagraph, ...] = ()
    amendments: tuple[SectionAmendment, ...] = ()  # XREF elements at section level
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/ecfr_parser.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): add amendments field to ParsedSection (M0)"
```

---

### Task 4: Extract XREF Tags into `SectionAmendment`

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/ecfr_parser.py:122-182` (the `parse_ecfr_xml` function)
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

```python
XREF_FIXTURE_XML = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.71" TYPE="SECTION">
      <HEAD>§ 50.71 Maintenance of records, making of reports.</HEAD>
      <XREF ID="20241230" REFID="14" AMDINSN="15">Link to an amendment published at 89 FR 106251, Dec. 30, 2024.</XREF>
      <P>(a) First paragraph.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""


def test_parse_ecfr_xml_extracts_xref_amendments() -> None:
    sections = parse_ecfr_xml(XREF_FIXTURE_XML)
    assert len(sections) == 1
    assert len(sections[0].amendments) == 1
    amend = sections[0].amendments[0]
    assert amend.amendment_id == "20241230"
    assert amend.ref_id == "14"
    assert "89 FR 106251" in amend.text
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_parse_ecfr_xml_extracts_xref_amendments -v`
Expected: FAIL — `sections[0].amendments` is empty `()`

**Step 3: Write minimal implementation**

In `parse_ecfr_xml()`, inside the section loop (after finding `head_elem` and before the paragraph loop), add XREF extraction:

```python
            # Extract XREF amendment metadata.
            amendments: list[SectionAmendment] = []
            for child in section_elem:
                if _local_name(child.tag) == "XREF":
                    amendments.append(
                        SectionAmendment(
                            amendment_id=child.get("ID", ""),
                            ref_id=child.get("REFID", ""),
                            text=_extract_text(child),
                        )
                    )
```

And update the `ParsedSection` construction at the end of the section loop:

```python
            sections.append(
                ParsedSection(
                    section_number=section_number,
                    title=title,
                    part_number=part_number,
                    paragraphs=tuple(paragraphs),
                    amendments=tuple(amendments),
                )
            )
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/ecfr_parser.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): extract XREF amendment metadata from sections (M0)"
```

---

### Task 5: Detect Textual CFR Cross-References in Paragraphs

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/ecfr_parser.py:89-103` (near `_extract_text`) and the paragraph loop
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

```python
def test_parse_ecfr_xml_detects_cfr_cross_references() -> None:
    fixture = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.55a" TYPE="SECTION">
      <HEAD>§ 50.55a Codes and standards.</HEAD>
      <P>(a) Licensees must comply with § 50.46 and 10 CFR 50.34.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""
    sections = parse_ecfr_xml(fixture)
    para = sections[0].paragraphs[0]
    citations = {ref.target_citation for ref in para.cross_references}
    assert "10 CFR §50.46" in citations
    assert "10 CFR §50.34" in citations
    assert all(ref.kind == "cfr" for ref in para.cross_references)
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_parse_ecfr_xml_detects_cfr_cross_references -v`
Expected: FAIL — `cross_references` is empty `()`

**Step 3: Write minimal implementation**

Add an import at the top of `ecfr_parser.py`:

```python
from rag.adapters.ingestion.regulatory.cross_references import extract_cross_references
```

Add a helper function after `_extract_text`:

```python
def _detect_cross_references(text: str) -> tuple[CrossRef, ...]:
    """Detect CFR cross-references in paragraph text."""
    cfr_refs = extract_cross_references(text)
    return tuple(CrossRef(target_citation=ref, kind="cfr") for ref in cfr_refs)
```

In the paragraph loop inside `parse_ecfr_xml()`, update the `ParsedParagraph` construction:

```python
                cross_refs = _detect_cross_references(text)
                paragraphs.append(
                    ParsedParagraph(
                        text=text,
                        level=level,
                        prefix=prefix,
                        subsection_tokens=subsection_tokens,
                        cross_references=cross_refs,
                    )
                )
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/ecfr_parser.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): detect textual CFR cross-references in paragraphs (M0)"
```

---

### Task 6: Detect Incorporated Standards (IEEE, ASME, etc.)

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/ecfr_parser.py` (extend `_detect_cross_references`)
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

```python
def test_parse_ecfr_xml_detects_incorporated_standards() -> None:
    fixture = """\
<?xml version="1.0" encoding="UTF-8"?>
<CFRGRANULE>
  <DIV5 N="Part 50" TYPE="PART">
    <DIV8 N="50.55a" TYPE="SECTION">
      <HEAD>§ 50.55a Codes and standards.</HEAD>
      <P>(a) Systems must meet ASME Boiler and Pressure Vessel Code, Section III requirements and IEEE 323-1974 qualification standards.</P>
    </DIV8>
  </DIV5>
</CFRGRANULE>
"""
    sections = parse_ecfr_xml(fixture)
    para = sections[0].paragraphs[0]
    std_refs = [ref for ref in para.cross_references if ref.kind == "incorporated_standard"]
    std_citations = {ref.target_citation for ref in std_refs}
    assert "ASME Boiler and Pressure Vessel Code, Section III" in std_citations or any(
        "ASME" in c for c in std_citations
    )
    assert any("IEEE 323-1974" in c or "IEEE" in c for c in std_citations)
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_parse_ecfr_xml_detects_incorporated_standards -v`
Expected: FAIL — only CFR refs detected, no incorporated standards

**Step 3: Write minimal implementation**

Add a regex and extend `_detect_cross_references` in `ecfr_parser.py`:

```python
# Matches incorporated standard references: "ASME BPV Code", "IEEE 323-1974", etc.
_INCORPORATED_STANDARD_RE = re.compile(
    r"(?P<body>"
    r"ASME\s+(?:Boiler and Pressure Vessel Code(?:,\s*Section\s+[IVX]+)?|BPV\s+[IVX]+|[A-Z]+\s*[\d./-]+)"
    r"|IEEE\s+[\d.]+-?[\d]*"
    r"|ASTM\s+[A-Z]+[\d.]+-?[\d]*"
    r"|ANS[I]?\s+[\d./]+-?[\d]*"
    r")"
)


def _detect_incorporated_standards(text: str) -> tuple[CrossRef, ...]:
    """Detect incorporated standard references in paragraph text."""
    seen: set[str] = set()
    refs: list[CrossRef] = []
    for match in _INCORPORATED_STANDARD_RE.finditer(text):
        citation = match.group("body").strip()
        if citation not in seen:
            seen.add(citation)
            refs.append(CrossRef(target_citation=citation, kind="incorporated_standard"))
    return tuple(refs)
```

Update `_detect_cross_references`:

```python
def _detect_cross_references(text: str) -> tuple[CrossRef, ...]:
    """Detect all cross-references in paragraph text."""
    cfr_refs = extract_cross_references(text)
    cfr = tuple(CrossRef(target_citation=ref, kind="cfr") for ref in cfr_refs)
    standards = _detect_incorporated_standards(text)
    return cfr + standards
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/ecfr_parser.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): detect incorporated standard references (M0)"
```

---

### Task 7: Export New Types from `__init__.py`

**Files:**
- Modify: `src/rag/adapters/ingestion/regulatory/__init__.py`
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the failing test**

```python
def test_public_api_exports_new_types() -> None:
    from rag.adapters.ingestion.regulatory import CrossRef, SectionAmendment

    assert CrossRef is not None
    assert SectionAmendment is not None
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_public_api_exports_new_types -v`
Expected: FAIL with `ImportError`

**Step 3: Write minimal implementation**

In `__init__.py`, add to the ecfr_parser import:

```python
from rag.adapters.ingestion.regulatory.ecfr_parser import (
    CrossRef,
    ParsedParagraph,
    ParsedSection,
    SectionAmendment,
    parse_ecfr_xml,
)
```

Add `"CrossRef"` and `"SectionAmendment"` to `__all__`.

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py -v`
Expected: All PASS

**Step 5: Commit**

```bash
git add src/rag/adapters/ingestion/regulatory/__init__.py tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "feat(ecfr_parser): export CrossRef and SectionAmendment from public API (M0)"
```

---

### Task 8: Integration Test Against Real XML

**Files:**
- Test: `tests/adapters/ingestion/regulatory/test_ecfr_parser.py`

**Step 1: Write the test**

```python
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not Path("data/ecfr/title-10-part-50.xml").exists(),
    reason="Real eCFR XML not available",
)
def test_real_xml_cross_references_detected() -> None:
    """Smoke test: real XML produces cross-references on at least some paragraphs."""
    xml_text = Path("data/ecfr/title-10-part-50.xml").read_text(encoding="utf-8")
    sections = parse_ecfr_xml(xml_text)
    assert len(sections) > 50  # part 50 has many sections

    # At least some paragraphs should have CFR cross-references
    cfr_ref_count = sum(
        1
        for section in sections
        for para in section.paragraphs
        if any(ref.kind == "cfr" for ref in para.cross_references)
    )
    assert cfr_ref_count > 0, "Expected CFR cross-references in real XML"

    # At least some paragraphs should have incorporated standards
    std_ref_count = sum(
        1
        for section in sections
        for para in section.paragraphs
        if any(ref.kind == "incorporated_standard" for ref in para.cross_references)
    )
    assert std_ref_count > 0, "Expected incorporated standard references in real XML"

    # § 50.71 should have an amendment (XREF tag)
    sec_50_71 = [s for s in sections if s.section_number == "50.71"]
    if sec_50_71:
        assert len(sec_50_71[0].amendments) >= 1
```

**Step 2: Run it**

Run: `./scripts/py -m pytest tests/adapters/ingestion/regulatory/test_ecfr_parser.py::test_real_xml_cross_references_detected -v`
Expected: PASS

**Step 3: Commit**

```bash
git add tests/adapters/ingestion/regulatory/test_ecfr_parser.py
git commit -m "test(ecfr_parser): add integration test for cross-references against real XML (M0)"
```

---

### Task 9: Lint, Typecheck, Full Test Suite

**Files:** None (validation only)

**Step 1: Run formatter**

Run: `make fmt`

**Step 2: Run linter**

Run: `make lint`

**Step 3: Run type checker**

Run: `./scripts/py -m mypy src/rag/adapters/ingestion/regulatory/ecfr_parser.py --config-file pyproject.toml`

Fix any issues that arise. Common: may need to type-narrow `child.get()` returns.

**Step 4: Run full test suite**

Run: `make test`

**Step 5: Commit any fixes**

```bash
git add -u
git commit -m "chore: fix lint/type issues from M0 cross-reference support"
```

---

### Task 10: Update M0 Issue Status

**Step 1: Close the issue**

```bash
gh pm move 3 --status "done" --repo qdon314/obsidian-vault-RAG
```
