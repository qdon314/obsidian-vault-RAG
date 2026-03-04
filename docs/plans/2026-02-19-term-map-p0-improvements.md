# Term Map P0 Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Introduce term confidence metadata (anchor vs contextual), add appendix-level regulatory anchors, and expand 50.59 coverage in the term map.

**Architecture:** The JSON schema changes from a flat `{term: [citations]}` dict to a list of structured objects `[{term, primary, confidence}]`. A new `TermEntry` frozen dataclass holds per-term metadata. `TermMapper` internals update to `dict[str, TermEntry]`, with `TermMatch` gaining a `confidence` field. `CaseQueryGenerator` tags strategy-2 queries with confidence level. P0-2 and P0-3 are purely additive data entries in the new format.

**Tech Stack:** Python 3.12, dataclasses, pytest, ruff

---

## Task 1: Add `TermEntry` dataclass and update `TermMatch`

**Files:**
- Modify: `src/rag/adapters/query_generation/term_mapper.py:1-18`

**Step 1: Add `TermEntry` dataclass and `confidence` field to `TermMatch`**

In `term_mapper.py`, add `TermEntry` above `TermMatch`, and add `confidence` to `TermMatch`:

```python
from enum import StrEnum


class Confidence(StrEnum):
    """How tightly a term binds to its CFR section."""

    ANCHOR = "anchor"
    CONTEXTUAL = "contextual"


@dataclass(frozen=True, slots=True)
class TermEntry:
    """A single entry in the term-to-regulation dictionary."""

    term: str
    primary: list[str]
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class TermMatch:
    """A dictionary term found in case content."""

    term: str
    citations: list[str]
    frequency: int
    confidence: Confidence
```

**Step 2: Run lint**

Run: `./scripts/py -m ruff check src/rag/adapters/query_generation/term_mapper.py`
Expected: PASS (or import-sorting fix)

**Step 3: Commit**

```
feat(query-gen): add TermEntry dataclass and confidence to TermMatch
```

---

## Task 2: Update `TermMapper._terms` type and `from_json` validator

**Files:**
- Modify: `src/rag/adapters/query_generation/term_mapper.py:20-48`

**Step 1: Write the failing tests for the new JSON format**

In `tests/adapters/query_generation/test_term_mapper.py`, replace `TestTermMapperFromJson` with tests that expect the new list-of-objects format:

```python
class TestTermMapperFromJson:
    def test_loads_valid_dictionary(self):
        data = [
            {"term": "ECCS", "primary": ["10 CFR 50.46", "10 CFR 50.34"], "confidence": "anchor"},
            {"term": "technical specification", "primary": ["10 CFR 50.36"], "confidence": "anchor"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        assert mapper.lookup("ECCS") == ["10 CFR 50.46", "10 CFR 50.34"]

    def test_rejects_empty_citation_list(self):
        data = [{"term": "bad", "primary": [], "confidence": "anchor"}]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)

    def test_rejects_non_string_citation(self):
        data = [{"term": "bad", "primary": [123], "confidence": "anchor"}]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)

    def test_rejects_invalid_confidence(self):
        data = [{"term": "bad", "primary": ["10 CFR 50.46"], "confidence": "high"}]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)

    def test_rejects_missing_required_field(self):
        data = [{"term": "bad", "primary": ["10 CFR 50.46"]}]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)

    def test_rejects_duplicate_terms(self):
        data = [
            {"term": "ECCS", "primary": ["10 CFR 50.46"], "confidence": "anchor"},
            {"term": "ECCS", "primary": ["10 CFR 50.34"], "confidence": "contextual"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            with pytest.raises(ValueError):
                TermMapper.from_json(p)
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py::TestTermMapperFromJson -v`
Expected: FAIL — `from_json` still expects old dict format

**Step 3: Update `TermMapper._terms` and `from_json`**

Replace `_terms` type and rewrite `from_json`:

```python
@dataclass(frozen=True, slots=True)
class TermMapper:
    """Loads a term-to-regulation dictionary and matches terms in text."""

    _terms: dict[str, TermEntry]

    @classmethod
    def from_json(cls, path: Path) -> TermMapper:
        """Load and validate a term dictionary from a JSON file.

        Expected format: [{"term": "...", "primary": ["10 CFR XX.YY", ...], "confidence": "anchor"|"contextual"}, ...]
        Raises ValueError on malformed entries.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            msg = f"Expected JSON array, got {type(raw).__name__}"
            raise ValueError(msg)

        terms: dict[str, TermEntry] = {}

        for entry in raw:
            if not isinstance(entry, dict):
                msg = f"Each entry must be an object, got {type(entry).__name__}"
                raise ValueError(msg)
            for field in ("term", "primary", "confidence"):
                if field not in entry:
                    msg = f"Entry missing required field '{field}': {entry}"
                    raise ValueError(msg)

            term = entry["term"]
            primary = entry["primary"]
            raw_confidence = entry["confidence"]

            if not isinstance(term, str):
                msg = f"Term must be str, got {type(term).__name__}"
                raise ValueError(msg)
            if not isinstance(primary, list) or len(primary) == 0:
                msg = f"Term '{term}' must map to a non-empty list"
                raise ValueError(msg)
            for c in primary:
                if not isinstance(c, str):
                    msg = f"Citation for '{term}' must be str, got {type(c).__name__}"
                    raise ValueError(msg)
            try:
                confidence = Confidence(raw_confidence)
            except ValueError:
                msg = f"Confidence for '{term}' must be one of {list(Confidence)}, got '{raw_confidence}'"
                raise ValueError(msg) from None

            key = term.lower()
            if key in terms:
                msg = f"Duplicate term (case-insensitive): '{term}'"
                raise ValueError(msg)

            terms[key] = TermEntry(term=term, primary=primary, confidence=confidence)

        return cls(_terms=terms)
```

**Step 4: Run `from_json` tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py::TestTermMapperFromJson -v`
Expected: PASS

**Step 5: Commit**

```
feat(query-gen): update TermMapper to accept structured JSON format
```

---

## Task 3: Update `lookup` and `scan_content`

**Files:**
- Modify: `src/rag/adapters/query_generation/term_mapper.py:50-74`
- Modify: `tests/adapters/query_generation/test_term_mapper.py`

**Step 1: Update test helpers to use new JSON format**

In `test_term_mapper.py`, update all `_make_mapper` helpers and inline data dicts to use the list-of-objects format. For example in `TestTermMapperLookup`:

```python
class TestTermMapperLookup:
    def _make_mapper(self) -> TermMapper:
        data = [
            {"term": "ECCS", "primary": ["10 CFR 50.46"], "confidence": "anchor"},
            {"term": "surveillance testing", "primary": ["10 CFR 50.36"], "confidence": "anchor"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)
```

And in `TestTermMapperScanContent`:

```python
class TestTermMapperScanContent:
    def _make_mapper(self) -> TermMapper:
        data = [
            {"term": "ECCS", "primary": ["10 CFR 50.46"], "confidence": "anchor"},
            {"term": "surveillance testing", "primary": ["10 CFR 50.36"], "confidence": "anchor"},
            {"term": "LCO", "primary": ["10 CFR 50.36"], "confidence": "anchor"},
            {"term": "peak cladding temperature", "primary": ["10 CFR 50.46"], "confidence": "anchor"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)
```

Also update inline data in `test_word_boundary_prevents_false_positives` and `test_word_boundary_matches_real_acronyms`:

```python
    def test_word_boundary_prevents_false_positives(self):
        data = [
            {"term": "IST", "primary": ["10 CFR 50.55a"], "confidence": "anchor"},
            {"term": "CAP", "primary": ["10 CFR 50.34"], "confidence": "contextual"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The existing list of capacity items was consistent."
        matches = mapper.scan_content(content)
        assert matches == []

    def test_word_boundary_matches_real_acronyms(self):
        data = [
            {"term": "IST", "primary": ["10 CFR 50.55a"], "confidence": "anchor"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = "The IST program was reviewed. IST results were satisfactory."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].term == "IST"
        assert matches[0].frequency == 2
        assert matches[0].confidence == Confidence.ANCHOR
```

Add `from rag.adapters.query_generation.term_mapper import Confidence` to the test file imports.

Add a new test for confidence propagation in scan results:

```python
    def test_scan_returns_confidence_from_entry(self):
        data = [
            {"term": "ECCS", "primary": ["10 CFR 50.46"], "confidence": "anchor"},
            {"term": "risk-informed", "primary": ["10 CFR 50.65"], "confidence": "contextual"},
        ]
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        content = (
            "The ECCS was tested. ECCS passed. "
            "A risk-informed approach was used. The risk-informed method worked."
        )
        matches = mapper.scan_content(content)
        assert len(matches) == 2
        by_term = {m.term: m for m in matches}
        assert by_term["ECCS"].confidence == Confidence.ANCHOR
        assert by_term["risk-informed"].confidence == Confidence.CONTEXTUAL
```

**Step 2: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py -v`
Expected: FAIL — `lookup` and `scan_content` reference old `_terms` structure

**Step 3: Update `lookup` and `scan_content`**

```python
    def lookup(self, term: str) -> list[str]:
        """Return citations for a term (case-insensitive), or empty list."""
        entry = self._terms.get(term.lower())
        if entry is not None:
            return list(entry.primary)
        return []

    def scan_content(self, content: str) -> list[TermMatch]:
        """Find all dictionary terms in content.

        Returns matches with frequency >= 2, sorted by descending frequency.
        Matching is case-insensitive with word boundaries.
        """
        content_lower = content.lower()
        matches: list[TermMatch] = []
        for entry in self._terms.values():
            pattern = re.compile(r"\b" + re.escape(entry.term.lower()) + r"\b")
            count = len(pattern.findall(content_lower))
            if count >= 2:
                matches.append(
                    TermMatch(
                        term=entry.term,
                        citations=list(entry.primary),
                        frequency=count,
                        confidence=entry.confidence,
                    )
                )
        matches.sort(key=lambda m: (-m.frequency, m.term))
        return matches
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py -v`
Expected: ALL PASS

**Step 5: Commit**

```
feat(query-gen): update lookup and scan_content for structured term entries
```

---

## Task 4: Update `CaseQueryGenerator` tests and implementation for confidence tagging

**Files:**
- Modify: `tests/adapters/query_generation/test_case_query_generator.py`
- Modify: `src/rag/adapters/query_generation/case_query_generator.py`

**Step 1: Update test helpers to use new JSON format**

In `test_case_query_generator.py`, update `_make_term_mapper`:

```python
def _make_term_mapper(terms: list[dict[str, Any]] | None = None) -> TermMapper:
    if terms is None:
        terms = [
            {"term": "ECCS", "primary": ["10 CFR 50.46"], "confidence": "anchor"},
            {"term": "surveillance testing", "primary": ["10 CFR 50.36"], "confidence": "anchor"},
            {"term": "LCO", "primary": ["10 CFR 50.36"], "confidence": "anchor"},
            {"term": "accumulator", "primary": ["10 CFR 50.46", "10 CFR 50.36"], "confidence": "anchor"},
        ]
    with TemporaryDirectory() as td:
        p = Path(td) / "terms.json"
        p.write_text(json.dumps(terms))
        return TermMapper.from_json(p)
```

Add `from typing import Any` import. Update `test_max_5_term_mapping_queries` to use new format:

```python
    def test_max_5_term_mapping_queries(self):
        many_terms = [
            {"term": f"term{i}", "primary": [f"10 CFR 50.{i}"], "confidence": "anchor"}
            for i in range(10)
        ]
        # ... rest stays the same
        mapper = _make_term_mapper(many_terms)
```

**Step 2: Add test for confidence tag on strategy 2 queries**

```python
class TestStrategy2TermMapping:
    # ... existing tests ...

    def test_anchor_term_tagged(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        eccs_queries = [q for q in s2 if q["technical_term"] == "ECCS"]
        assert len(eccs_queries) >= 1
        assert "anchor-term" in eccs_queries[0]["tags"]

    def test_contextual_term_tagged(self):
        terms = [
            {"term": "risk-informed", "primary": ["10 CFR 50.65"], "confidence": "contextual"},
        ]
        body_lines = ["A risk-informed approach was used. " * 3]
        md = (
            "---\n"
            'accession_number: "ML99999A001"\n'
            'document_type: "Report"\n'
            "cross_references: []\n"
            "---\n\n" + "\n".join(body_lines)
        )
        mapper = _make_term_mapper(terms)
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td, content=md)
            queries = gen.generate(case_file)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        assert len(s2) >= 1
        assert "contextual-term" in s2[0]["tags"]
        assert "anchor-term" not in s2[0]["tags"]
```

**Step 3: Run tests to verify they fail**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`
Expected: FAIL — confidence tag not added yet

**Step 4: Update `CaseQueryGenerator.generate` strategy 2 to add confidence tag**

In the strategy 2 loop in `case_query_generator.py`, after `tags = ["case-derived", "term-mapping"]`, add:

```python
            tags.append(f"{match.confidence}-term")
```

This adds either `"anchor-term"` or `"contextual-term"` to every strategy 2 query.

**Step 5: Run all query generation tests**

Run: `./scripts/py -m pytest tests/adapters/query_generation/ -v`
Expected: ALL PASS

**Step 6: Commit**

```
feat(query-gen): tag strategy-2 queries with confidence level
```

---

## Task 5: Migrate `case_regulatory_terms.json` to new format and add P0-2 + P0-3 entries

**Files:**
- Modify: `config/case_regulatory_terms.json`

**Step 1: Convert existing entries to structured format and add new entries**

Rewrite `config/case_regulatory_terms.json` as a JSON array. Each existing entry gets `"confidence": "anchor"` or `"confidence": "contextual"` based on how tightly the term binds to its CFR section.

Classification guidance:
- **anchor**: Term is defined by or uniquely associated with a specific CFR section (e.g., "ECCS" → 50.46, "SBO" → 50.63, "LER" → 50.73)
- **contextual**: Term appears across many regulatory contexts or has a weaker/broader association (e.g., "risk-informed", "operability", "corrective action program", "safety-related")

New entries to add for **P0-2** (appendix-level anchors):

```json
{"term": "general design criteria", "primary": ["10 CFR 50 Appendix A"], "confidence": "anchor"},
{"term": "GDC", "primary": ["10 CFR 50 Appendix A"], "confidence": "anchor"},
{"term": "single failure criterion", "primary": ["10 CFR 50 Appendix A"], "confidence": "anchor"},
{"term": "quality assurance criteria", "primary": ["10 CFR 50 Appendix B"], "confidence": "anchor"},
{"term": "Appendix B", "primary": ["10 CFR 50 Appendix B"], "confidence": "anchor"},
{"term": "ECCS evaluation model", "primary": ["10 CFR 50 Appendix K"], "confidence": "anchor"},
{"term": "Appendix K", "primary": ["10 CFR 50 Appendix K"], "confidence": "anchor"},
{"term": "fire protection program", "primary": ["10 CFR 50 Appendix R"], "confidence": "anchor"},
{"term": "Appendix R", "primary": ["10 CFR 50 Appendix R"], "confidence": "anchor"},
{"term": "safe shutdown", "primary": ["10 CFR 50 Appendix R"], "confidence": "anchor"},
{"term": "emergency planning zone", "primary": ["10 CFR 50 Appendix E"], "confidence": "anchor"},
{"term": "EPZ", "primary": ["10 CFR 50 Appendix E"], "confidence": "anchor"},
{"term": "containment leakage", "primary": ["10 CFR 50 Appendix J"], "confidence": "anchor"},
{"term": "ILRT", "primary": ["10 CFR 50 Appendix J"], "confidence": "anchor"},
{"term": "integrated leak rate test", "primary": ["10 CFR 50 Appendix J"], "confidence": "anchor"},
{"term": "local leak rate test", "primary": ["10 CFR 50 Appendix J"], "confidence": "anchor"},
{"term": "LLRT", "primary": ["10 CFR 50 Appendix J"], "confidence": "anchor"}
```

New entries to add for **P0-3** (50.59 granularity):

```json
{"term": "50.59 screening", "primary": ["10 CFR 50.59"], "confidence": "anchor"},
{"term": "50.59 evaluation", "primary": ["10 CFR 50.59"], "confidence": "anchor"},
{"term": "design change", "primary": ["10 CFR 50.59"], "confidence": "contextual"},
{"term": "prior NRC approval", "primary": ["10 CFR 50.59"], "confidence": "anchor"}
```

Also reclassify existing entries. Terms that should be `"contextual"`:
- `"risk-informed"` — appears in many regulatory contexts beyond 50.65
- `"operability"` / `"operable"` — broadly used across tech specs, not uniquely 50.36
- `"safety-related"` — definitional term used everywhere, not uniquely 50.2
- `"corrective action program"` — spans QA, enforcement, and licensing
- `"nonconformance"` — same reasoning as CAP
- `"quality assurance"` / `"QA program"` — broad program concept, not uniquely 50.34
- `"design basis accident"` / `"design basis event"` — definitional, used broadly
- `"design change"` — spans 50.59, 50.90, and general licensing
- `"preventive maintenance"` / `"corrective maintenance"` — general terms, not uniquely 50.65
- `"administrative control"` — broad tech spec concept
- `"design feature"` — broad tech spec concept

All other terms default to `"anchor"`.

**Step 2: Validate JSON is well-formed**

Run: `./scripts/py -c "import json; json.load(open('config/case_regulatory_terms.json')); print('OK')"`
Expected: `OK`

**Step 3: Run the full `from_json` path against the real file**

Run: `./scripts/py -c "from pathlib import Path; from rag.adapters.query_generation.term_mapper import TermMapper; m = TermMapper.from_json(Path('config/case_regulatory_terms.json')); print(f'{len(m._terms)} terms loaded')"`
Expected: prints count (should be ~109 with new entries, no duplicates error)

**Step 4: Run all tests**

Run: `./scripts/py -m pytest tests/adapters/query_generation/ -v`
Expected: ALL PASS

**Step 5: Run lint and typecheck**

Run: `./scripts/py -m ruff check src/rag/adapters/query_generation/ && ./scripts/py -m mypy src/rag`
Expected: PASS

**Step 6: Commit**

```
feat(query-gen): migrate term map to structured format, add appendix anchors and 50.59 terms
```

---

## Summary of all changes

| File | Change |
|------|--------|
| `src/rag/adapters/query_generation/term_mapper.py` | Add `Confidence` StrEnum, `TermEntry` dataclass; add `confidence` to `TermMatch`; rewrite `from_json` for list-of-objects JSON; update `lookup`/`scan_content` for `TermEntry` dict |
| `src/rag/adapters/query_generation/case_query_generator.py` | Add `anchor-term`/`contextual-term` tag to strategy 2 queries |
| `config/case_regulatory_terms.json` | Convert to list-of-objects format; classify each term; add ~17 appendix entries; add 4 entries for 50.59 |
| `tests/adapters/query_generation/test_term_mapper.py` | Update all test data to new format; add validation tests for confidence, duplicates; add confidence propagation test |
| `tests/adapters/query_generation/test_case_query_generator.py` | Update helper to new format; add confidence tagging tests |
