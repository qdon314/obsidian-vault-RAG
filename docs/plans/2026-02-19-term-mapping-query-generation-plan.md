# Term Mapping & Query Generation (Strategies 1-2) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a term-to-regulation mapping dictionary and a query generator that produces eval JSONL from case documents using direct citation (Strategy 1) and term mapping (Strategy 2) strategies.

**Architecture:** A `TermMapper` frozen dataclass loads a JSON dictionary and scans case text for matching terms. A `CaseQueryGenerator` frozen dataclass consumes a `TermMapper` plus case markdown files, parses frontmatter via `split_obsidian_frontmatter`, and emits query dicts. A CLI script glues them together.

**Tech Stack:** Python stdlib only (json, pathlib, argparse). PyYAML via existing `split_obsidian_frontmatter`. No new dependencies.

**Design doc:** `docs/plans/2026-02-19-term-mapping-query-generation-design.md`

**Key conventions:**
- All commands via `./scripts/py` or `make`, never bare `python`/`pytest`
- Frozen dataclasses with `slots=True`
- Empty `__init__.py` files (no re-exports)
- Tests: plain classes, bare `assert`, inline object construction, no fixtures
- Do not commit — provide suggested commits at the end of each task

---

## Task 1: TermMatch and TermMapper — Loading and Lookup

**Files:**
- Create: `src/rag/adapters/query_generation/__init__.py`
- Create: `src/rag/adapters/query_generation/term_mapper.py`
- Create: `tests/adapters/query_generation/__init__.py`
- Create: `tests/adapters/query_generation/test_term_mapper.py`

### Step 1: Create package directories with empty `__init__.py` files

Create two empty `__init__.py` files:
- `src/rag/adapters/query_generation/__init__.py`
- `tests/adapters/query_generation/__init__.py`

### Step 2: Write failing tests for TermMapper loading and lookup

File: `tests/adapters/query_generation/test_term_mapper.py`

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.adapters.query_generation.term_mapper import TermMapper


class TestTermMapperFromJson:
    def test_loads_valid_dictionary(self):
        data = {
            "ECCS": ["10 CFR 50.46", "10 CFR 50.34"],
            "technical specification": ["10 CFR 50.36"],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            mapper = TermMapper.from_json(p)
        assert mapper.lookup("ECCS") == ["10 CFR 50.46", "10 CFR 50.34"]

    def test_rejects_empty_citation_list(self):
        data = {"bad_term": []}
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            try:
                TermMapper.from_json(p)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass

    def test_rejects_non_string_citation(self):
        data = {"bad_term": [123]}
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            try:
                TermMapper.from_json(p)
                assert False, "Should have raised ValueError"
            except ValueError:
                pass


class TestTermMapperLookup:
    def _make_mapper(self) -> TermMapper:
        data = {
            "ECCS": ["10 CFR 50.46"],
            "surveillance testing": ["10 CFR 50.36"],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)

    def test_lookup_exact_match(self):
        mapper = self._make_mapper()
        assert mapper.lookup("ECCS") == ["10 CFR 50.46"]

    def test_lookup_case_insensitive(self):
        mapper = self._make_mapper()
        assert mapper.lookup("eccs") == ["10 CFR 50.46"]

    def test_lookup_miss_returns_empty(self):
        mapper = self._make_mapper()
        assert mapper.lookup("nonexistent") == []
```

### Step 3: Run tests to verify they fail

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'rag.adapters.query_generation.term_mapper'`

### Step 4: Implement TermMatch and TermMapper

File: `src/rag/adapters/query_generation/term_mapper.py`

```python
"""Term-to-regulation mapping for case-derived query generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TermMatch:
    """A dictionary term found in case content."""

    term: str
    citations: list[str]
    frequency: int


@dataclass(frozen=True, slots=True)
class TermMapper:
    """Loads a term→regulation dictionary and matches terms in text."""

    _terms: dict[str, list[str]]

    @classmethod
    def from_json(cls, path: Path) -> TermMapper:
        """Load and validate a term dictionary from a JSON file.

        Expected format: {"term": ["10 CFR XX.YY", ...], ...}
        Raises ValueError on malformed entries.
        """
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            msg = f"Expected JSON object, got {type(raw).__name__}"
            raise ValueError(msg)
        for key, citations in raw.items():
            if not isinstance(key, str):
                msg = f"Term key must be str, got {type(key).__name__}"
                raise ValueError(msg)
            if not isinstance(citations, list) or len(citations) == 0:
                msg = f"Term '{key}' must map to a non-empty list"
                raise ValueError(msg)
            for c in citations:
                if not isinstance(c, str):
                    msg = f"Citation for '{key}' must be str, got {type(c).__name__}"
                    raise ValueError(msg)
        return cls(_terms=raw)

    def lookup(self, term: str) -> list[str]:
        """Return citations for a term (case-insensitive), or empty list."""
        key = term.lower()
        for k, v in self._terms.items():
            if k.lower() == key:
                return v
        return []

    def scan_content(self, content: str) -> list[TermMatch]:
        """Find all dictionary terms in content.

        Returns matches with frequency >= 2, sorted by descending frequency.
        Matching is case-insensitive exact substring.
        """
        content_lower = content.lower()
        matches: list[TermMatch] = []
        for term, citations in self._terms.items():
            term_lower = term.lower()
            count = content_lower.count(term_lower)
            if count >= 2:
                matches.append(TermMatch(term=term, citations=citations, frequency=count))
        matches.sort(key=lambda m: m.frequency, reverse=True)
        return matches
```

### Step 5: Run tests to verify they pass

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py -v`

Expected: All 5 tests PASS.

### Step 6: Run lint and type check

Run: `./scripts/py -m ruff check src/rag/adapters/query_generation/term_mapper.py tests/adapters/query_generation/test_term_mapper.py`

Run: `./scripts/py -m mypy src/rag/adapters/query_generation/term_mapper.py`

Fix any issues.

### Step 7: Commit

Suggested commit:
```
feat(query-gen): add TermMapper with JSON loading and lookup
```
Files: `src/rag/adapters/query_generation/__init__.py`, `src/rag/adapters/query_generation/term_mapper.py`, `tests/adapters/query_generation/__init__.py`, `tests/adapters/query_generation/test_term_mapper.py`

---

## Task 2: TermMapper.scan_content Tests

**Files:**
- Modify: `tests/adapters/query_generation/test_term_mapper.py`

### Step 1: Write failing tests for scan_content

Append to `tests/adapters/query_generation/test_term_mapper.py`:

```python
class TestTermMapperScanContent:
    def _make_mapper(self) -> TermMapper:
        data = {
            "ECCS": ["10 CFR 50.46"],
            "surveillance testing": ["10 CFR 50.36"],
            "LCO": ["10 CFR 50.36"],
            "peak cladding temperature": ["10 CFR 50.46"],
        }
        with TemporaryDirectory() as td:
            p = Path(td) / "terms.json"
            p.write_text(json.dumps(data))
            return TermMapper.from_json(p)

    def test_finds_terms_case_insensitive(self):
        mapper = self._make_mapper()
        content = "The ECCS was tested. The eccs met all criteria. ECCS passed."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].term == "ECCS"
        assert matches[0].frequency == 3

    def test_sorts_by_descending_frequency(self):
        mapper = self._make_mapper()
        content = (
            "LCO 3.5.1 requires ECCS operability. "
            "The LCO was met. LCO was verified. LCO again. "
            "ECCS accumulator levels were checked. ECCS passed."
        )
        matches = mapper.scan_content(content)
        assert len(matches) == 2
        assert matches[0].term == "LCO"
        assert matches[1].term == "ECCS"
        assert matches[0].frequency > matches[1].frequency

    def test_excludes_terms_with_frequency_below_2(self):
        mapper = self._make_mapper()
        content = "The ECCS was tested. The peak cladding temperature was fine."
        matches = mapper.scan_content(content)
        # ECCS appears once, peak cladding temperature appears once — both excluded
        assert len(matches) == 0

    def test_returns_empty_for_no_matches(self):
        mapper = self._make_mapper()
        content = "This document discusses reactor coolant pumps only."
        matches = mapper.scan_content(content)
        assert matches == []

    def test_returns_citations_from_dictionary(self):
        mapper = self._make_mapper()
        content = "Surveillance testing was performed. Surveillance testing passed."
        matches = mapper.scan_content(content)
        assert len(matches) == 1
        assert matches[0].citations == ["10 CFR 50.36"]
```

### Step 2: Run tests to verify they pass

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_term_mapper.py -v`

Expected: All 10 tests PASS (scan_content was already implemented in Task 1).

### Step 3: Commit

Suggested commit:
```
test(query-gen): add scan_content test coverage for TermMapper
```
Files: `tests/adapters/query_generation/test_term_mapper.py`

---

## Task 3: Seed Term Dictionary

**Files:**
- Create: `config/case_regulatory_terms.json`

### Step 1: Create the config directory and seed dictionary

File: `config/case_regulatory_terms.json`

```json
{
    "ECCS": ["10 CFR 50.46", "10 CFR 50.34"],
    "emergency core cooling": ["10 CFR 50.46"],
    "emergency core cooling system": ["10 CFR 50.46"],
    "accumulator": ["10 CFR 50.46", "10 CFR 50.36"],
    "peak cladding temperature": ["10 CFR 50.46"],
    "cladding oxidation": ["10 CFR 50.46"],
    "coolable geometry": ["10 CFR 50.46"],
    "long-term cooling": ["10 CFR 50.46"],
    "LOCA": ["10 CFR 50.46"],
    "loss of coolant accident": ["10 CFR 50.46"],
    "loss-of-coolant accident": ["10 CFR 50.46"],
    "technical specification": ["10 CFR 50.36"],
    "technical specifications": ["10 CFR 50.36"],
    "tech spec": ["10 CFR 50.36"],
    "LCO": ["10 CFR 50.36"],
    "limiting condition for operation": ["10 CFR 50.36"],
    "surveillance requirement": ["10 CFR 50.36"],
    "surveillance requirements": ["10 CFR 50.36"],
    "surveillance testing": ["10 CFR 50.36"],
    "surveillance test": ["10 CFR 50.36"],
    "safety limit": ["10 CFR 50.36"],
    "design feature": ["10 CFR 50.36"],
    "administrative control": ["10 CFR 50.36"],
    "50.59": ["10 CFR 50.59"],
    "unreviewed safety question": ["10 CFR 50.59"],
    "license amendment": ["10 CFR 50.59", "10 CFR 50.90"],
    "license amendment request": ["10 CFR 50.90"],
    "maintenance rule": ["10 CFR 50.65"],
    "preventive maintenance": ["10 CFR 50.65"],
    "corrective maintenance": ["10 CFR 50.65"],
    "maintenance effectiveness": ["10 CFR 50.65"],
    "risk-informed": ["10 CFR 50.65"],
    "Part 21": ["10 CFR 21"],
    "defect reporting": ["10 CFR 21"],
    "substantial safety hazard": ["10 CFR 21"],
    "event notification": ["10 CFR 50.72"],
    "immediate notification": ["10 CFR 50.72"],
    "four-hour notification": ["10 CFR 50.72"],
    "eight-hour notification": ["10 CFR 50.72"],
    "licensee event report": ["10 CFR 50.73"],
    "LER": ["10 CFR 50.73"],
    "station blackout": ["10 CFR 50.63"],
    "SBO": ["10 CFR 50.63"],
    "alternate AC source": ["10 CFR 50.63"],
    "ATWS": ["10 CFR 50.62"],
    "anticipated transient without scram": ["10 CFR 50.62"],
    "fire protection": ["10 CFR 50.48"],
    "combustible gas control": ["10 CFR 50.44"],
    "inerted atmosphere": ["10 CFR 50.44"],
    "pressurized thermal shock": ["10 CFR 50.61"],
    "PTS": ["10 CFR 50.61"],
    "reactor vessel embrittlement": ["10 CFR 50.61"],
    "quality assurance": ["10 CFR 50.34"],
    "QA program": ["10 CFR 50.34"],
    "safety-related": ["10 CFR 50.2"],
    "design basis accident": ["10 CFR 50.2"],
    "design basis event": ["10 CFR 50.2"],
    "emergency plan": ["10 CFR 50.47"],
    "emergency planning": ["10 CFR 50.47"],
    "emergency preparedness": ["10 CFR 50.47"],
    "physical security": ["10 CFR 73.55"],
    "security plan": ["10 CFR 73.55"],
    "fitness for duty": ["10 CFR 26"],
    "access authorization": ["10 CFR 73.56"],
    "radiation protection": ["10 CFR 20"],
    "dose limit": ["10 CFR 20"],
    "ALARA": ["10 CFR 20"],
    "effluent": ["10 CFR 50.36a"],
    "environmental monitoring": ["10 CFR 50.36a"],
    "decommissioning": ["10 CFR 50.82"],
    "decommissioning fund": ["10 CFR 50.75"],
    "inservice inspection": ["10 CFR 50.55a"],
    "ISI": ["10 CFR 50.55a"],
    "inservice testing": ["10 CFR 50.55a"],
    "IST": ["10 CFR 50.55a"],
    "ASME Code": ["10 CFR 50.55a"],
    "reactor coolant pressure boundary": ["10 CFR 50.2", "10 CFR 50.55a"],
    "operability": ["10 CFR 50.36"],
    "operable": ["10 CFR 50.36"],
    "reportable condition": ["10 CFR 50.72", "10 CFR 50.73"],
    "corrective action program": ["10 CFR 50.34"],
    "CAP": ["10 CFR 50.34"],
    "condition report": ["10 CFR 50.34"],
    "nonconformance": ["10 CFR 50.34"],
    "violation": ["10 CFR 2.201"],
    "civil penalty": ["10 CFR 2.205"],
    "severity level": ["10 CFR 2.201"],
    "enforcement action": ["10 CFR 2.201"],
    "confirmatory order": ["10 CFR 2.202"],
    "demand for information": ["10 CFR 2.204"]
}
```

~85 entries covering the major clusters: ECCS/safety analysis, tech specs, change control, maintenance, reporting, station blackout, ATWS, fire protection, combustible gas, PTS, QA, emergency planning, security, radiation protection, ISI/IST, enforcement.

### Step 2: Validate the JSON is well-formed

Run: `./scripts/py -c "import json; json.load(open('config/case_regulatory_terms.json')); print('OK')"`

Expected: `OK`

### Step 3: Validate via TermMapper loader

Run: `./scripts/py -c "from rag.adapters.query_generation.term_mapper import TermMapper; m = TermMapper.from_json(__import__('pathlib').Path('config/case_regulatory_terms.json')); print(f'{len(m._terms)} terms loaded')"`

Expected: `85 terms loaded` (approximately)

### Step 4: Commit

Suggested commit:
```
feat(query-gen): add seed term-to-regulation dictionary (85 entries)
```
Files: `config/case_regulatory_terms.json`

---

## Task 4: CaseQueryGenerator — Frontmatter Parsing and Strategy 1

**Files:**
- Create: `src/rag/adapters/query_generation/case_query_generator.py`
- Create: `tests/adapters/query_generation/test_case_query_generator.py`

### Step 1: Write a test fixture markdown string

We need a synthetic case document for testing. Define it as a module-level constant in the test file so all test classes can use it.

### Step 2: Write failing tests for Strategy 1

File: `tests/adapters/query_generation/test_case_query_generator.py`

```python
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from rag.adapters.query_generation.case_query_generator import CaseQueryGenerator
from rag.adapters.query_generation.term_mapper import TermMapper

FIXTURE_CASE_MD = """\
---
corpus: "case"
regime: "us-nrc"
corpus_label: "nrc-cases"
accession_number: "ML99999A001"
title: "Test Inspection Report"
document_type: "Inspection Report"
document_date: "2024-06-15T00:00:00"
case_category: "inspection"
case_subcategory: "inspection_routine_report"
case_category_confidence: 0.95
facility_name: "Test Nuclear Station"
reactor_type: "PWR"
dockets: ["05000999"]
regulation_parts: []
regulation_sections: ["50.46", "50.36"]
citation_keys: ["cfr:10:50.46", "cfr:10:50.36"]
cross_references: ["10 CFR §50.46", "10 CFR §50.36"]
source_url: "https://example.com"
---

# ML99999A001 — Test Inspection Report

**Facility:** Test Nuclear Station

## Content

The licensee failed to perform ECCS accumulator surveillance testing as
required. The ECCS accumulator level was found below the LCO limit.
Surveillance testing was not completed within the required interval.
The ECCS system was declared inoperable. ECCS operability was restored
after corrective maintenance. Surveillance testing confirmed ECCS
accumulator parameters met technical specification requirements.
The LCO action statement was entered and exited appropriately.
LCO 3.5.1 requires accumulator operability when reactor coolant
system pressure is above 1000 psig.
"""


def _make_term_mapper(terms: dict[str, list[str]] | None = None) -> TermMapper:
    if terms is None:
        terms = {
            "ECCS": ["10 CFR 50.46"],
            "surveillance testing": ["10 CFR 50.36"],
            "LCO": ["10 CFR 50.36"],
            "accumulator": ["10 CFR 50.46", "10 CFR 50.36"],
        }
    with TemporaryDirectory() as td:
        p = Path(td) / "terms.json"
        p.write_text(json.dumps(terms))
        return TermMapper.from_json(p)


def _make_case_file(tmp_dir: str, content: str = FIXTURE_CASE_MD) -> Path:
    p = Path(tmp_dir) / "ML99999A001.md"
    p.write_text(content)
    return p


class TestStrategy1DirectCitation:
    def test_generates_one_query_per_cross_reference(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        # Fixture has 2 cross_references
        assert len(s1) == 2

    def test_query_references_cited_regulation(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        citations_in_queries = {q["relevant_citations"][0] for q in s1}
        assert "10 CFR §50.46" in citations_in_queries
        assert "10 CFR §50.36" in citations_in_queries

    def test_query_has_required_fields(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        q = queries[0]
        required = [
            "qid", "query", "relevant_citations", "relevant_doc_citations",
            "query_type", "difficulty", "requires_synthesis", "tags",
            "source_case", "is_unanswerable", "expected_answer", "metadata",
        ]
        for field in required:
            assert field in q, f"Missing field: {field}"

    def test_strategy1_field_values(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        for q in s1:
            assert q["difficulty"] == "easy"
            assert q["query_type"] == "factual"
            assert q["requires_synthesis"] is False
            assert q["is_unanswerable"] is False
            assert q["expected_answer"] is None
            assert q["source_case"] == "ML99999A001"
            assert q["metadata"] == {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}}

    def test_uses_varied_templates(self):
        """With 2 citations, at least 2 different query phrasings should appear."""
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s1 = [q for q in queries if "citation-direct" in q["tags"]]
        query_texts = [q["query"] for q in s1]
        # Two queries should use different templates (index 0 and 1)
        assert query_texts[0] != query_texts[1]
```

### Step 3: Run tests to verify they fail

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`

Expected: FAIL — `ModuleNotFoundError: No module named 'rag.adapters.query_generation.case_query_generator'`

### Step 4: Implement CaseQueryGenerator with Strategy 1

File: `src/rag/adapters/query_generation/case_query_generator.py`

```python
"""Generate evaluation queries from NRC case documents."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import (
    split_obsidian_frontmatter,
)
from rag.adapters.query_generation.term_mapper import TermMapper

DIRECT_CITATION_TEMPLATES = [
    "What are the requirements of {citation}?",
    "What does {citation} require?",
    "Summarize the key provisions of {citation}.",
]

TERM_MAPPING_TEMPLATES = [
    "What are the regulatory requirements for {term}?",
    "What regulations govern {term} at nuclear power plants?",
    "What does the NRC require regarding {term}?",
]


@dataclass(frozen=True, slots=True)
class CaseQueryGenerator:
    """Generate eval queries from case markdown using citation and term strategies."""

    term_mapper: TermMapper
    max_queries_per_case: int = 20
    _dc_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0],
    )
    _tm_counter: list[int] = field(
        init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0],
    )

    def generate_from_case(self, case_path: Path) -> list[dict[str, Any]]:
        """Parse case markdown, run strategies, return query dicts."""
        text = case_path.read_text(encoding="utf-8")
        fm, content = split_obsidian_frontmatter(text)
        queries: list[dict[str, Any]] = []
        queries.extend(self._direct_citation_queries(fm))
        queries.extend(self._term_mapping_queries(fm, content))
        return queries[: self.max_queries_per_case]

    def _next_qid(self, prefix: str) -> str:
        counter = self._dc_counter if prefix == "dc" else self._tm_counter
        counter[0] += 1
        return f"case-{prefix}-{counter[0]:03d}"

    def _direct_citation_queries(self, fm: dict[str, Any]) -> list[dict[str, Any]]:
        """Strategy 1: one factual query per cited regulation."""
        cross_refs: list[str] = fm.get("cross_references", [])
        queries: list[dict[str, Any]] = []
        for i, citation in enumerate(cross_refs):
            template = DIRECT_CITATION_TEMPLATES[i % len(DIRECT_CITATION_TEMPLATES)]
            queries.append(
                {
                    "qid": self._next_qid("dc"),
                    "query": template.format(citation=citation),
                    "relevant_citations": [citation],
                    "relevant_doc_citations": [citation],
                    "expected_answer": None,
                    "query_type": "factual",
                    "difficulty": "easy",
                    "requires_synthesis": False,
                    "tags": ["case-derived", "citation-direct"],
                    "source_case": fm.get("accession_number", ""),
                    "case_document_type": fm.get("document_type", ""),
                    "is_unanswerable": False,
                    "unanswerable_reason": None,
                    "metadata": {
                        "filter": {
                            "type": "Eq",
                            "field": "corpus",
                            "value": "regulatory",
                        }
                    },
                }
            )
        return queries

    def _term_mapping_queries(
        self, fm: dict[str, Any], content: str,
    ) -> list[dict[str, Any]]:
        """Strategy 2: interpretive queries from term→regulation mapping."""
        matches = self.term_mapper.scan_content(content)
        cross_refs_lower = {c.lower() for c in fm.get("cross_references", [])}
        queries: list[dict[str, Any]] = []
        for i, match in enumerate(matches[:5]):
            overlaps = any(
                c.lower() in cross_refs_lower for c in match.citations
            )
            tags = ["case-derived", "term-mapping"]
            if overlaps:
                tags.append("overlaps-citation")
            template = TERM_MAPPING_TEMPLATES[i % len(TERM_MAPPING_TEMPLATES)]
            q: dict[str, Any] = {
                "qid": self._next_qid("tm"),
                "query": template.format(term=match.term),
                "relevant_citations": match.citations,
                "relevant_doc_citations": match.citations,
                "expected_answer": None,
                "query_type": "interpretive",
                "difficulty": "medium",
                "requires_synthesis": True,
                "tags": tags,
                "source_case": fm.get("accession_number", ""),
                "technical_term": match.term,
                "is_unanswerable": False,
                "unanswerable_reason": None,
                "metadata": {
                    "filter": {
                        "type": "Eq",
                        "field": "corpus",
                        "value": "regulatory",
                    }
                },
            }
            if overlaps:
                q["overlaps_direct_citation"] = True
            queries.append(q)
        return queries
```

### Step 5: Run tests to verify they pass

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`

Expected: All 5 tests PASS.

### Step 6: Lint and type check

Run: `./scripts/py -m ruff check src/rag/adapters/query_generation/case_query_generator.py tests/adapters/query_generation/test_case_query_generator.py`

Run: `./scripts/py -m mypy src/rag/adapters/query_generation/case_query_generator.py`

Fix any issues.

### Step 7: Commit

Suggested commit:
```
feat(query-gen): add CaseQueryGenerator with Strategy 1 (direct citation)
```
Files: `src/rag/adapters/query_generation/case_query_generator.py`, `tests/adapters/query_generation/test_case_query_generator.py`

---

## Task 5: Strategy 2 Tests — Term Mapping and Overlap Tagging

**Files:**
- Modify: `tests/adapters/query_generation/test_case_query_generator.py`

### Step 1: Write tests for Strategy 2

Append to `tests/adapters/query_generation/test_case_query_generator.py`:

```python
class TestStrategy2TermMapping:
    def test_generates_queries_for_matched_terms(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        assert len(s2) > 0

    def test_includes_technical_term_field(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        for q in s2:
            assert "technical_term" in q
            assert isinstance(q["technical_term"], str)

    def test_strategy2_field_values(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        for q in s2:
            assert q["difficulty"] == "medium"
            assert q["query_type"] == "interpretive"
            assert q["requires_synthesis"] is True
            assert q["source_case"] == "ML99999A001"

    def test_overlap_tagging_when_citation_in_cross_references(self):
        """ECCS maps to 10 CFR 50.46 which is in cross_references."""
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        eccs_queries = [q for q in s2 if q["technical_term"] == "ECCS"]
        if eccs_queries:
            q = eccs_queries[0]
            assert q.get("overlaps_direct_citation") is True
            assert "overlaps-citation" in q["tags"]

    def test_max_5_term_mapping_queries(self):
        terms = {
            f"term{i}": [f"10 CFR 50.{i}"]
            for i in range(10)
        }
        mapper = _make_term_mapper(terms)
        # Create content mentioning all 10 terms twice each
        content_lines = [f"term{i} appears here. term{i} again." for i in range(10)]
        case_md = FIXTURE_CASE_MD.split("## Content")[0] + "## Content\n\n" + "\n".join(content_lines)
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td, content=case_md)
            queries = gen.generate_from_case(case_path)
        s2 = [q for q in queries if "term-mapping" in q["tags"]]
        assert len(s2) <= 5


class TestQidUniqueness:
    def test_qids_are_unique_within_case(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        qids = [q["qid"] for q in queries]
        assert len(qids) == len(set(qids))

    def test_qids_are_unique_across_cases(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        all_queries: list[dict] = []
        with TemporaryDirectory() as td:
            for suffix in ["001", "002"]:
                case_md = FIXTURE_CASE_MD.replace("ML99999A001", f"ML99999A{suffix}")
                p = Path(td) / f"ML99999A{suffix}.md"
                p.write_text(case_md)
                all_queries.extend(gen.generate_from_case(p))
        qids = [q["qid"] for q in all_queries]
        assert len(qids) == len(set(qids))


class TestMaxQueriesPerCase:
    def test_truncates_to_max(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper, max_queries_per_case=3)
        with TemporaryDirectory() as td:
            case_path = _make_case_file(td)
            queries = gen.generate_from_case(case_path)
        assert len(queries) <= 3
```

### Step 2: Run tests to verify they pass

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`

Expected: All tests PASS (Strategy 2 was already implemented in Task 4).

### Step 3: Commit

Suggested commit:
```
test(query-gen): add Strategy 2, overlap tagging, QID, and truncation tests
```
Files: `tests/adapters/query_generation/test_case_query_generator.py`

---

## Task 6: CLI Script

**Files:**
- Create: `scripts/generate_case_queries.py`

### Step 1: Implement the CLI script

File: `scripts/generate_case_queries.py`

```python
"""Generate evaluation queries from NRC case documents.

Usage:
    ./scripts/py scripts/generate_case_queries.py \
        --case-dir corpus/us-nrc/cases/ \
        --output eval/datasets/case_generated_queries_DRAFT.jsonl \
        --term-map config/case_regulatory_terms.json \
        --max-queries-per-case 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate evaluation queries from NRC case documents.",
    )
    parser.add_argument(
        "--case-dir",
        type=Path,
        required=True,
        help="Directory containing case markdown files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL file path.",
    )
    parser.add_argument(
        "--term-map",
        type=Path,
        required=True,
        help="Path to term-to-regulation JSON dictionary.",
    )
    parser.add_argument(
        "--max-queries-per-case",
        type=int,
        default=20,
        help="Maximum queries to generate per case document (default: 20).",
    )
    args = parser.parse_args()

    if not args.case_dir.is_dir():
        print(f"Error: {args.case_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.term_map.is_file():
        print(f"Error: {args.term_map} not found", file=sys.stderr)
        sys.exit(1)

    from rag.adapters.query_generation.case_query_generator import CaseQueryGenerator
    from rag.adapters.query_generation.term_mapper import TermMapper

    mapper = TermMapper.from_json(args.term_map)
    generator = CaseQueryGenerator(
        term_mapper=mapper,
        max_queries_per_case=args.max_queries_per_case,
    )

    case_files = sorted(args.case_dir.rglob("*.md"))
    print(f"Found {len(case_files)} case files", file=sys.stderr)

    all_queries: list[dict] = []
    zero_match_cases = 0
    for case_path in case_files:
        queries = generator.generate_from_case(case_path)
        if not queries:
            zero_match_cases += 1
        all_queries.extend(queries)

    # Write JSONL
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary
    s1_count = sum(1 for q in all_queries if "citation-direct" in q["tags"])
    s2_count = sum(1 for q in all_queries if "term-mapping" in q["tags"])
    overlap_count = sum(1 for q in all_queries if q.get("overlaps_direct_citation"))
    print(f"\n--- Generation Summary ---", file=sys.stderr)
    print(f"Cases processed:    {len(case_files)}", file=sys.stderr)
    print(f"Zero-match cases:   {zero_match_cases}", file=sys.stderr)
    print(f"Total queries:      {len(all_queries)}", file=sys.stderr)
    print(f"  Strategy 1 (DC):  {s1_count}", file=sys.stderr)
    print(f"  Strategy 2 (TM):  {s2_count}", file=sys.stderr)
    print(f"  Overlap-tagged:   {overlap_count}", file=sys.stderr)
    print(f"Output: {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

### Step 2: Verify the script runs with --help

Run: `./scripts/py scripts/generate_case_queries.py --help`

Expected: Prints usage/help text without errors.

### Step 3: Smoke test on real corpus

Run: `./scripts/py scripts/generate_case_queries.py --case-dir corpus/us-nrc/cases/ --output eval/datasets/case_generated_queries_DRAFT.jsonl --term-map config/case_regulatory_terms.json --max-queries-per-case 20`

Expected: Prints summary with query counts. No tracebacks. Output file is created.

### Step 4: Inspect output

Run: `head -5 eval/datasets/case_generated_queries_DRAFT.jsonl`

Verify: Each line is valid JSON with the expected fields.

Run: `wc -l eval/datasets/case_generated_queries_DRAFT.jsonl`

Verify: Non-trivial number of queries (expect 200+ from 170 cases).

### Step 5: Lint

Run: `./scripts/py -m ruff check scripts/generate_case_queries.py`

Fix any issues.

### Step 6: Commit

Suggested commit:
```
feat(query-gen): add CLI script for case query generation
```
Files: `scripts/generate_case_queries.py`

---

## Task 7: Full Validation Pass

### Step 1: Run entire test suite

Run: `make test`

Expected: All tests pass, including the new query_generation tests.

### Step 2: Lint everything

Run: `make lint`

Expected: No errors.

### Step 3: Type check

Run: `./scripts/py -m mypy src/rag/adapters/query_generation/`

Expected: No errors.

### Step 4: Commit (if any fixes were needed)

Suggested commit:
```
chore(query-gen): fix lint/type issues from full validation
```

---

## Summary of Suggested Commits

| # | Message | Files |
|---|---------|-------|
| 1 | `feat(query-gen): add TermMapper with JSON loading and lookup` | `src/rag/adapters/query_generation/{__init__,term_mapper}.py`, `tests/adapters/query_generation/{__init__,test_term_mapper}.py` |
| 2 | `test(query-gen): add scan_content test coverage for TermMapper` | `tests/adapters/query_generation/test_term_mapper.py` |
| 3 | `feat(query-gen): add seed term-to-regulation dictionary (85 entries)` | `config/case_regulatory_terms.json` |
| 4 | `feat(query-gen): add CaseQueryGenerator with Strategy 1 (direct citation)` | `src/rag/adapters/query_generation/case_query_generator.py`, `tests/adapters/query_generation/test_case_query_generator.py` |
| 5 | `test(query-gen): add Strategy 2, overlap tagging, QID, and truncation tests` | `tests/adapters/query_generation/test_case_query_generator.py` |
| 6 | `feat(query-gen): add CLI script for case query generation` | `scripts/generate_case_queries.py` |
| 7 | `chore(query-gen): fix lint/type issues from full validation` | (if needed) |
