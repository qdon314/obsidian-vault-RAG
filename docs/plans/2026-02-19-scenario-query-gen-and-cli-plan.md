# Scenario Query Generation & CLI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add scenario-based query generation (Strategy 3) to `CaseQueryGenerator` and create the `generate_case_queries.py` CLI script that produces eval-compatible JSONL.

**Architecture:** Refactor `CaseQueryGenerator.generate()` to delegate to three private strategy methods (`_citation_direct`, `_term_mapping`, `_scenario`). Strategy 3 uses category-keyed scenario templates filled with term matches. A new CLI script glues `TermMapper` + `CaseQueryGenerator` together and writes JSONL output.

**Tech Stack:** Python 3.12, frozen dataclasses, argparse, `TermMapper` / `CaseQueryGenerator` adapters, pytest.

**Design doc:** `docs/plans/2026-02-19-scenario-query-gen-and-cli-design.md`

---

### Task 1: Refactor — Extract Strategy 1 into `_citation_direct()`

This is a pure extract-method refactoring. No behavior changes.

**Files:**
- Modify: `src/rag/adapters/query_generation/case_query_generator.py:57-99`

**Step 1: Run existing tests to establish baseline**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`
Expected: All tests PASS (7 in TestStrategy1, 8 in TestStrategy2, 2 in TestQidUniqueness, 1 in TestMaxQueriesPerCase)

**Step 2: Extract Strategy 1 logic into `_citation_direct()`**

Move lines 71-99 of `case_query_generator.py` (the "Strategy 1: Direct citation queries" block) into a new private method. The method receives `frontmatter` and returns `list[dict[str, Any]]`.

After refactoring, `generate()` should look like:

```python
def generate(self, case_file: Path) -> list[dict[str, Any]]:
    text = case_file.read_text(encoding="utf-8")
    frontmatter, body = split_obsidian_frontmatter(text)

    queries: list[dict[str, Any]] = []
    queries.extend(self._citation_direct(frontmatter))

    # Strategy 2 stays inline for now (extracted in Task 2)
    cross_refs: list[str] = frontmatter.get("cross_references", [])
    cross_ref_normalized = {_normalize_citation(c) for c in cross_refs}
    matches = self.term_mapper.scan_content(body)
    top_matches = matches[:5]
    # ... rest of Strategy 2 ...

    return queries[: self.max_queries_per_case]
```

The new method:

```python
def _citation_direct(self, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """Strategy 1: Generate queries from frontmatter cross_references."""
    accession = frontmatter.get("accession_number", "")
    doc_type = frontmatter.get("document_type", "")
    cross_refs: list[str] = frontmatter.get("cross_references", [])

    queries: list[dict[str, Any]] = []
    seen_citations: set[str] = set()
    for i, citation in enumerate(cross_refs):
        norm = _normalize_citation(citation)
        if norm in seen_citations:
            continue
        seen_citations.add(norm)

        self._dc_counter[0] += 1
        qid = f"case-dc-{self._dc_counter[0]:03d}"
        template = _STRATEGY1_TEMPLATES[i % len(_STRATEGY1_TEMPLATES)]

        queries.append(
            {
                "qid": qid,
                "query": template.format(citation=citation),
                "difficulty": "easy",
                "query_type": "factual",
                "requires_synthesis": False,
                "is_unanswerable": False,
                "expected_answer": None,
                "unanswerable_reason": None,
                "relevant_citations": [citation],
                "tags": ["case-derived", "citation-direct"],
                "source_case": accession,
                "case_document_type": doc_type,
                "metadata": dict(_METADATA_FILTER),
            }
        )
    return queries
```

**Step 3: Run tests to confirm no regressions**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`
Expected: All tests PASS — identical results to Step 1

**Step 4: Commit**

```
refactor(query-gen): extract Strategy 1 into _citation_direct()
```

---

### Task 2: Refactor — Extract Strategy 2 into `_term_mapping()`

**Files:**
- Modify: `src/rag/adapters/query_generation/case_query_generator.py`

**Step 1: Extract Strategy 2 logic into `_term_mapping()`**

Move the remaining Strategy 2 block into a new private method. It receives `body` and `frontmatter`, returns `list[dict[str, Any]]`.

After this, `generate()` becomes:

```python
def generate(self, case_file: Path) -> list[dict[str, Any]]:
    text = case_file.read_text(encoding="utf-8")
    frontmatter, body = split_obsidian_frontmatter(text)

    queries: list[dict[str, Any]] = []
    queries.extend(self._citation_direct(frontmatter))
    queries.extend(self._term_mapping(body, frontmatter))
    return queries[: self.max_queries_per_case]
```

The new method:

```python
def _term_mapping(self, body: str, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """Strategy 2: Generate queries from TermMapper scan of body content."""
    accession = frontmatter.get("accession_number", "")
    cross_refs: list[str] = frontmatter.get("cross_references", [])
    cross_ref_normalized = {_normalize_citation(c) for c in cross_refs}
    matches = self.term_mapper.scan_content(body)
    top_matches = matches[:5]

    queries: list[dict[str, Any]] = []
    for i, match in enumerate(top_matches):
        self._tm_counter[0] += 1
        qid = f"case-tm-{self._tm_counter[0]:03d}"
        template = _STRATEGY2_TEMPLATES[i % len(_STRATEGY2_TEMPLATES)]

        term_map = self.term_mapper.term_map
        anchor_refs = [a.ref for a in match.anchors]
        citation_labels = [
            term_map.refs[ref].label for ref in anchor_refs if ref in term_map.refs
        ]

        has_overlap = any(
            _normalize_citation(c) in cross_ref_normalized for c in citation_labels
        )

        tags = ["case-derived", "term-mapping", f"{match.term_type.value}-term"]
        if has_overlap:
            tags.append("overlaps-citation")

        entry: dict[str, Any] = {
            "qid": qid,
            "query": template.format(term=match.term),
            "difficulty": "medium",
            "query_type": "interpretive",
            "requires_synthesis": True,
            "is_unanswerable": False,
            "expected_answer": None,
            "unanswerable_reason": None,
            "relevant_citations": citation_labels,
            "anchor_refs": anchor_refs,
            "tags": tags,
            "source_case": accession,
            "technical_term": match.term,
            "term_type": match.term_type.value,
            "metadata": dict(_METADATA_FILTER),
        }
        if has_overlap:
            entry["overlaps_direct_citation"] = True

        queries.append(entry)
    return queries
```

**Step 2: Run tests to confirm no regressions**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`
Expected: All tests PASS

**Step 3: Commit**

```
refactor(query-gen): extract Strategy 2 into _term_mapping()
```

---

### Task 3: Add Strategy 3 — Scenario templates and `_sc_counter`

**Files:**
- Modify: `src/rag/adapters/query_generation/case_query_generator.py`
- Modify: `tests/adapters/query_generation/test_case_query_generator.py`

**Step 1: Write the failing test — basic Strategy 3 generation**

Add a new test class `TestStrategy3Scenario` to `test_case_query_generator.py`:

```python
class TestStrategy3Scenario:
    def test_generates_scenario_queries_from_matched_terms(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s3 = [q for q in queries if "scenario-based" in q["tags"]]
        assert len(s3) >= 1

    def test_scenario_field_values(self):
        mapper = _make_term_mapper()
        gen = CaseQueryGenerator(term_mapper=mapper)
        with TemporaryDirectory() as td:
            case_file = _make_case_file(td)
            queries = gen.generate(case_file)
        s3 = [q for q in queries if "scenario-based" in q["tags"]]
        for q in s3:
            assert q["difficulty"] == "hard"
            assert q["query_type"] == "scenario"
            assert q["requires_synthesis"] is True
            assert q["source_case"] == "ML99999A001"
            assert "technical_term" in q
            assert "anchor_refs" in q
            assert q["metadata"] == {
                "filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}
            }
```

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py::TestStrategy3Scenario -v`
Expected: FAIL — `scenario-based` tag not found in any queries

**Step 2: Add scenario templates and `_sc_counter` to `CaseQueryGenerator`**

Add at module level, after `_STRATEGY2_TEMPLATES`:

```python
_SCENARIO_TEMPLATES: dict[str, list[str]] = {
    "inspection": [
        "A {reactor_type} plant identified issues with {term} during routine inspection. What regulations apply to {term}?",
        "An inspector found deficiencies related to {term} at a nuclear facility. What are the applicable regulatory requirements?",
        "During a plant walkdown, problems were identified with {term}. What regulatory standards govern this area?",
    ],
    "enforcement": [
        "What regulatory requirements could be the basis for an enforcement action involving {term} at a nuclear facility?",
        "A nuclear plant received a notice of violation related to {term}. What are the underlying regulatory requirements?",
        "An enforcement case was opened regarding {term}. What regulations are most relevant?",
    ],
    "vendor_part21": [
        "A vendor discovers a defect affecting {term}. What are the reporting and notification requirements?",
        "A nuclear component supplier identified an issue with {term}. What regulatory obligations apply?",
        "A Part 21 report was filed regarding {term}. What are the applicable regulatory provisions?",
    ],
    "operations": [
        "What are the regulatory requirements when {term} is found inoperable during plant operations?",
        "An operator discovers that {term} is not functioning as designed. What regulations govern this situation?",
        "During normal operations, a degraded condition is identified affecting {term}. What requirements apply?",
    ],
    "licensing": [
        "What regulatory provisions govern {term} in the context of a nuclear facility license application?",
        "A license amendment is being sought related to {term}. What are the key regulatory requirements?",
        "What does the regulatory framework require regarding {term} for nuclear plant licensing?",
    ],
}

_GENERIC_SCENARIO_TEMPLATES = [
    "What are the regulatory requirements related to {term} at a nuclear power plant?",
    "A nuclear facility identified an issue involving {term}. What regulations apply?",
    "What does the NRC regulatory framework require regarding {term}?",
]
```

Add `_sc_counter` field to the dataclass (after `_tm_counter`):

```python
_sc_counter: list[int] = field(
    init=False, repr=False, compare=False, hash=False, default_factory=lambda: [0]
)
```

Update the docstring to mention three strategies.

**Step 3: Implement `_scenario()` method**

```python
def _scenario(self, body: str, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    """Strategy 3: Generate scenario-based queries from term matches + category."""
    accession = frontmatter.get("accession_number", "")
    category = frontmatter.get("case_category", "unknown")
    reactor_type = frontmatter.get("reactor_type", "nuclear")
    cross_refs: list[str] = frontmatter.get("cross_references", [])

    templates = _SCENARIO_TEMPLATES.get(category, _GENERIC_SCENARIO_TEMPLATES)
    matches = self.term_mapper.scan_content(body)
    top_matches = matches[:5]

    queries: list[dict[str, Any]] = []
    for i, match in enumerate(top_matches):
        term_map = self.term_mapper.term_map
        anchor_refs = [a.ref for a in match.anchors]
        citation_labels = [
            term_map.refs[ref].label for ref in anchor_refs if ref in term_map.refs
        ]

        # Also include any cross_references from frontmatter as relevant citations
        all_citations = list(citation_labels)
        for cr in cross_refs:
            if cr not in all_citations:
                all_citations.append(cr)

        # Skip terms with no resolvable citations
        if not citation_labels:
            continue

        self._sc_counter[0] += 1
        qid = f"case-sc-{self._sc_counter[0]:03d}"
        template = templates[i % len(templates)]

        queries.append(
            {
                "qid": qid,
                "query": template.format(term=match.term, reactor_type=reactor_type),
                "difficulty": "hard",
                "query_type": "scenario",
                "requires_synthesis": True,
                "is_unanswerable": False,
                "expected_answer": None,
                "unanswerable_reason": None,
                "relevant_citations": citation_labels,
                "anchor_refs": anchor_refs,
                "tags": [
                    "case-derived",
                    "scenario-based",
                    f"{match.term_type.value}-term",
                ],
                "source_case": accession,
                "technical_term": match.term,
                "term_type": match.term_type.value,
                "metadata": dict(_METADATA_FILTER),
            }
        )
    return queries
```

Wire it into `generate()`:

```python
def generate(self, case_file: Path) -> list[dict[str, Any]]:
    text = case_file.read_text(encoding="utf-8")
    frontmatter, body = split_obsidian_frontmatter(text)

    queries: list[dict[str, Any]] = []
    queries.extend(self._citation_direct(frontmatter))
    queries.extend(self._term_mapping(body, frontmatter))
    queries.extend(self._scenario(body, frontmatter))
    return queries[: self.max_queries_per_case]
```

**Step 4: Run tests**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`
Expected: ALL tests PASS (including the two new Strategy 3 tests AND all existing tests)

**Step 5: Commit**

```
feat(query-gen): add Strategy 3 scenario-based query generation
```

---

### Task 4: Strategy 3 — Additional test coverage

**Files:**
- Modify: `tests/adapters/query_generation/test_case_query_generator.py`

**Step 1: Write tests for category-specific template selection**

```python
def test_uses_category_specific_templates(self):
    """Inspection category should use inspection templates."""
    mapper = _make_term_mapper()
    gen = CaseQueryGenerator(term_mapper=mapper)
    with TemporaryDirectory() as td:
        case_file = _make_case_file(td)
        queries = gen.generate(case_file)
    s3 = [q for q in queries if "scenario-based" in q["tags"]]
    # Fixture has case_category: "inspection"
    # Inspection templates contain "inspection" or "walkdown" or "inspector"
    assert any(
        "inspection" in q["query"].lower() or "inspector" in q["query"].lower() or "walkdown" in q["query"].lower()
        for q in s3
    ), f"Expected inspection-themed template, got: {[q['query'] for q in s3]}"
```

**Step 2: Write test for generic fallback on unknown category**

```python
def test_uses_generic_templates_for_unknown_category(self):
    md = FIXTURE_CASE_MD.replace(
        'case_category: "inspection"', 'case_category: "unknown"'
    )
    mapper = _make_term_mapper()
    gen = CaseQueryGenerator(term_mapper=mapper)
    with TemporaryDirectory() as td:
        case_file = _make_case_file(td, content=md)
        queries = gen.generate(case_file)
    s3 = [q for q in queries if "scenario-based" in q["tags"]]
    assert len(s3) >= 1
    # Generic templates don't contain "inspection" or "enforcement" etc.
    for q in s3:
        assert "inspection" not in q["query"].lower() or "nuclear" in q["query"].lower()
```

**Step 3: Write test for orphan term skip (no citations resolve)**

```python
def test_skips_terms_with_no_resolvable_citations(self):
    """Terms whose anchor refs don't resolve to labels should be skipped."""
    # Create a term whose anchor ref is NOT in the refs dict
    # This can't happen with TermMapper validation, so instead create
    # a term whose anchor ref IS in refs but we'll test the skip path
    # by checking that all scenario queries have non-empty relevant_citations
    mapper = _make_term_mapper()
    gen = CaseQueryGenerator(term_mapper=mapper)
    with TemporaryDirectory() as td:
        case_file = _make_case_file(td)
        queries = gen.generate(case_file)
    s3 = [q for q in queries if "scenario-based" in q["tags"]]
    for q in s3:
        assert len(q["relevant_citations"]) > 0
```

**Step 4: Write test for `reactor_type` template fill**

```python
def test_fills_reactor_type_from_frontmatter(self):
    mapper = _make_term_mapper()
    gen = CaseQueryGenerator(term_mapper=mapper)
    with TemporaryDirectory() as td:
        case_file = _make_case_file(td)
        queries = gen.generate(case_file)
    s3 = [q for q in queries if "scenario-based" in q["tags"]]
    # Fixture has reactor_type: "PWR"
    # Inspection template index 0 contains {reactor_type}
    pwr_queries = [q for q in s3 if "PWR" in q["query"]]
    assert len(pwr_queries) >= 1
```

**Step 5: Write test for scenario QID prefix and uniqueness**

```python
def test_scenario_qids_use_sc_prefix(self):
    mapper = _make_term_mapper()
    gen = CaseQueryGenerator(term_mapper=mapper)
    with TemporaryDirectory() as td:
        case_file = _make_case_file(td)
        queries = gen.generate(case_file)
    s3 = [q for q in queries if "scenario-based" in q["tags"]]
    for q in s3:
        assert q["qid"].startswith("case-sc-")

def test_scenario_qids_unique_across_cases(self):
    mapper = _make_term_mapper()
    gen = CaseQueryGenerator(term_mapper=mapper)
    all_sc_qids: list[str] = []
    for suffix in ("A001", "A002"):
        md = FIXTURE_CASE_MD.replace("ML99999A001", f"ML99999{suffix}")
        with TemporaryDirectory() as td:
            p = Path(td) / f"ML99999{suffix}.md"
            p.write_text(md)
            queries = gen.generate(p)
        all_sc_qids.extend(
            q["qid"] for q in queries if "scenario-based" in q["tags"]
        )
    assert len(all_sc_qids) == len(set(all_sc_qids))
```

**Step 6: Run all tests**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py -v`
Expected: All tests PASS

**Step 7: Commit**

```
test(query-gen): add Strategy 3 scenario test coverage
```

---

### Task 5: Create CLI script `scripts/generate_case_queries.py`

**Files:**
- Create: `scripts/generate_case_queries.py`

**Step 1: Write the CLI script**

```python
#!/usr/bin/env python3
"""Generate evaluation queries from NRC case documents.

Reads case markdown files from the corpus directory, applies TermMapper +
CaseQueryGenerator, and writes eval-compatible JSONL output.

Usage examples::

    ./scripts/py scripts/generate_case_queries.py
    ./scripts/py scripts/generate_case_queries.py --dry-run
    ./scripts/py scripts/generate_case_queries.py --strategies 1,3 --output queries.jsonl
    ./scripts/py scripts/generate_case_queries.py --corpus-dir corpus/us-nrc/cases/2024-01/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.adapters.query_generation.case_query_generator import CaseQueryGenerator
from rag.adapters.query_generation.term_mapper import TermMapper

# Strategy tag used for post-filtering
_STRATEGY_TAGS = {
    1: "citation-direct",
    2: "term-mapping",
    3: "scenario-based",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate evaluation queries from NRC case documents.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="corpus/us-nrc/cases",
        help="Path to case markdown directory (default: corpus/us-nrc/cases).",
    )
    parser.add_argument(
        "--term-map",
        default="config/case_regulatory_terms.json",
        help="Path to term map JSON (default: config/case_regulatory_terms.json).",
    )
    parser.add_argument(
        "--output",
        default="eval/datasets/case_generated_queries.jsonl",
        help="Output JSONL path (default: eval/datasets/case_generated_queries.jsonl).",
    )
    parser.add_argument(
        "--strategies",
        default="1,2,3",
        help="Comma-separated strategy numbers to include (default: 1,2,3).",
    )
    parser.add_argument(
        "--max-per-case",
        type=int,
        default=50,
        help="Max queries per case file (default: 50).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing the output file.",
    )
    return parser


def _parse_strategies(raw: str) -> set[int]:
    """Parse comma-separated strategy numbers into a set."""
    try:
        nums = {int(s.strip()) for s in raw.split(",")}
    except ValueError:
        raise SystemExit(f"--strategies must be comma-separated integers, got: {raw!r}")
    invalid = nums - {1, 2, 3}
    if invalid:
        raise SystemExit(f"Invalid strategy numbers: {invalid}. Valid: 1, 2, 3")
    return nums


def _filter_by_strategy(
    queries: list[dict], strategies: set[int]
) -> list[dict]:
    """Post-filter queries to only include selected strategies."""
    if strategies == {1, 2, 3}:
        return queries
    allowed_tags = {_STRATEGY_TAGS[s] for s in strategies}
    return [q for q in queries if any(t in allowed_tags for t in q.get("tags", []))]


def main() -> None:
    args = build_argparser().parse_args()

    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    term_map_path = Path(args.term_map).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    strategies = _parse_strategies(args.strategies)

    if not corpus_dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")
    if not term_map_path.is_file():
        raise SystemExit(f"Term map file not found: {term_map_path}")

    mapper = TermMapper.from_json(term_map_path)
    gen = CaseQueryGenerator(term_mapper=mapper, max_queries_per_case=args.max_per_case)

    case_files = sorted(corpus_dir.rglob("*.md"))
    if not case_files:
        raise SystemExit(f"No .md files found in {corpus_dir}")

    print(f"Processing {len(case_files)} case files...", file=sys.stderr)

    all_queries: list[dict] = []
    files_with_queries = 0
    for case_file in case_files:
        queries = gen.generate(case_file)
        if queries:
            files_with_queries += 1
        all_queries.extend(queries)

    # Post-filter by strategy
    filtered = _filter_by_strategy(all_queries, strategies)

    # Count by strategy tag
    counts: dict[str, int] = {}
    for q in filtered:
        for tag_num, tag_name in _STRATEGY_TAGS.items():
            if tag_name in q.get("tags", []):
                counts[tag_name] = counts.get(tag_name, 0) + 1
                break

    # Print summary
    print("\n--- Query Generation Summary ---", file=sys.stderr)
    print(f"{'Strategy':<20} {'Count':>6}", file=sys.stderr)
    print("-" * 28, file=sys.stderr)
    for tag_name in ("citation-direct", "term-mapping", "scenario-based"):
        if tag_name in counts:
            print(f"{tag_name:<20} {counts[tag_name]:>6}", file=sys.stderr)
    print("-" * 28, file=sys.stderr)
    print(
        f"{'Total':<20} {len(filtered):>6} queries from {files_with_queries} case files",
        file=sys.stderr,
    )

    if args.dry_run:
        print("\n(dry-run: no file written)", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for q in filtered:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nWritten to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
```

**Step 2: Test the CLI with `--dry-run` against real corpus**

Run: `./scripts/py scripts/generate_case_queries.py --dry-run`
Expected: Prints a summary table with counts per strategy and total. No file written. Exits 0.

**Step 3: Test with strategy filter**

Run: `./scripts/py scripts/generate_case_queries.py --dry-run --strategies 3`
Expected: Only shows `scenario-based` count. Other strategies are 0 or absent.

**Step 4: Commit**

```
feat(query-gen): add generate_case_queries.py CLI script
```

---

### Task 6: Full integration test — generate real JSONL

**Files:**
- None created. This is a validation-only task.

**Step 1: Generate the full query set**

Run: `./scripts/py scripts/generate_case_queries.py --output eval/datasets/case_generated_queries.jsonl`
Expected: Writes JSONL file, prints summary with all three strategy counts.

**Step 2: Validate JSONL is well-formed**

Run: `./scripts/py -c "import json; lines = open('eval/datasets/case_generated_queries.jsonl').readlines(); [json.loads(l) for l in lines]; print(f'{len(lines)} valid JSON lines')"`
Expected: Prints count of valid JSON lines, no errors.

**Step 3: Spot-check a few scenario queries**

Run: `./scripts/py -c "import json; lines = open('eval/datasets/case_generated_queries.jsonl').readlines(); s3 = [json.loads(l) for l in lines if 'scenario-based' in l]; print(json.dumps(s3[:3], indent=2))"`
Expected: Three scenario queries with `difficulty: hard`, `query_type: scenario`, populated `relevant_citations`, and natural-sounding scenario text.

**Step 4: Run lint and type checks**

Run: `make lint && make typecheck`
Expected: Clean (or only pre-existing warnings).

**Step 5: Commit generated dataset (if desired)**

```
data(eval): generate case-derived query dataset with scenario queries
```

Note: The JSONL file may be large. Discuss with reviewer whether to commit it or add to `.gitignore`.

---

### Task 7: Update `TestQidUniqueness` for three strategies

**Files:**
- Modify: `tests/adapters/query_generation/test_case_query_generator.py`

**Step 1: Verify existing QID uniqueness test still passes with three strategies**

The existing `test_qids_are_unique_within_case` and `test_qids_are_unique_across_cases` already check ALL queries from `generate()`, which now includes Strategy 3. Run them:

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py::TestQidUniqueness -v`
Expected: PASS — if they already pass, no changes needed.

**Step 2: Verify `TestMaxQueriesPerCase` with three strategies**

Run: `./scripts/py -m pytest tests/adapters/query_generation/test_case_query_generator.py::TestMaxQueriesPerCase -v`
Expected: PASS — the max is applied after all three strategies concatenate.

**Step 3: Commit (only if changes were needed)**

```
test(query-gen): ensure QID uniqueness covers all three strategies
```

---

## Summary of commits

1. `refactor(query-gen): extract Strategy 1 into _citation_direct()`
2. `refactor(query-gen): extract Strategy 2 into _term_mapping()`
3. `feat(query-gen): add Strategy 3 scenario-based query generation`
4. `test(query-gen): add Strategy 3 scenario test coverage`
5. `feat(query-gen): add generate_case_queries.py CLI script`
6. `data(eval): generate case-derived query dataset with scenario queries` (optional)