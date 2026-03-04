# Scenario Query Generation (Strategy 3) + CLI Script

**Date:** 2026-02-19
**Phase:** 5 (partial) — NRC Case Ingestion

---

## Summary

Add a third query generation strategy ("scenario-based") to `CaseQueryGenerator` and create
the `scripts/generate_case_queries.py` CLI script that produces eval-compatible JSONL output.

This completes the query generation pipeline: the adapters (Phase 4) get a scenario strategy,
and the CLI script (Phase 5) provides the operational interface.

---

## Strategy 3: Scenario-Based Queries

### Motivation

Strategies 1 and 2 leak answer signals into the query:

- **Strategy 1 (citation-direct):** Names the regulation explicitly ("What does 10 CFR 50.46 require?")
- **Strategy 2 (term-mapping):** Uses regulatory vocabulary ("What regulations govern ECCS?")

Strategy 3 tests whether retrieval can bridge from a **real-world situation description** to the
applicable regulation — the way an actual user would query the system.

### Design

**Signal-template approach:** Combine structured metadata from case frontmatter with term
matches to fill scenario templates that describe situations without naming regulations.

**Inputs (already available):**

| Signal | Source | Purpose |
|---|---|---|
| `case_category` | Frontmatter | Select template family |
| `case_subcategory` | Frontmatter | Available but not used in v1 |
| `reactor_type` | Frontmatter (optional) | Template fill for specificity |
| Term matches | `TermMapper.scan_content(body)` | Technical subject for the scenario |
| `cross_references` | Frontmatter | Contribute to `relevant_citations` |

**Template registry** — `dict[CaseCategory, list[str]]`, keyed by category enum:

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

Categories not in the registry fall back to `_GENERIC_SCENARIO_TEMPLATES`.

**Query construction (per term match):**

1. Look up `case_category` from frontmatter → select template list (or generic fallback)
2. Fill `{term}` from match; fill `{reactor_type}` if present (else substitute "nuclear")
3. Resolve `relevant_citations` from:
   - Term's anchor refs → `term_map.refs[ref].label`
   - Any overlapping `cross_references` from frontmatter
4. **Skip if no `relevant_citations` resolve** — no orphan queries
5. Rotate template via counter mod list-length

**Output fields per query:**

```python
{
    "qid": "case-sc-001",           # prefix case-sc-, monotonic across cases
    "query": "...",
    "difficulty": "hard",
    "query_type": "scenario",
    "requires_synthesis": True,
    "is_unanswerable": False,
    "expected_answer": None,
    "unanswerable_reason": None,
    "relevant_citations": [...],     # from anchor refs + cross_references
    "tags": ["case-derived", "scenario-based", "<term_type>-term"],
    "source_case": "<accession_number>",
    "technical_term": "<matched term>",
    "term_type": "<anchor|contextual|cross_cutting>",
    "anchor_refs": [...],
    "metadata": {"filter": {"type": "Eq", "field": "corpus", "value": "regulatory"}},
}
```

**Overlap with Strategy 2:** Allowed by design. Same terms may appear in both, but the query
*type* differs (`interpretive` vs `scenario`) and they test different retrieval behaviors.
Strategy 2 asks "what does the NRC require regarding X?" while Strategy 3 asks "this situation
happened with X — what applies?"

**Top-N term matches:** 5 (same as Strategy 2).

---

## Refactor: Extract Strategy Methods

Currently, Strategies 1 and 2 are inline in `generate()`. Refactor to extract each into a
private method before adding Strategy 3.

**Before:**
```python
def generate(self, case_file: Path) -> list[dict[str, Any]]:
    # ... ~80 lines of inline strategy logic
```

**After:**
```python
def generate(self, case_file: Path) -> list[dict[str, Any]]:
    text = case_file.read_text(encoding="utf-8")
    frontmatter, body = split_obsidian_frontmatter(text)
    queries: list[dict[str, Any]] = []
    queries.extend(self._citation_direct(frontmatter))
    queries.extend(self._term_mapping(body, frontmatter))
    queries.extend(self._scenario(body, frontmatter))
    return queries[:self.max_queries_per_case]

def _citation_direct(self, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    # Existing Strategy 1 logic, unchanged

def _term_mapping(self, body: str, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    # Existing Strategy 2 logic, unchanged

def _scenario(self, body: str, frontmatter: dict[str, Any]) -> list[dict[str, Any]]:
    # NEW Strategy 3
```

Add a `_sc_counter` field (same mutable-list-in-frozen-dataclass pattern as existing counters).

This is a pure extract-method refactoring for S1/S2. Existing tests serve as safety net.

---

## CLI Script: `scripts/generate_case_queries.py`

### Arguments

```
scripts/generate_case_queries.py
  --corpus-dir PATH       # Path to case markdown directory
                          # Default: corpus/us-nrc/cases/
  --term-map PATH         # Path to term map JSON
                          # Default: config/case_regulatory_terms.json
  --output PATH           # Output JSONL path
                          # Default: eval/datasets/case_generated_queries.jsonl
  --strategies 1,2,3      # Comma-separated strategy numbers to run
                          # Default: 1,2,3 (all)
  --max-per-case INT      # Override max_queries_per_case
                          # Default: 50
  --dry-run               # Print stats without writing file
```

### Workflow

1. Parse arguments
2. Load `TermMapper.from_json(term_map_path)`
3. Build `CaseQueryGenerator(term_mapper=mapper, max_queries_per_case=max_per_case)`
4. Glob `corpus_dir/**/*.md` for case files
5. For each case file: call `generator.generate(case_file)`
6. If `--strategies` filter: post-filter queries by tag (`citation-direct`, `term-mapping`, `scenario-based`)
7. Write as JSONL (one JSON object per line) — or print summary if `--dry-run`
8. Print summary to stderr:
   ```
   Strategy         Count
   citation-direct     12
   term-mapping        34
   scenario-based      18
   ────────────────────────
   Total               64 queries from 8 case files
   ```

### Output format

One JSON object per line, compatible with `eval/datasets/regulatory_adversarial.jsonl` schema
so the eval harness can consume both dataset files.

### `--strategies` filtering

Applied as a post-filter on the `tags` field rather than modifying `generate()`. This keeps
the adapter clean — the CLI script is responsible for filtering, not the adapter.

---

## Files Changed

| File | Change type | Description |
|---|---|---|
| `src/rag/adapters/query_generation/case_query_generator.py` | Modify | Extract strategy methods, add `_scenario()`, add `_sc_counter`, add scenario templates |
| `tests/adapters/query_generation/test_case_query_generator.py` | Modify | Add Strategy 3 tests: template selection by category, term filling, citation resolution, orphan skip, QID format, generic fallback |
| `scripts/generate_case_queries.py` | New | CLI script with argparse, JSONL output, dry-run, strategy filtering |

**Not changed:**
- `term_mapper.py` — no modifications needed
- `settings.toml` — CLI defaults are sufficient
- `container.py` — query gen stays outside the container

---

## Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Scenario templates sound too similar across categories | Start with distinct templates per category; iterate based on generated output review |
| Few terms resolve citations (many orphan skips) | The term map already has 67 terms with anchor refs; most will resolve. Monitor skip rate in dry-run output |
| Strategy 3 queries too similar to Strategy 2 | Accepted by design — different query types test different retrieval behaviors |

---

## Follow-ups (not in scope)

- Consume `TermActions.retrieval_boost` / `adams_prefilter` downstream in the retrieval pipeline
- Add `--strategies` as a `generate()` parameter (if adapter-level filtering becomes needed)
- Add `[query_generation]` settings section (if paths need to be configurable beyond CLI defaults)
- Cross-case deduplication (same regulation+term from multiple cases)