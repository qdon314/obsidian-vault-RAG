# Term Mapping Dictionary & Query Generation (Strategies 1-2) — Design

## Scope

Phase 4 of the NRC case ingestion plan: build a term-to-regulation mapping dictionary and a query generator that produces eval queries from case documents using two strategies:

1. **Direct Citation Queries** (Strategy 1) — one factual query per explicit CFR citation in case frontmatter
2. **Term Mapping Queries** (Strategy 2) — interpretive queries derived from informal/technical terms found in case text, mapped to regulations via the dictionary

Strategies 3-6 (violation context, adversarial, facility-specific, abstention) are deferred to a follow-up.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Automation level | Template-only (no LLM) | Strategies 1-2 are mechanical; get output fast, learn before investing in harder strategies |
| Citation format in dictionary | Human-readable (`10 CFR 50.46`) | Dictionary is hand-authored; readability over programmatic consistency |
| Term extraction method | Case-insensitive exact substring | Simple, predictable, debuggable; variant coverage via dictionary entries |
| Strategy 1/2 overlap | Generate both, tag overlap | Preserves full output for analysis; tag enables filtering during curation |
| Template variety | 3 templates per strategy, deterministic selection | Minimal surface diversity without complexity |
| Code location | `src/rag/adapters/query_generation/` | Proper adapter with domain logic; deserves same structure as ingestion adapters |

## Component 1: Term Mapping Dictionary

**File:** `config/case_regulatory_terms.json`

Flat JSON: keys are lowercase term strings, values are lists of human-readable CFR citations.

```json
{
  "ECCS": ["10 CFR 50.46", "10 CFR 50.34"],
  "emergency core cooling": ["10 CFR 50.46"],
  "accumulator": ["10 CFR 50.46", "10 CFR 50.36"],
  "peak cladding temperature": ["10 CFR 50.46"],
  "technical specification": ["10 CFR 50.36"],
  "LCO": ["10 CFR 50.36"],
  "limiting condition for operation": ["10 CFR 50.36"],
  "surveillance requirement": ["10 CFR 50.36"],
  "surveillance testing": ["10 CFR 50.36"],
  "maintenance rule": ["10 CFR 50.65"],
  "license amendment": ["10 CFR 50.59", "10 CFR 50.90"],
  "defect reporting": ["10 CFR 21"],
  "event notification": ["10 CFR 50.72"],
  "licensee event report": ["10 CFR 50.73"]
}
```

Seed: ~50-100 entries covering ECCS/safety analysis, tech specs, change control, maintenance, reporting, security, quality assurance, licensing. Morphological variants are separate entries (e.g., "surveillance test" and "surveillance testing" both present).

Validation at load time: all keys are strings, all values are non-empty lists of strings. No separate schema file.

## Component 2: TermMapper Adapter

**File:** `src/rag/adapters/query_generation/term_mapper.py`

```python
@dataclass(frozen=True, slots=True)
class TermMatch:
    term: str
    citations: list[str]
    frequency: int

@dataclass(frozen=True, slots=True)
class TermMapper:
    _terms: dict[str, list[str]]

    @classmethod
    def from_json(cls, path: Path) -> "TermMapper": ...

    def lookup(self, term: str) -> list[str]: ...

    def scan_content(self, content: str) -> list[TermMatch]: ...
```

`scan_content` lowercases content once, checks each dictionary term via `term.lower() in content_lower`, counts occurrences, returns matches sorted by descending frequency.

## Component 3: CaseQueryGenerator

**File:** `src/rag/adapters/query_generation/case_query_generator.py`

```python
@dataclass(frozen=True, slots=True)
class CaseQueryGenerator:
    term_mapper: TermMapper
    max_queries_per_case: int = 20

    def generate_from_case(self, case_path: Path) -> list[dict]: ...
```

### Strategy 1: Direct Citation Queries

For each unique regulation in frontmatter `cross_references`, generate one query.

Templates (deterministic selection via index modulo):
- `"What are the requirements of {citation}?"`
- `"What does {citation} require?"`
- `"Summarize the key provisions of {citation}."`

Fields: `difficulty: "easy"`, `query_type: "factual"`, `requires_synthesis: false`, tags `["case-derived", "citation-direct"]`.

### Strategy 2: Term Mapping Queries

Run `term_mapper.scan_content(content)`, take top 5 terms with frequency >= 2.

Templates (deterministic selection):
- `"What are the regulatory requirements for {term}?"`
- `"What regulations govern {term} at nuclear power plants?"`
- `"What does the NRC require regarding {term}?"`

Fields: `difficulty: "medium"`, `query_type: "interpretive"`, `requires_synthesis: true`, tags `["case-derived", "term-mapping"]`, `technical_term` field records the matched term.

**Overlap tagging:** If any of a term's mapped citations appear in frontmatter `cross_references`, add `"overlaps_direct_citation": true` and tag `"overlaps-citation"`.

**Relevant citations:** Set to the dictionary's mapped citations (section-level approximations, not subsection-level).

### Shared Fields

All queries include:
- `source_case`: accession number from frontmatter
- `is_unanswerable: false`
- `expected_answer: null` (filled during manual curation)
- `metadata.filter`: `{"type": "Eq", "field": "corpus", "value": "regulatory"}`

QID format: `case-{strategy_prefix}-{counter:03d}` — counters are per-strategy-prefix, globally unique across a generation run.

## Component 4: CLI Script

**File:** `scripts/generate_case_queries.py`

```
./scripts/py scripts/generate_case_queries.py \
  --case-dir corpus/us-nrc/cases/ \
  --output eval/datasets/case_generated_queries_DRAFT.jsonl \
  --term-map config/case_regulatory_terms.json \
  --max-queries-per-case 20
```

Steps:
1. Load `TermMapper` from JSON
2. Construct `CaseQueryGenerator`
3. Glob `**/*.md` under case-dir
4. Generate queries per case, collect results
5. Assign globally unique QIDs
6. Write JSONL output
7. Print summary to stderr (total, by strategy, overlap count, zero-match cases)

## File Layout

```
config/
  case_regulatory_terms.json

src/rag/adapters/query_generation/
  __init__.py
  term_mapper.py
  case_query_generator.py

scripts/
  generate_case_queries.py

tests/adapters/query_generation/
  __init__.py
  test_term_mapper.py
  test_case_query_generator.py
```

## Testing

- **test_term_mapper.py**: `from_json` validation, `lookup` hits/misses, `scan_content` case-insensitive matching, frequency counting, sort order, frequency >= 2 threshold.
- **test_case_query_generator.py**: Synthetic fixture case markdown. Strategy 1 produces one query per `cross_references` entry, Strategy 2 for matched terms, overlap tagging, QID uniqueness, schema completeness, `max_queries_per_case` truncation.

Unit tests only — no integration test against real corpus (CLI run serves that purpose).

## Not In Scope

- No changes to `settings.toml`, `container.py`, or runtime RAG code
- Strategies 3-6 (violation context, adversarial, facility-specific, abstention)
- LLM-assisted query generation
- Validation script (schema compliance enforced by generator)
- Expected answer population (manual curation step)
