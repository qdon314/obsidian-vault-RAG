# NRC Benchmark Generation Pipeline — Revised Design

_Revised from `docs/specs/nrc-benchmark-generation.md` following architectural review._

---

## Design principles

### 1. Evidence precedes question

Do not start from "generate a question from this chunk."
Start from a **structured regulatory unit** with clear provenance.

### 2. Narrow beats broad

Prefer queries answerable from a small, local evidence set over "summarize this section" prompts.

### 3. Retrieval and answer eval are separate products

One dataset feeds both, but the schema must distinguish:

- evidence labels for retrieval
- gold answer / rubric for generation

### 4. Human review is selective, not universal

Lightweight sanity gates at high-leverage stages, not universal lawyer review.

### 5. Query classes drive scoring

Different query classes imply different expectations and metrics.

### 6. Span-first truth

Derive chunk truth from spans, not the other way around.
Regulatory truth must not depend on today's chunk boundaries.

---

## Package architecture

The benchmark pipeline is a **standalone package** at `src/benchmark/`.
It imports `rag.domain` types (frozen dataclasses) but never touches `rag.adapters` or `rag.ports`.
All stages are protocol-based with swappable adapters, mirroring the RAG pipeline's hexagonal structure.

```
src/benchmark/
├── domain/
│   ├── models.py          # BenchmarkSourceSpan, RegulatoryUnit, EvidenceSet,
│   │                      #   QueryCandidate, ValidatedQuery, GoldAnswer,
│   │                      #   BenchmarkDataset — all frozen dataclasses
│   └── enums.py           # QueryClass, UnitKind, ReviewStatus, EvidenceTier
├── ports/
│   ├── unit_extractor.py
│   ├── evidence_builder.py
│   ├── query_generator.py
│   ├── query_validator.py
│   ├── gold_answer_synthesizer.py
│   └── exporter.py
├── adapters/
│   ├── extraction/
│   │   ├── rules_extractor.py      # wraps ecfr_parser.ParsedParagraph (Stage 1a)
│   │   └── llm_extractor.py        # semantic classification pass (Stage 1b)
│   ├── generation/
│   │   ├── template_generator.py
│   │   └── llm_generator.py
│   ├── validation/
│   │   └── llm_validator.py
│   └── export/
│       └── jsonl_exporter.py
├── pipeline/
│   └── runner.py          # orchestrates stages; takes all ports as constructor args
└── scripts/
    └── run_benchmark_gen.py
```

Port contracts:

```python
class UnitExtractor(Protocol):
    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]: ...

class EvidenceBuilder(Protocol):
    def build(self, unit: RegulatoryUnit) -> EvidenceSet: ...

class QueryGenerator(Protocol):
    def generate(self, unit: RegulatoryUnit, query_class: QueryClass) -> list[QueryCandidate]: ...

class QueryValidator(Protocol):
    def validate(self, candidate: QueryCandidate) -> ValidationResult: ...

class GoldAnswerSynthesizer(Protocol):
    def synthesize(self, query: ValidatedQuery) -> GoldAnswer: ...

class BenchmarkExporter(Protocol):
    def export(self, dataset: BenchmarkDataset) -> None: ...
```

---

## Pipeline stages

### Stage 0: Corpus normalization (source view)

Read from the existing ingested corpus and build a benchmark-friendly source view.
This is a plain builder function, not a swappable port — there is one sensible implementation.

Output record: `BenchmarkSourceSpan`

- `source_doc_id`
- `citation`
- `citation_key`
- `section_title`
- `text`
- `char_start`, `char_end`
- `chunk_ids_overlapping_span`
- `parent_section_id`
- `effective_date`
- `corpus_snapshot_id` — hash or date-pinned label for the corpus version
- `metadata`

---

### Stage 1a: Structural segmentation (deterministic)

**Adapter: `RulesExtractor`**

Consumes `ParsedParagraph` objects from the existing `ecfr_parser.py`.
Each `(section, subsection_chain)` pair becomes a `BenchmarkSourceSpan`.
Provenance is structurally derived — citation, char_start, char_end are never guessed.

`unit_id` is minted here from the subsection chain (e.g. `50.46_b_1_peak_cladding_temp`)
and is **immutable** from this point forward. Unit identity must not depend on LLM output.

Cross-reference detection happens here, not in 1b.
When a paragraph references `§ 50.55a` or an incorporated standard (IEEE, ASME),
a `UnitKind.cross_reference` record is emitted with a `target_citation` field.
The eCFR XML cross-reference tags are machine-readable; use them.

---

### Stage 1b: Semantic classification (LLM)

**Adapter: `LLMExtractor`**

Receives the span text and subsection chain from Stage 1a.
Outputs `UnitKind` and structured fields only — it cannot alter span boundaries or unit_id.

Fields classified in this pass:

- `kind`: obligation / prohibition / threshold / exception / condition / definition / process / cross_reference
- `canonical_statement`: normalized single-sentence statement of the regulatory fact
- `entities`: named regulatory entities in the unit
- `value`: numeric threshold if present
- `conditions`: list of qualifying conditions

Low-confidence outputs are flagged (not silently dropped) and queued for human review.

---

### Stage 2: Evidence set construction

For each `RegulatoryUnit`, build tiered evidence using the formal graduation rule:

- **Critical**: removing this span makes the query unanswerable from remaining evidence
- **Supporting**: removing this span degrades completeness but does not make it unanswerable
- **Contextual**: nearby material that may help but is neither critical nor supporting

Each span maps to: source doc/span IDs, overlapping chunk IDs, section-level metadata.

This directly fixes the denominator explosion problem: recall becomes meaningful only when
the relevance set is tight. Median critical evidence target: ≤ 2 spans, ≤ 4 chunk IDs.

---

### Stage 3: Query class generation

Generate questions from regulatory units using class-specific templates plus controlled paraphrasing.

#### A. Citation lookup

Best for retrieval precision.

- "What is the maximum peak cladding temperature allowed under ECCS criteria?"
- Outputs: critical spans, short gold answer, acceptable citation forms

#### B. Narrow factual lookup

Answerable from one or a few units.

- "What temperature limit applies to peak cladding temperature?"
- Outputs: critical spans, short gold answer

#### C. Rule explanation

Slightly broader but still bounded.

- "What does 10 CFR 50.55a require with respect to ASME code compliance?"
- Outputs: critical + supporting spans, answer rubric with required points

#### D. Cross-reference / dependency

Tests retrieval across linked provisions.

- "How does provision A interact with the exception in provision B?"
- Outputs: multiple critical spans, answer rubric

#### E. Scenario application

Use sparingly. Prefer operational/procedural scenarios over pure fact lookup —
these are less likely to be answered from model weights rather than retrieved context.

- "Would a design violate the regulation if X occurred under Y condition?"
- Outputs: scenario facts, expected reasoning points, gold disposition if clear

#### F. Unanswerable

**Required class — safety-critical for a nuclear regulatory corpus.**

Queries where the correct retrieval result is empty: out-of-scope questions,
superseded provisions, adjacent-domain questions (OSHA, EPA) not covered by the corpus.

- Outputs: `expected_retrieval: "empty"`, `gold_answer: null`, `is_unanswerable: true`
- Target: 5–10% of the retrieval-core set

#### G. Robustness variants

Derived from A–D, not standalone first-class generation.
Variants: paraphrase, shorthand, typo, citation-only query.
Inherit the same evidence labels and `robustness_parent_qid` as the source query.

---

### Stage 4: Controlled query authoring

Use a structured prompt that includes:

- the extracted regulatory unit
- permitted query class
- target difficulty
- forbidden patterns
- answerability constraints
- required evidence locality

**Preferred generation objective:**
"Produce a realistic user question that is answerable primarily from the listed critical evidence."

**Hard constraints for the generator — reject outputs that:**

- ask for full-section summaries
- require broad synthesis across large regions unless explicitly classed as synthesis
- contain malformed citations
- duplicate prior queries semantically
- ask vague "what are the rules about X" questions without bounded scope

---

### Stage 5: Auto-validation and filtering

#### Deterministic checks

- citation format valid
- no duplicate normalized query text
- no banned broad verbs unless class allows it
- query length within bounds
- evidence set non-empty
- evidence set size below threshold for narrow classes
- chunk overlap resolves successfully
- `corpus_snapshot_id` and `valid_as_of` populated

#### Model-based checks

Score each candidate on:

- **plausibility**: would a real user ask this?
- **boundedness**: is this answerable from the supplied evidence? (uses formal graduation rule from Stage 2)
- **ambiguity**: does the question have multiple materially different readings?
- **specificity**: is the target clear?
- **leakage**: does the query mirror statutory phrasing too directly?

Low-score items are flagged for revision, not silently dropped.

#### Hard negative mining

After initial validation, run the live retriever against each query.
Top-k results that are not in the evidence set become `hard_negative_chunk_ids` (minimum 2 per query).
These are required for valid reranker eval — without them, reranker eval measures easy separation, not real discrimination.

#### Contamination probe

For each query, run the generator with an **empty context window**.
If it produces the correct answer, flag the query as `llm_contamination_risk: true`.

- Contamination-flagged queries are still valid for retrieval eval
- They are **excluded** from the answer-core set
- Prefer scenario / cross-reference / operational queries for answer-core to minimize contamination risk

---

### Stage 6: Gold answer / rubric creation

#### Retrieval-core set

For most queries, store:

- critical / supporting / contextual evidence tiers
- hard negative chunk IDs
- optional short answer
- no correctness score required

#### Answer-core set

For a smaller curated subset (zero contamination-flagged queries), add:

- `gold_answer`
- `acceptable_answer_variants`
- `required_points`
- `forbidden_errors`
- `is_unanswerable` / `unanswerable_reason` where applicable

---

### Stage 7: Human review workflow

#### Pass 1: dataset editor review

Review for: realism, boundedness, duplicates, malformed outputs.

#### Pass 2: spot audit

Review 10–20% of accepted queries by class.

#### Pass 3: answer-core promotion

Only promote queries to answer-core after reviewing:

- evidence sufficiency
- answer correctness / rubric quality
- contamination probe result

---

## Dataset schema

```json
{
  "qid": "reg_fact_000123",
  "query": "What is the maximum peak cladding temperature allowed by the ECCS acceptance criteria?",
  "query_class": "citation_lookup",
  "difficulty": "easy",
  "source_unit_ids": ["50.46_b_1_peak_cladding_temp"],
  "source_citations": ["10 CFR 50.46(b)(1)"],
  "critical_evidence": [
    {
      "span_id": "span_abc",
      "citation": "10 CFR 50.46(b)(1)",
      "char_start": 1204,
      "char_end": 1292,
      "chunk_ids": ["chunk_17", "chunk_18"]
    }
  ],
  "supporting_evidence": [],
  "contextual_evidence": [],
  "hard_negative_chunk_ids": ["chunk_31", "chunk_44"],
  "gold_answer": "The peak cladding temperature must not exceed 2200°F.",
  "acceptable_answer_variants": [
    "2200°F maximum peak cladding temperature",
    "No more than 2200 degrees Fahrenheit"
  ],
  "required_points": [
    "Contains the 2200°F threshold",
    "Associates it with peak cladding temperature"
  ],
  "forbidden_errors": [
    "Wrong threshold value",
    "Attributing the threshold to the wrong metric"
  ],
  "is_unanswerable": false,
  "unanswerable_reason": null,
  "robustness_parent_qid": null,
  "llm_contamination_risk": false,
  "corpus_snapshot_id": "ecfr_2026-01-01",
  "valid_as_of": "2026-01-01",
  "metadata": {
    "generator_version": "qgen_v1",
    "validator_version": "qval_v1",
    "review_status": "approved"
  }
}
```

---

## Benchmark products

### `reg_retrieval_core`

Purpose: retriever/reranker tuning.

Contains: citation lookup, narrow factual, some cross-reference, unanswerable class.
All queries have tiered evidence and hard negatives populated.
Gold answers optional.

### `reg_answer_core`

Purpose: end-to-end answer evaluation.

Curated subset of retrieval-core. Zero contamination-flagged queries.
All queries have gold answers, rubric points, and unanswerable labels where needed.

### `reg_robustness_core`

Purpose: robustness under paraphrase, shorthand, typos.
Derived from approved retrieval-core queries only.

---

## Acceptance criteria for v1

### Target size

| Dataset | Count | Constraint |
|---|---|---|
| `reg_retrieval_core` | 50–75 queries | ≥5 unanswerable class |
| `reg_answer_core` | 20–30 queries | 0 contamination-flagged |
| `reg_robustness_core` | 10–20 variants | — |

### Quality bar

At least 90% of reviewed items satisfy:

- realistic phrasing
- bounded evidence
- no malformed citations
- no semantic duplicates
- clear class assignment

### Structural bar

For citation lookup and factual classes:

- median critical evidence size ≤ 2 spans
- median overlapping chunk count for critical evidence ≤ 4 chunks
- `hard_negative_chunk_ids` non-empty for all retrieval-core queries
- `corpus_snapshot_id` and `valid_as_of` populated on every record

---

## Milestones

```
M1: Stage 0 source view + Stage 1a structural segmentation
    Validates ecfr_parser bridge, mints stable unit_ids.
    No LLM involvement yet.

M2: Stage 1b LLM semantic classification + Stage 2 evidence builder
    First RegulatoryUnit records with tiered evidence.
    Validate unit extraction quality before investing in generation.

M3: Stage 3+4 citation-lookup generator + Stage 5 deterministic validator
    First QueryCandidates through the validation gate.

M4: LLM validator + hard negative mining + unanswerable class generation
    Retrieval-core v1 complete.

M5: Contamination probe + answer-core promotion
    Answer-core v1 complete.

M6: Robustness variants
    Full v1 benchmark shipped.
```

Later additions (post-v1): narrow factual, cross-reference, scenario classes.
