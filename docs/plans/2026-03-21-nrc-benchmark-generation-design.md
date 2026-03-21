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

## Eval framework integration

> **ADR needed:** `docs/decisions/adr-benchmark-eval-schema-boundary.md` — formalize the
> relationship between the benchmark domain (`src/benchmark/domain/`) and the eval
> framework (`src/rag/eval/schema.py`).

The benchmark pipeline maintains its own richer domain schema (tiered evidence, rubrics,
contamination probes, regulatory unit provenance). It does **not** extend or modify
`EvalQuery` from `src/rag/eval/schema.py`.

Instead, `BenchmarkExporter` (the exporter port) is responsible for emitting
`EvalQuery`-compatible JSONL as its **primary output format**. This ensures:

- The existing eval harness, metrics, judges, and Streamlit app work unchanged.
- The benchmark domain retains full fidelity for benchmark-specific analysis.
- No cross-layer coupling between the benchmark package and eval internals.

The exporter maps benchmark fields to eval fields as follows:

| Benchmark field | EvalQuery field |
|---|---|
| `qid` | `qid` |
| `query` | `query` |
| `critical_evidence[*].chunk_ids` | `relevant_chunk_ids` |
| `source_citations` | `relevant_citations` |
| `query_class` | `query_type` (mapped via enum translation) |
| `difficulty` | `difficulty` |

Fields without an `EvalQuery` counterpart (tiered evidence, rubric, contamination flags)
are preserved only in the full benchmark JSONL export.

---

## Package architecture

The benchmark pipeline is a **standalone package** at `src/benchmark/`.
It imports `rag.domain` types (frozen dataclasses) but never touches `rag.adapters` or `rag.ports`.
All stages are protocol-based with swappable adapters, mirroring the RAG pipeline's hexagonal structure.

**Exception:** Stage 5b (hard negative mining) intentionally crosses this boundary by
accepting a `Retriever` port from the RAG pipeline. See Stage 5b below.

```
src/benchmark/
├── domain/
│   ├── models.py          # BenchmarkSourceSpan, RegulatoryUnit, EvidenceSet,
│   │                      #   QueryCandidate, ValidatedQuery, GoldAnswer,
│   │                      #   BenchmarkDataset, StageConfig — all frozen dataclasses
│   ├── enums.py           # QueryClass, UnitKind, ReviewStatus, EvidenceTier
│   └── snapshot.py        # compute_snapshot_id(), verify_snapshot()
├── ports/
│   ├── unit_extractor.py
│   ├── evidence_builder.py
│   ├── query_generator.py
│   ├── query_validator.py
│   ├── gold_answer_synthesizer.py
│   ├── llm_client.py       # LLMClient protocol for all LLM calls
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
│       ├── jsonl_exporter.py        # full benchmark JSONL
│       └── eval_query_exporter.py   # EvalQuery-compatible JSONL
├── pipeline/
│   └── runner.py          # orchestrates stages; takes all ports as constructor args
└── scripts/
    └── run_benchmark_gen.py
```

Port contracts:

```python
class LLMClient(Protocol):
    """All LLM calls in the benchmark pipeline route through this port."""
    def complete(self, prompt: str, config: StageConfig) -> str: ...

class UnitExtractor(Protocol):
    def extract(self, spans: list[BenchmarkSourceSpan]) -> list[RegulatoryUnit]: ...

class EvidenceBuilder(Protocol):
    def build(self, unit: RegulatoryUnit) -> EvidenceSet: ...

class QueryGenerator(Protocol):
    def generate(self, unit: RegulatoryUnit, query_class: QueryClass) -> list[QueryCandidate]: ...

class QueryValidator(Protocol):
    def validate(self, candidate: QueryCandidate) -> ValidationResult: ...
    def refine_evidence(self, query: ValidatedQuery, evidence: EvidenceSet) -> EvidenceSet: ...

class GoldAnswerSynthesizer(Protocol):
    def synthesize(self, query: ValidatedQuery) -> GoldAnswer: ...

class BenchmarkExporter(Protocol):
    def export(self, dataset: BenchmarkDataset) -> None: ...
```

### `StageConfig` dataclass

Each pipeline stage receives its own `StageConfig`:

```python
@dataclass(frozen=True, slots=True)
class StageConfig:
    model: str                    # e.g., "gpt-4o", "claude-sonnet-4-20250514"
    temperature: float = 0.0     # deterministic by default for reproducibility
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_s: float = 60.0
```

---

## Pipeline execution model

> **Documentation needed:** `docs/operations/benchmark-pipeline-runbook.md` — operational
> guide for running the benchmark pipeline, including checkpointing, resume, and cost
> estimation.

### Checkpointing and resume

The runner writes intermediate JSONL after each stage completes:

```
benchmark_runs/<run_id>/
├── stage_0_spans.jsonl
├── stage_1a_units.jsonl
├── stage_1b_classified.jsonl
├── stage_2_evidence.jsonl
├── stage_3_candidates.jsonl
├── stage_5a_validated.jsonl
├── stage_5b_hard_negatives.jsonl
├── stage_5c_contamination.jsonl
├── stage_6_gold_answers.jsonl
└── run_config.json          # full pipeline config for reproducibility
```

The runner supports `--resume-from=stage_N` to skip completed stages, reading from
the checkpoint JSONL for prior stages.

### LLM call routing

All LLM calls route through the `LLMClient` protocol. This ensures:

- Model backend is swappable (OpenAI, Anthropic, local).
- Retry and timeout policies are centralized.
- Cost tracking can be implemented at the port level.

### Performance expectations

For a v1 corpus of ~100 eCFR sections producing 50–75 queries:

- Stages 0, 1a: seconds (deterministic, no LLM).
- Stage 1b: ~5 minutes (one LLM call per paragraph, batchable).
- Stages 3–5: ~10–15 minutes (one LLM call per unit per class, plus validation).
- Stage 5b: ~2 minutes (retriever calls, no LLM).
- Stage 5c: ~5 minutes (one LLM call per query).

LLM-heavy stages support batch parallelism via `asyncio.gather()` or
`concurrent.futures.ThreadPoolExecutor`. Distributed infrastructure is not needed
at v1 scale.

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
- `parent_section_id` — the `ParsedSection.section_number` from the eCFR parser (e.g., `"50.46"`), used for grouping spans by regulatory section
- `effective_date` — section-level effective date from the eCFR API; falls back to the corpus fetch date if section-level dates are unavailable
- `corpus_snapshot_id` — SHA-256 of sorted `(doc_id, content_hash)` pairs for all documents in the corpus
- `metadata`

#### Corpus snapshot mechanism

`corpus_snapshot_id` is a content-addressable hash computed by `compute_snapshot_id()`:

```python
def compute_snapshot_id(corpus: Sequence[Document]) -> str:
    """SHA-256 of sorted (doc_id, content_hash) pairs."""
    pairs = sorted((doc.doc_id, doc.content_hash) for doc in corpus)
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()

def verify_snapshot(corpus: Sequence[Document], expected_id: str) -> bool:
    """Confirm the current corpus matches the claimed snapshot."""
    return compute_snapshot_id(corpus) == expected_id
```

This is deterministic, requires no external tooling, and catches any corpus drift.
The runner calls `verify_snapshot()` at pipeline start and aborts if the corpus has
changed since the snapshot was taken.

---

### Stage 1a: Structural segmentation (deterministic)

**Adapter: `RulesExtractor`**

**Prerequisite:** `ecfr_parser.py` must expose cross-reference tags. See M0 prerequisite below.

Consumes `ParsedParagraph` objects from the existing `ecfr_parser.py`.
Each `(section, subsection_chain)` pair becomes a `BenchmarkSourceSpan`.
Provenance is structurally derived — citation, char_start, char_end are never guessed.

`unit_id` is minted here from the subsection chain (e.g. `50.46_b_1_peak_cladding_temp`)
and is **immutable** from this point forward. Unit identity must not depend on LLM output.

Cross-reference detection happens here, not in 1b.
When a paragraph references `§ 50.55a` or an incorporated standard (IEEE, ASME),
a `UnitKind.cross_reference` record is emitted with a `target_citation` field.
The eCFR XML cross-reference tags (`XREF`, `AREF`) are machine-readable; use them.

> **Prerequisite task (M0):** Audit `ecfr_parser.py` for `XREF`/`AREF` tag handling.
> If not present, extend `ParsedParagraph` with a `cross_references: tuple[CrossRef, ...]`
> field. This must be completed before M1.

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

> **ADR needed:** `docs/decisions/adr-evidence-tier-semantics.md` — document the decision
> that evidence tiers are unit-relative (not query-relative) and the rationale for
> post-generation refinement via `QueryValidator.refine_evidence()`.

For each `RegulatoryUnit`, build **unit-relative** tiered evidence. Tier definitions:

- **Critical**: removing this span makes the regulatory unit's normative content incomprehensible
- **Supporting**: removing this span degrades completeness of understanding but the core obligation/threshold remains clear
- **Contextual**: nearby material that may help interpretation but is neither critical nor supporting

These tiers reflect the unit's intrinsic evidence structure, not any specific query's
retrieval needs. This is intentional: evidence is built before queries exist, and a single
unit may feed multiple query classes.

**Post-generation refinement:** After queries are generated (Stage 3) and validated (Stage 5a),
`QueryValidator.refine_evidence()` may narrow the unit-level evidence set for a specific
query. For example, a citation-lookup query may only need the critical tier, while a
cross-reference query may promote contextual spans from a linked unit to supporting.

Each span maps to: source doc/span IDs, overlapping chunk IDs, section-level metadata.

This directly fixes the denominator explosion problem: recall becomes meaningful only when
the relevance set is tight. Median critical evidence target: <= 2 spans, <= 4 chunk IDs.

---

### Stage 3: Query class generation

Generate questions from regulatory units using class-specific templates plus controlled paraphrasing.

`QueryClass` enum values use `snake_case` string representations (e.g., `citation_lookup`,
`narrow_factual`). These are intentionally distinct from the `QueryType` enum in
`src/rag/eval/schema.py`; the exporter handles translation.

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
- Target: 5-10% of the retrieval-core set

**Generation strategies for unanswerable queries:**

1. **Near-miss**: take a real unit and ask about a related but uncovered subsection (e.g., ask about 10 CFR 50.46(b)(6) when only (b)(1)-(5) exist).
2. **Domain boundary**: use a curated seed list of adjacent-domain topics (OSHA 29 CFR 1910, EPA 40 CFR, DOE 10 CFR 830) as question subjects.
3. **Fabricated citation**: generate a plausible but non-existent CFR reference (e.g., "10 CFR 50.48(f)" when 50.48 only goes to (e)).
4. **Temporal (post-v1)**: ask about provisions that were amended or removed, using historical eCFR data. Deferred — requires historical corpus access.

#### G. Robustness variants

Derived from A-D, not standalone first-class generation.
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

### Stage 5a: Auto-validation and filtering

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

#### Evidence refinement

After validation, call `QueryValidator.refine_evidence()` to narrow unit-level evidence
for each specific query. This produces per-query tier assignments from the unit-level
evidence set.

---

### Stage 5b: Hard negative mining (optional, crosses RAG boundary)

> **ADR needed:** `docs/decisions/adr-benchmark-rag-boundary-crossing.md` — document why
> hard negative mining requires a live `Retriever` port and the implications for the
> standalone package boundary.

This stage **intentionally crosses** the standalone package boundary by accepting a
`Retriever` port from the RAG pipeline. It is optional — the runner skips it if no
retriever is provided.

Run the live retriever against each query. Top-k results not in the evidence set
become `hard_negative_chunk_ids` (minimum 2 per query). These are required for valid
reranker eval — without them, reranker eval measures easy separation, not real
discrimination.

The schema records the retriever configuration used:

- `hard_negatives_retriever_config`: model name, index version, top_k used
- If the retriever changes (new embedding model, re-indexed corpus), hard negatives
  become stale. The `hard_negatives_retriever_config` field makes staleness detectable.

**Fallback when < 2 hard negatives found:** flag the query with
`hard_negatives_insufficient: true` for manual curation. Do not relax the minimum —
the reranker eval value depends on having genuine distractors.

---

### Stage 5c: Contamination probe

> **Documentation needed:** `docs/operations/contamination-probe-runbook.md` — guide for
> re-running contamination probes when the production generator model changes.

For each query, run the production generator model (as specified in `StageConfig`) with
an **empty context window**. The model is the same one used in the production RAG
pipeline's `Generator` port.

**Match criterion:** the existing `evaluate_vs_expected_answer` judge from
`src/rag/eval/judges.py` scores the model's ungrounded answer >= 0.7 against the gold
answer. This reuses proven evaluation infrastructure rather than inventing a new metric.

Contamination is **model-version-specific**. The schema stores per-model probe results:

```json
"contamination_probes": {
  "gpt-4o-2025-01-01": false,
  "claude-sonnet-4-20250514": true
}
```

- Queries flagged as contaminated for the current production model are excluded from the answer-core set.
- Queries flagged as contaminated are still valid for retrieval eval.
- Prefer scenario / cross-reference / operational queries for answer-core to minimize contamination risk.
- **Probes must be re-run** when the production generator model changes. The runner
  warns if the current model has no probe results.

---

### Stage 6: Gold answer / rubric creation

#### Retrieval-core set

For most queries, store:

- critical / supporting / contextual evidence tiers
- hard negative chunk IDs
- optional short answer
- no correctness score required

#### Answer-core set

For a smaller curated subset (zero contamination-flagged queries for the current model), add:

- `gold_answer`
- `acceptable_answer_variants`
- `required_points`
- `forbidden_errors`
- `is_unanswerable` / `unanswerable_reason` where applicable

---

### Stage 7: Human review workflow

> **Documentation needed:** `docs/operations/benchmark-review-guide.md` — reviewer
> instructions, acceptance criteria per query class, and examples of common rejection
> reasons.

#### Tooling

For v1, reviewers work directly in JSONL files. The dataset is small (50-75 queries) and
does not warrant dedicated review UI. A Streamlit review interface extending `eval/app_v2/`
is a candidate for v2.

#### Review state machine

The `ReviewStatus` enum tracks each query's review state in `metadata.review_status`:

```
pending -> approved
pending -> rejected
pending -> needs_revision -> pending (re-enters review)
```

Review decisions are recorded in-place in the JSONL with reviewer identity and timestamp
in `metadata.reviewed_by` and `metadata.reviewed_at`. This provides an audit trail,
which is important for a nuclear regulatory context.

#### Pass 1: dataset editor review

Review for: realism, boundedness, duplicates, malformed outputs.

#### Pass 2: spot audit

Review 10-20% of accepted queries by class.

#### Pass 3: answer-core promotion

Only promote queries to answer-core after reviewing:

- evidence sufficiency
- answer correctness / rubric quality
- contamination probe result

---

## Dataset schema

> **ADR needed:** `docs/decisions/adr-benchmark-schema-versioning.md` — define the
> compatibility policy for schema evolution (additive minor versions, breaking major
> versions with migration scripts).

```json
{
  "schema_version": "1.0",
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
  "hard_negatives_retriever_config": {
    "model": "text-embedding-3-small",
    "index_version": "ecfr_2026-01-01",
    "top_k": 20
  },
  "gold_answer": "The peak cladding temperature must not exceed 2200\u00b0F.",
  "acceptable_answer_variants": [
    "2200\u00b0F maximum peak cladding temperature",
    "No more than 2200 degrees Fahrenheit"
  ],
  "required_points": [
    "Contains the 2200\u00b0F threshold",
    "Associates it with peak cladding temperature"
  ],
  "forbidden_errors": [
    "Wrong threshold value",
    "Attributing the threshold to the wrong metric"
  ],
  "is_unanswerable": false,
  "unanswerable_reason": null,
  "robustness_parent_qid": null,
  "contamination_probes": {
    "gpt-4o-2025-01-01": false
  },
  "corpus_snapshot_id": "a1b2c3d4e5f6...",
  "valid_as_of": "2026-01-01",
  "metadata": {
    "generator_version": "qgen_v1",
    "validator_version": "qval_v1",
    "review_status": "pending",
    "reviewed_by": null,
    "reviewed_at": null
  }
}
```

### Schema versioning policy

- **Minor versions** (1.0 -> 1.1): additive fields only, backward compatible. Consumers must tolerate missing optional fields.
- **Major versions** (1.x -> 2.0): breaking changes. A migration script at `src/benchmark/scripts/migrate_schema.py` is required for each major bump.

---

## Data handling

> **Documentation needed:** If the benchmark dataset is ever published or shared externally,
> create `docs/operations/benchmark-data-policy.md` covering distribution terms and
> provenance attribution.

- NRC regulations (10 CFR) are public domain. Source material contains no PII or non-public information.
- Generated queries and gold answers are synthetic artifacts derived from public regulations. They contain no non-public operational information.
- LLM API calls transmit public regulatory text. Use zero-retention API endpoints where available (e.g., OpenAI's zero-data-retention policy for API usage).
- The benchmark dataset is intended for **internal evaluation use**. If external distribution is planned, a separate data policy review is required.

---

## Benchmark products

### `reg_retrieval_core`

Purpose: retriever/reranker tuning.

Contains (v1): citation lookup and unanswerable class only.
Planned for v1.1: narrow factual and cross-reference classes.
All queries have tiered evidence and hard negatives populated.
Gold answers optional.

### `reg_answer_core`

Purpose: end-to-end answer evaluation.

Curated subset of retrieval-core. Zero contamination-flagged queries (for the current production model).
All queries have gold answers, rubric points, and unanswerable labels where needed.

### `reg_robustness_core`

Purpose: robustness under paraphrase, shorthand, typos.
Derived from approved retrieval-core queries only.

---

## Acceptance criteria for v1

### Target size

| Dataset | Count | Constraint |
|---|---|---|
| `reg_retrieval_core` | 50-75 queries | >= 5 unanswerable class; citation lookup + unanswerable only |
| `reg_answer_core` | 20-30 queries | 0 contamination-flagged (current model) |
| `reg_robustness_core` | 10-20 variants | -- |

### Quality bar

At least 90% of reviewed items satisfy:

- realistic phrasing
- bounded evidence
- no malformed citations
- no semantic duplicates
- clear class assignment

### Structural bar

For citation lookup and factual classes:

- median critical evidence size <= 2 spans
- median overlapping chunk count for critical evidence <= 4 chunks
- `hard_negative_chunk_ids` non-empty for all retrieval-core queries
- `corpus_snapshot_id` and `valid_as_of` populated on every record
- `schema_version` populated on every record

---

## Milestones

```
M0 (prerequisite): ecfr_parser cross-reference tag support
    Audit ecfr_parser.py for XREF/AREF tag handling.
    If missing, extend ParsedParagraph with cross_references field.
    Gate: ParsedParagraph emits cross-reference data for known sections.

M1: Stage 0 source view + Stage 1a structural segmentation
    Validates ecfr_parser bridge, mints stable unit_ids.
    Includes compute_snapshot_id() and verify_snapshot() utilities.
    No LLM involvement yet.

M2: Stage 1b LLM semantic classification + Stage 2 evidence builder
    First RegulatoryUnit records with unit-relative tiered evidence.
    Validate unit extraction quality before investing in generation.

M3: Stage 3+4 citation-lookup generator + Stage 5a deterministic validator
    First QueryCandidates through the validation gate.
    Checkpoint/resume infrastructure operational.

M4: LLM validator + Stage 5b hard negative mining + unanswerable class generation
    Retrieval-core v1 complete.
    EvalQuery-compatible export validated against eval harness.

M5: Stage 5c contamination probe + answer-core promotion
    Answer-core v1 complete.

M6: Robustness variants
    Full v1 benchmark shipped.
```

Later additions (post-v1): narrow factual, cross-reference, scenario classes;
Streamlit review UI; temporal unanswerable queries from historical eCFR data.

---

## Required ADRs and documentation

> This section tracks documentation artifacts that must be created alongside
> implementation. Each item is also called out inline at the relevant design section.

### Architecture Decision Records

| ADR | Covers | Create by milestone |
|---|---|---|
| `docs/decisions/adr-benchmark-eval-schema-boundary.md` | Benchmark domain vs. `EvalQuery` relationship; exporter-based integration | M1 |
| `docs/decisions/adr-evidence-tier-semantics.md` | Unit-relative (not query-relative) evidence tiers; post-generation refinement via `refine_evidence()` | M2 |
| `docs/decisions/adr-benchmark-rag-boundary-crossing.md` | Hard negative mining requires live `Retriever`; standalone boundary exception | M4 |
| `docs/decisions/adr-benchmark-schema-versioning.md` | Schema compatibility policy; minor (additive) vs. major (breaking + migration) versions | M1 |

### Operational documentation

| Document | Covers | Create by milestone |
|---|---|---|
| `docs/operations/benchmark-pipeline-runbook.md` | Running the pipeline, checkpointing, resume, cost estimation, troubleshooting | M3 |
| `docs/operations/contamination-probe-runbook.md` | Re-running contamination probes when generator model changes | M5 |
| `docs/operations/benchmark-review-guide.md` | Reviewer instructions, acceptance criteria per query class, rejection examples | M4 |
| `docs/operations/benchmark-data-policy.md` | Data handling, distribution terms, provenance (only if external sharing planned) | Post-v1 |
