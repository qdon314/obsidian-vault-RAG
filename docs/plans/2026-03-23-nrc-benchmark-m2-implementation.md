# NRC Benchmark M2: LLM Semantic Classification + Evidence Builder

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task.

**Goal:** Enrich Stage 1a regulatory units with LLM-classified semantic
fields (Stage 1b) and build unit-relative tiered evidence sets (Stage 2),
producing the first complete `RegulatoryUnit` + `EvidenceSet` records.

**Architecture:** Stage 1b is a new `LLMExtractor` adapter that takes
Stage 1a's `RegulatoryUnit` records and returns enriched copies via
`dataclasses.replace()`. Stage 2 introduces the `EvidenceBuilder` port
and an LLM-based adapter that assigns each span to an evidence tier.
Both stages route all LLM calls through a new `LLMClient` protocol port,
keeping model backend swappable. Domain models (`EvidenceEntry`,
`EvidenceSet`) are added to the existing `models.py`.

**Tech Stack:** Python 3.12, frozen dataclasses, `typing.Protocol`,
`dataclasses.replace()` for immutable updates

**Design doc:** `docs/plans/2026-03-21-nrc-benchmark-generation-design.md`

---

## Rationale

The dependency structure is:

1. **LLMClient port** is the foundation — both Stage 1b and Stage 2
   adapters depend on it for all LLM calls.
2. **Evidence domain models** (`EvidenceEntry`, `EvidenceSet`) must exist
   before the `EvidenceBuilder` port can reference them.
3. **ADR** for evidence tier semantics is independent of code and can
   parallelize with the port and model work.
4. **LLMExtractor** (Stage 1b) depends only on the LLMClient port.
5. **EvidenceBuilder port + adapter** (Stage 2) depends on both the
   LLMClient port and the evidence domain models.
6. **Lint/typecheck pass** validates everything integrates cleanly.

Tasks 1–3 are independent and form the first parallel group. Tasks 4–5
depend on that group. Task 6 depends on everything.

---

### Tasks 1–3 (parallel): Foundation ports, models, and ADR

> These tasks are independent and can be executed in parallel.
> No dependencies on prior M2 work.

### Task 1: LLMClient port

**Why:** Every LLM call in the benchmark pipeline must route through a
single protocol to keep model backend swappable and enable centralized
retry/timeout/cost-tracking. Both Stage 1b and Stage 2 adapters depend
on this port.

**Files:**
- Create: `src/benchmark/ports/llm_client.py`
- Test: `tests/benchmark/ports/test_llm_client.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response from an LLM call."""
    text: str
    model: str
    usage: dict[str, int]  # {"prompt_tokens": ..., "completion_tokens": ...}

class LLMClient(Protocol):
    """All LLM calls in the benchmark pipeline route through this port."""
    def complete(self, prompt: str, config: StageConfig) -> LLMResponse: ...
```

**Acceptance:**
- `LLMClient` is a `typing.Protocol` (structural subtyping, no ABC)
- `LLMResponse` is a frozen dataclass with `slots=True`
- A concrete class with matching `complete` signature passes
  `isinstance` check via `runtime_checkable` or satisfies the protocol
  structurally
- Test: a trivial stub satisfies the protocol (no real LLM call)
- Imports resolve: `from benchmark.ports.llm_client import LLMClient, LLMResponse`

**Constraints:**
- Must use `StageConfig` from `benchmark.domain.models` (already exists)
- Do not import from `rag.adapters` or `rag.ports`
- `LLMResponse.usage` is a plain dict, not a dataclass — keeps it
  backend-agnostic (OpenAI and Anthropic return different shapes)

---

### Task 2: Evidence domain models

**Why:** Stage 2 needs `EvidenceEntry` and `EvidenceSet` types to
represent tiered evidence. These must exist in the domain layer before
the `EvidenceBuilder` port can reference them.

**Files:**
- Modify: `src/benchmark/domain/models.py`
- Test: `tests/benchmark/domain/test_models.py` (extend existing)

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """A single span assigned to an evidence tier."""
    span_id: str          # unique within the evidence set
    citation: str         # e.g. "10 CFR 50.46(b)(1)"
    text: str             # span text for downstream prompt construction
    char_start: int
    char_end: int
    chunk_ids: tuple[str, ...]
    tier: EvidenceTier

@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Unit-relative tiered evidence for a single RegulatoryUnit."""
    unit_id: str
    critical: tuple[EvidenceEntry, ...]
    supporting: tuple[EvidenceEntry, ...]
    contextual: tuple[EvidenceEntry, ...]
```

**Acceptance:**
- Both are frozen dataclasses with `slots=True`
- `EvidenceEntry.tier` uses `EvidenceTier` enum from `benchmark.domain.enums`
- `EvidenceSet` groups entries by tier in named tuples
- Existing `BenchmarkSourceSpan`, `RegulatoryUnit`, `StageConfig` tests
  still pass (no regression)
- Test: round-trip construction, immutability, field access
- `EvidenceSet` with empty tuples for supporting/contextual is valid

**Constraints:**
- Add to existing `models.py` — do not create a separate file
- Import `EvidenceTier` from `benchmark.domain.enums` (already exists)
- `span_id` is minted by the evidence builder (not this task) — the
  model only stores it
- `text` field is included so downstream stages (query generation,
  contamination probe) can build prompts without re-reading spans

---

### Task 3: ADR — Evidence tier semantics

**Why:** The design doc calls for `docs/decisions/adr-evidence-tier-semantics.md`
by M2. Documents the decision that evidence tiers are unit-relative (not
query-relative) and the rationale for post-generation refinement.

**Files:**
- Create: `docs/decisions/adr-evidence-tier-semantics.md`

**Contract:**

ADR must cover:
1. **Context:** evidence tiers need to be assigned before queries exist,
   because a single unit feeds multiple query classes
2. **Decision:** tiers are unit-relative — they describe how important
   each span is to the regulatory unit's normative content, not to any
   specific query
3. **Tier definitions:**
   - Critical: removing this span makes the unit's normative content
     incomprehensible
   - Supporting: removing degrades completeness but core
     obligation/threshold remains clear
   - Contextual: nearby material that aids interpretation
4. **Post-generation refinement:** `QueryValidator.refine_evidence()`
   (future, Stage 5a) may narrow unit-level evidence for a specific
   query after generation
5. **Consequences:** evidence sets are reusable across query classes;
   per-query narrowing is a separate downstream concern

**Acceptance:**
- File exists at the specified path
- Follows the ADR format used by existing ADRs in `docs/decisions/`
- References the design doc sections on Stage 2 and Stage 5a

**Constraints:**
- Check existing ADRs (`adr-benchmark-eval-schema-boundary.md`,
  `adr-benchmark-schema-versioning.md`) for format/style precedent

---

### Task 4: LLMExtractor adapter (Stage 1b semantic classification)

**Why:** Stage 1a assigns a preliminary `UnitKind` (OBLIGATION or
CROSS_REFERENCE) based on structural cues only. Stage 1b uses an LLM
to refine the kind and populate the semantic fields: `canonical_statement`,
`entities`, `value`, `conditions`. This is where regulatory units become
semantically rich enough to drive query generation.

> Depends on: Task 1 (LLMClient port)

**Files:**
- Create: `src/benchmark/adapters/extraction/llm_extractor.py`
- Test: `tests/benchmark/adapters/extraction/test_llm_extractor.py`

**Contract:**

```python
class LLMExtractor:
    """Stage 1b: LLM-based semantic classification of regulatory units.

    Takes Stage 1a ``RegulatoryUnit`` records and returns enriched copies
    with semantic fields populated. Does NOT alter ``unit_id``, ``spans``,
    or span boundaries.
    """

    def __init__(self, llm_client: LLMClient, config: StageConfig) -> None: ...

    def classify(self, units: list[RegulatoryUnit]) -> list[RegulatoryUnit]:
        """Classify each unit and return enriched copies.

        Uses ``dataclasses.replace()`` to produce new instances with
        populated semantic fields. Units where classification confidence
        is below threshold are returned with ``metadata["low_confidence"] = True``.
        """
        ...
```

The LLM prompt for each unit must include:
- The span text (concatenated from `unit.spans[*].text`)
- The subsection chain for structural context
- The full `UnitKind` enum values as valid output categories
- Instruction to return structured JSON with fields: `kind`, `canonical_statement`,
  `entities`, `value`, `conditions`

**Acceptance:**
- `classify()` returns one `RegulatoryUnit` per input unit
- Immutable fields are preserved: `unit_id`, `spans`, `citation`,
  `subsection_chain`, `parent_section_id`, `corpus_snapshot_id`,
  `cross_references` must be identical between input and output
- `kind` is updated from the LLM response (may differ from Stage 1a's
  preliminary assignment)
- `canonical_statement` is populated (non-None) for all units where
  classification succeeded
- Low-confidence outputs get `metadata["low_confidence"] = True` —
  they are NOT dropped
- Test with a mock `LLMClient` that returns canned JSON responses
- Test: malformed LLM output (missing fields, invalid kind) raises
  or flags rather than silently corrupting the unit
- Test: `unit_id` is never altered even when all semantic fields change

**Constraints:**
- Do NOT define a new port/protocol for Stage 1b — `LLMExtractor` is
  a concrete adapter class, not a protocol implementation. The pipeline
  runner calls it directly. (The design doc's `UnitExtractor` protocol
  is for Stage 1a only.)
- Use `dataclasses.replace()` for immutable updates — never mutate
- Confidence threshold for flagging should be configurable via
  `StageConfig.metadata` or a class attribute, not hardcoded
- Prompt construction should be a separate private method for testability

---

### Task 5: EvidenceBuilder port + LLM adapter (Stage 2)

**Why:** Stage 2 takes each `RegulatoryUnit` and builds a tiered
`EvidenceSet` describing how important each span is to the unit's
normative content. This is the foundation for bounded retrieval eval —
without tight evidence sets, recall metrics have a denominator explosion
problem.

> Depends on: Task 1 (LLMClient port), Task 2 (Evidence domain models)

**Files:**
- Create: `src/benchmark/ports/evidence_builder.py`
- Create: `src/benchmark/adapters/evidence/__init__.py`
- Create: `src/benchmark/adapters/evidence/llm_evidence_builder.py`
- Test: `tests/benchmark/adapters/evidence/test_llm_evidence_builder.py`

**Contract:**

Port protocol:

```python
class EvidenceBuilder(Protocol):
    """Build unit-relative tiered evidence for a regulatory unit."""
    def build(self, unit: RegulatoryUnit) -> EvidenceSet: ...
```

LLM adapter:

```python
class LLMEvidenceBuilder:
    """Stage 2: LLM-based evidence tier assignment.

    For each unit, asks the LLM to classify each span into a tier based
    on its importance to the unit's normative content. Also considers
    neighboring spans (from the same parent section) as candidates for
    the contextual tier.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: StageConfig,
        all_spans: list[BenchmarkSourceSpan],  # full corpus for neighbor lookup
    ) -> None: ...

    def build(self, unit: RegulatoryUnit) -> EvidenceSet: ...
```

Span-to-`EvidenceEntry` mapping:
- `span_id`: `"{unit_id}_{index}"` where index is the span's position
  in the tier
- `citation`, `text`, `char_start`, `char_end`: from `BenchmarkSourceSpan`
- `chunk_ids`: from `BenchmarkSourceSpan.chunk_ids_overlapping_span`
- `tier`: assigned by LLM classification

**Acceptance:**
- `EvidenceBuilder` is a `typing.Protocol`
- `LLMEvidenceBuilder` satisfies the protocol structurally
- Every span from `unit.spans` appears in exactly one tier (critical,
  supporting, or contextual) — no span is dropped or duplicated
- Neighboring spans (same `parent_section_id`, not in `unit.spans`) may
  appear in the contextual tier only
- Median critical evidence target: <= 2 spans per unit (test with
  representative sample, not enforced as a hard constraint)
- `EvidenceSet.unit_id` matches the input `RegulatoryUnit.unit_id`
- Test with mock LLMClient returning canned tier assignments
- Test: single-span unit → that span is critical
- Test: unit with cross-references → referenced spans considered for
  supporting/contextual tiers if available in `all_spans`
- Test: malformed LLM output (missing spans, unknown tier) is handled
  gracefully — fall back to all-critical rather than crash

**Constraints:**
- Port goes in `src/benchmark/ports/evidence_builder.py`, adapter in
  `src/benchmark/adapters/evidence/`
- Do not import from `rag.adapters` or `rag.ports`
- `all_spans` parameter enables neighbor lookup without coupling to the
  corpus store — the pipeline runner passes the full Stage 0 output
- The LLM prompt must include tier definitions from the design doc
  (critical = removing makes normative content incomprehensible, etc.)

---

### Task 6: Lint, typecheck, and full test pass

**Why:** Validate that all M2 code integrates cleanly with M1 and the
broader codebase. Catch import errors, type mismatches, and regressions.

> Depends on: Tasks 1–5

**Files:**
- Modify: any files with lint/type errors discovered during validation
- No new files

**Acceptance:**
- `make lint` passes
- `make typecheck` passes (specifically `./scripts/py -m mypy --config-file pyproject.toml src`)
- `make test` passes (all existing + new tests)
- No regressions in M1 tests (`tests/benchmark/`)

**Constraints:**
- Fix issues in M2 code only — do not modify M1 code unless a genuine
  type incompatibility is discovered
- Use `./scripts/py -m ruff check --fix <file>` for import sorting
