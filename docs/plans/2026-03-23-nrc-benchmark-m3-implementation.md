# NRC Benchmark M3: Citation-Lookup Generator + Deterministic Validator + Pipeline Runner

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task.

**Goal:** Generate citation-lookup `QueryCandidate` records from validated
regulatory units, filter them through deterministic validation gates, and
establish the checkpoint/resume pipeline runner that orchestrates all stages.

**Architecture:** Stage 3 introduces the `QueryGenerator` port and a
`TemplateQueryGenerator` adapter that produces citation-lookup queries from
`RegulatoryUnit` + `EvidenceSet` pairs using LLM-powered controlled
paraphrasing. Stage 5a introduces the `QueryValidator` port and a
`DeterministicValidator` adapter that applies rule-based checks (citation
format, duplicates, length bounds, evidence constraints). The pipeline
runner orchestrates all stages (0 through 5a), writing JSONL checkpoints
after each stage and supporting `--resume-from` to skip completed work.

**Tech Stack:** Python 3.12, frozen dataclasses, `typing.Protocol`,
`dataclasses.replace()`, `json` / JSONL serialization

**Design doc:** `docs/plans/2026-03-21-nrc-benchmark-generation-design.md`

---

## Rationale

The dependency structure is:

1. **Domain models** (`QueryCandidate`, `ValidationResult`, `ValidatedQuery`)
   must exist before any port can reference them.
2. **QueryGenerator port** depends on the domain models.
3. **QueryValidator port** depends on the domain models.
4. **TemplateQueryGenerator adapter** (Stage 3) depends on the
   `QueryGenerator` port, `LLMClient` port, and domain models.
5. **DeterministicValidator adapter** (Stage 5a) depends on the
   `QueryValidator` port and domain models. It does NOT need Stage 3.
6. **Pipeline runner** depends on all ports and adapters — it orchestrates
   stages 0 through 5a with JSONL checkpointing and resume.
7. **Benchmark pipeline runbook** documents operational usage and is
   independent of code.
8. **Lint/typecheck pass** validates everything integrates cleanly.

Tasks 1–3 are independent (models, two ports) and form the first parallel
group. Tasks 4–5 are independent of each other but both depend on the
first group. Task 6 (runner) depends on tasks 4–5. Task 7 (docs) is
independent. Task 8 (lint) depends on everything.

---

### Tasks 1–3 (parallel): Domain models and ports

> These tasks are independent and can be executed in parallel.
> No dependencies on prior M3 work.

### Task 1: Query-stage domain models

**Why:** `QueryCandidate`, `ValidationResult`, and `ValidatedQuery` are
the data contracts that flow between Stage 3 (generation), Stage 5a
(validation), and the pipeline runner. Everything downstream depends on
these shapes.

**Files:**
- Modify: `src/benchmark/domain/models.py`
- Test: `tests/benchmark/domain/test_models.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class QueryCandidate:
    candidate_id: str           # unique ID, e.g. "qc_50.46_b_1_citation_0"
    unit_id: str                # back-ref to source RegulatoryUnit
    query: str                  # the generated question text
    query_class: QueryClass
    source_citations: tuple[str, ...]  # citations from the unit
    evidence_span_ids: tuple[str, ...]  # span_ids from the unit's EvidenceSet
    difficulty: str = "easy"    # "easy" | "medium" | "hard"
    corpus_snapshot_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    candidate_id: str
    is_valid: bool
    flags: tuple[str, ...]      # validation failure reasons (empty if valid)
    scores: dict[str, float] = field(default_factory=dict)  # named scores


@dataclass(frozen=True, slots=True)
class ValidatedQuery:
    candidate_id: str
    unit_id: str
    query: str
    query_class: QueryClass
    source_citations: tuple[str, ...]
    evidence_span_ids: tuple[str, ...]
    difficulty: str
    corpus_snapshot_id: str
    validation_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
```

**Acceptance:**
- All three models are frozen dataclasses with `slots=True`
- `QueryCandidate` and `ValidatedQuery` share the same field names for
  common fields (candidate_id, unit_id, query, query_class,
  source_citations, evidence_span_ids, difficulty, corpus_snapshot_id,
  metadata) so conversion is straightforward via dict unpacking
- Immutability: `FrozenInstanceError` on attribute assignment
- Default values work correctly (empty tuples, empty dicts, `"easy"`)
- `query_class` accepts all `QueryClass` enum values

**Constraints:**
- Follow the existing `models.py` conventions: `from __future__ import
  annotations`, `field(default_factory=dict)` for mutable defaults
- Import `QueryClass` from `benchmark.domain.enums`
- Do not add `GoldAnswer` yet — that's M4+

---

### Task 2: QueryGenerator port

**Why:** The `QueryGenerator` protocol defines the interface for all query
generation adapters. Stage 3 (template generator) is the first, but future
milestones add LLM-only generators and other query classes.

**Files:**
- Create: `src/benchmark/ports/query_generator.py`
- Test: `tests/benchmark/ports/test_query_generator.py`

**Contract:**

```python
@runtime_checkable
class QueryGenerator(Protocol):
    """Generate query candidates from a regulatory unit."""

    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]: ...
```

**Acceptance:**
- Protocol is `runtime_checkable`
- `isinstance` check passes for a mock implementing `generate()`
- Port accepts `EvidenceSet` alongside `RegulatoryUnit` — the generator
  needs evidence spans to construct answerable questions
- Imports only from `benchmark.domain`

**Constraints:**
- The design doc's port signature is `generate(unit, query_class)` — we
  add `evidence: EvidenceSet` because the generator must reference
  critical spans to create answerable questions. This is an intentional
  deviation documented in the port's docstring.
- Do not add `query_class` filtering to the port — that's the caller's
  responsibility.

---

### Task 3: QueryValidator port

**Why:** The `QueryValidator` protocol defines the interface for validation
adapters. M3 implements deterministic checks only; M4 adds LLM-based
scoring.

**Files:**
- Create: `src/benchmark/ports/query_validator.py`
- Test: `tests/benchmark/ports/test_query_validator.py`

**Contract:**

```python
@runtime_checkable
class QueryValidator(Protocol):
    """Validate query candidates and optionally refine evidence."""

    def validate(self, candidate: QueryCandidate) -> ValidationResult: ...

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet: ...
```

**Acceptance:**
- Protocol is `runtime_checkable`
- `isinstance` check passes for a mock implementing both methods
- `refine_evidence` exists in the protocol — even though the M3 adapter
  returns evidence unchanged, the contract must be stable for M4

**Constraints:**
- Both methods are required for protocol satisfaction
- `refine_evidence` returns `EvidenceSet` (same type in, same type out)

---

### Tasks 4–5 (parallel): Generation and validation adapters

> These tasks are independent and can be executed in parallel.
> All depend on: Tasks 1, 2, 3.

### Task 4: TemplateQueryGenerator adapter (Stage 3)

**Why:** This is the first query generation adapter, producing
citation-lookup queries from regulatory units. It uses the LLM to
paraphrase template-based questions into realistic user queries.

**Files:**
- Create: `src/benchmark/adapters/generation/__init__.py`
- Create: `src/benchmark/adapters/generation/template_generator.py`
- Test: `tests/benchmark/adapters/generation/__init__.py`
- Test: `tests/benchmark/adapters/generation/test_template_generator.py`

**Contract:**

```python
class TemplateQueryGenerator:
    def __init__(
        self,
        llm_client: LLMClient,
        config: StageConfig,
    ) -> None: ...

    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]: ...
```

Generation strategy for `QueryClass.CITATION_LOOKUP`:
1. Build a template question from the unit's citation, canonical_statement,
   and critical evidence spans (e.g., "What does {citation} require
   regarding {topic}?")
2. Send the template + unit context to the LLM with instructions to
   produce 2–3 realistic paraphrases
3. Parse the LLM response (JSON array of query strings)
4. Mint `candidate_id` as `qc_{unit_id}_{query_class}_{index}`
5. Return `QueryCandidate` records with `evidence_span_ids` from the
   critical tier of the evidence set

For non-citation-lookup classes: raise `ValueError` — only citation_lookup
is supported in M3.

**Acceptance:**
- Satisfies `QueryGenerator` protocol (`isinstance` check passes)
- Produces 1+ `QueryCandidate` per unit for `CITATION_LOOKUP` class
- `candidate_id` is unique per candidate and deterministic from inputs
- `evidence_span_ids` references only critical-tier span IDs from the
  evidence set
- `source_citations` comes from the unit's citation
- `corpus_snapshot_id` is propagated from the unit
- Malformed LLM response falls back to the template question itself
  (never returns empty list)
- `ValueError` raised for non-CITATION_LOOKUP query classes
- LLM prompt follows the pattern established by `LLMExtractor` and
  `LLMEvidenceBuilder` (structured prompt template, JSON response
  instruction)

**Constraints:**
- Constructor takes `LLMClient` + `StageConfig` (same pattern as
  `LLMExtractor` and `LLMEvidenceBuilder`)
- Parse LLM JSON responses using the same fence-stripping logic from
  `LLMExtractor._parse_response` — extract into a shared utility or
  inline (keep it simple, don't over-abstract for two call sites)
- `difficulty` defaults to `"easy"` for citation-lookup (the simplest
  query class)

---

### Task 5: DeterministicValidator adapter (Stage 5a)

**Why:** The first validation gate — catches malformed citations,
duplicates, banned broad verbs, length violations, and evidence
constraint violations before any LLM-based scoring (M4).

**Files:**
- Create: `src/benchmark/adapters/validation/__init__.py`
- Create: `src/benchmark/adapters/validation/deterministic_validator.py`
- Test: `tests/benchmark/adapters/validation/__init__.py`
- Test: `tests/benchmark/adapters/validation/test_deterministic_validator.py`

**Contract:**

```python
class DeterministicValidator:
    def __init__(
        self,
        *,
        known_queries: list[str] | None = None,
        max_query_length: int = 500,
        min_query_length: int = 10,
        max_evidence_spans: int = 6,
    ) -> None: ...

    def validate(self, candidate: QueryCandidate) -> ValidationResult: ...

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet: ...
```

Deterministic checks (from design doc Stage 5a):
1. **Citation format valid**: query text contains a well-formed CFR
   citation (pattern: `\d+ CFR \d+\.\d+`)
2. **No duplicate normalized query text**: compare against `known_queries`
   (lowercased, stripped)
3. **No banned broad verbs**: reject if query starts with
   "Summarize", "List all", "Describe everything", "What are all the rules"
4. **Query length within bounds**: `min_query_length <= len(query) <= max_query_length`
5. **Evidence set non-empty**: `evidence_span_ids` must be non-empty
6. **Evidence set size below threshold**: `len(evidence_span_ids) <= max_evidence_spans`
   for narrow query classes (citation_lookup, narrow_factual)
7. **`corpus_snapshot_id` populated**: must be non-empty string

Each failing check adds a flag string to `ValidationResult.flags`.
`is_valid` is `True` only when `flags` is empty.

`refine_evidence()`: returns evidence unchanged (identity function).
This is the M3 stub — M4's LLM validator will implement per-query
evidence refinement.

**Acceptance:**
- Satisfies `QueryValidator` protocol (`isinstance` check passes)
- Each check is independently testable (one test per check type)
- Valid candidate returns `ValidationResult(is_valid=True, flags=())`
- Invalid candidate returns `is_valid=False` with descriptive flag strings
- Multiple failing checks produce multiple flags
- `known_queries` defaults to empty list (no dedup check)
- Duplicate detection is case-insensitive and whitespace-normalized
- `refine_evidence()` returns the input `EvidenceSet` unchanged
- Citation format check uses regex — does not validate that the citation
  actually exists in the corpus (that's a semantic check for M4)

**Constraints:**
- No LLM calls — this is purely deterministic
- Do not import from `benchmark.adapters` or `benchmark.ports.llm_client`
- The banned verbs list is hardcoded (not configurable) — keep it short
  and obvious. These are high-confidence rejections.

---

### Task 6: Pipeline runner with checkpoint/resume

**Why:** The runner orchestrates all stages (0 → 1a → 1b → 2 → 3 → 5a)
with JSONL checkpoints after each stage. This is the operational backbone
— without it, every run restarts from scratch and there's no cost control.

**Files:**
- Create: `src/benchmark/pipeline/__init__.py`
- Create: `src/benchmark/pipeline/runner.py`
- Test: `tests/benchmark/pipeline/__init__.py`
- Test: `tests/benchmark/pipeline/test_runner.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class PipelineConfig:
    run_id: str                          # unique run identifier
    output_dir: str                      # path to benchmark_runs/<run_id>/
    resume_from: str | None = None       # e.g., "stage_3" to skip earlier stages
    corpus_snapshot_id: str = ""         # expected snapshot ID (abort if mismatch)

class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        # Stage 0 inputs
        corpus_spans_builder: Callable[[], list[BenchmarkSourceSpan]],
        # Stage 1a
        unit_extractor: UnitExtractor,
        # Stage 1b
        llm_extractor_factory: Callable[[], LLMExtractor] | None = None,
        # Stage 2
        evidence_builder: EvidenceBuilder | None = None,
        # Stage 3
        query_generator: QueryGenerator | None = None,
        # Stage 5a
        query_validator: QueryValidator | None = None,
    ) -> None: ...

    def run(self) -> PipelineResult: ...
```

```python
@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: str
    stages_completed: tuple[str, ...]   # e.g., ("stage_0", "stage_1a", ...)
    output_dir: str
    total_candidates: int
    total_validated: int
    total_flagged: int
```

Stage execution flow:
- Each stage reads its input from the prior stage's output (in-memory
  during a full run, or from checkpoint JSONL on resume)
- Each stage writes a JSONL checkpoint: `stage_0_spans.jsonl`,
  `stage_1a_units.jsonl`, `stage_1b_classified.jsonl`,
  `stage_2_evidence.jsonl`, `stage_3_candidates.jsonl`,
  `stage_5a_validated.jsonl`
- `run_config.json` is written at run start with the full pipeline config
- On resume, the runner reads from the checkpoint file for the resume
  stage and proceeds from there

JSONL serialization:
- Frozen dataclasses → dict via `dataclasses.asdict()` for write
- Dict → frozen dataclass via constructor kwargs for read
- Enum values serialize as strings (StrEnum handles this naturally)

**Acceptance:**
- Full run (no resume) executes stages 0 → 1a → 1b → 2 → 3 → 5a in order
- Each stage writes its checkpoint JSONL to `output_dir`
- `run_config.json` written at start with PipelineConfig fields
- Resume from `stage_3` reads `stage_2_evidence.jsonl` and proceeds
- Resume from `stage_5a` reads `stage_3_candidates.jsonl` and proceeds
- Missing checkpoint file on resume raises `FileNotFoundError` with
  descriptive message
- Optional ports (llm_extractor_factory, evidence_builder,
  query_generator, query_validator) — if `None` and the stage is reached,
  raise `ValueError` naming the missing port
- `PipelineResult` accurately reports counts
- Snapshot verification: if `corpus_snapshot_id` is set in config, verify
  it matches the Stage 0 output; abort with `ValueError` on mismatch

**Constraints:**
- Do not import concrete adapters — the runner depends only on port
  protocols and the Stage 0 builder function
- `llm_extractor_factory` is a callable (not an `LLMExtractor` directly)
  because `LLMExtractor.classify()` has a different signature than the
  `UnitExtractor` protocol — the runner calls it directly. Keep this
  pragmatic, not over-abstracted.
- JSONL write/read is the runner's responsibility (not delegated to
  adapters or a separate serialization layer)
- Use `pathlib.Path` for all file operations
- `PipelineConfig` goes in `runner.py` (not `models.py`) — it's pipeline
  infrastructure, not domain

---

### Task 7: Benchmark pipeline runbook

**Why:** The design doc requires `docs/operations/benchmark-pipeline-runbook.md`
by M3 — an operational guide for running the pipeline, understanding
checkpoints, estimating costs, and troubleshooting.

**Files:**
- Create: `docs/operations/benchmark-pipeline-runbook.md`

**Contract:**

The runbook covers:
1. **Prerequisites**: Python environment, API keys, corpus availability
2. **Running a full pipeline**: command, expected output directory structure
3. **Checkpoint files**: what each file contains, how to inspect
4. **Resuming a run**: `--resume-from` usage, what gets skipped
5. **Cost estimation**: approximate LLM calls per stage at v1 scale
   (reference the design doc's performance expectations)
6. **Troubleshooting**: common errors (snapshot mismatch, missing port,
   malformed checkpoint)

**Acceptance:**
- File exists at `docs/operations/benchmark-pipeline-runbook.md`
- References the correct script path (`src/benchmark/scripts/run_benchmark_gen.py`
  per design doc, or the runner module)
- Checkpoint file names match what the runner actually writes
- Cost estimates are consistent with the design doc's performance
  expectations section

**Constraints:**
- Operational documentation — not a design document
- Use `./scripts/py` for all Python commands (per CLAUDE.md)
- Keep it concise — this is a runbook, not a tutorial

---

### Task 8: Lint, typecheck, and full test pass

**Why:** Final integration validation — ensures all new code passes the
project's quality gates and integrates cleanly with the existing
benchmark package.

**Files:**
- Modify: (any files needing lint/type fixes)
- Test: (run full test suite)

**Acceptance:**
- `make lint` passes with zero errors
- `make typecheck` passes with zero errors
- `make test` passes — all existing + new tests green
- No import cycle issues between new modules and existing code
- `ruff check` and `ruff format` produce no changes

**Constraints:**
- Fix issues in the files created/modified by M3 tasks only
- Do not refactor existing M1/M2 code unless it breaks
