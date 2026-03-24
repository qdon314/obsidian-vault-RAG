# M4: LLM Validator + Hard Negative Mining + Unanswerable Class Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task.

**Goal:** Complete the retrieval-core v1 benchmark by adding LLM-based validation, hard negative mining via the RAG retriever, unanswerable query generation, and EvalQuery-compatible export.

**Architecture:** M4 extends the existing hexagonal benchmark pipeline with three new adapters (LLM validator, unanswerable generator, EvalQuery exporter), one new stage function (5b hard negative mining that intentionally crosses the RAG boundary), and domain models to carry hard-negative and export-ready data. The runner gains multi-class generation dispatch, optional Stage 5b, and a terminal export stage.

**Tech Stack:** Python 3.12, frozen dataclasses, Protocol-based ports, `LLMClient` port for all LLM calls, `rag.ports.Retriever` for Stage 5b, `rag.eval.schema.EvalQuery` as export target.

**Design doc:** `docs/plans/2026-03-21-nrc-benchmark-generation-design.md`

---

## Rationale

The decomposition follows the data flow through the pipeline: new domain models first (they define the contracts everything else depends on), then the three independent adapter/stage implementations (LLM validator, unanswerable generator, hard negative miner + exporter), and finally the runner integration that wires everything together. Documentation tasks are independent of code.

**Task 1** adds the domain models that Tasks 2, 4, and 5 consume. **Tasks 2, 3, and 4** are independent of each other — they implement different adapters/stages against different ports, touch disjoint file sets, and can be parallelized. **Task 5** integrates everything into the runner and depends on all prior tasks. **Tasks 6 and 7** are documentation with no code dependencies.

The grouping enables three parallel worktrees after Task 1 lands.

---

### Task 1: Domain models and exporter port for M4

**Why:** Every downstream task needs `HardNegativeResult` (Stage 5b output), `BenchmarkRecord` (the full assembled record for export), and `BenchmarkDataset` (collection wrapper). The `BenchmarkExporter` port protocol must exist before the export adapter can be built.

**Files:**
- Modify: `src/benchmark/domain/models.py`
- Create: `src/benchmark/ports/exporter.py`
- Create: `tests/benchmark/domain/test_models_m4.py`
- Create: `tests/benchmark/ports/test_exporter.py`

**Contract:**

New dataclasses in `models.py`:

```python
@dataclass(frozen=True, slots=True)
class HardNegativeResult:
    """Stage 5b output: hard negatives for a single validated query."""
    candidate_id: str
    hard_negative_chunk_ids: tuple[str, ...]
    retriever_config: dict[str, Any]  # model, index_version, top_k
    insufficient: bool = False  # True when < 2 hard negatives found

@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """A fully assembled benchmark record ready for export.

    Combines data from validated query, evidence, and hard negatives
    into the schema defined in the design doc.
    """
    qid: str
    query: str
    query_class: QueryClass
    difficulty: str
    source_unit_ids: tuple[str, ...]
    source_citations: tuple[str, ...]
    critical_evidence: tuple[EvidenceEntry, ...]
    supporting_evidence: tuple[EvidenceEntry, ...]
    contextual_evidence: tuple[EvidenceEntry, ...]
    hard_negative_chunk_ids: tuple[str, ...]
    hard_negatives_retriever_config: dict[str, Any]
    corpus_snapshot_id: str
    valid_as_of: str  # ISO date
    is_unanswerable: bool = False
    unanswerable_reason: str | None = None
    robustness_parent_qid: str | None = None
    validation_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class BenchmarkDataset:
    """A collection of benchmark records with schema metadata."""
    schema_version: str
    records: tuple[BenchmarkRecord, ...]
    corpus_snapshot_id: str
    created_at: str  # ISO datetime
    metadata: dict[str, Any] = field(default_factory=dict)
```

New port in `ports/exporter.py`:

```python
@runtime_checkable
class BenchmarkExporter(Protocol):
    """Export a benchmark dataset to an external format."""
    def export(self, dataset: BenchmarkDataset) -> None: ...
```

**Acceptance:**
- All three new dataclasses are frozen with `slots=True`
- `BenchmarkRecord` round-trips through `dataclasses.asdict()` + JSON serialization (same pattern as existing models)
- `HardNegativeResult.insufficient` defaults to `False`
- `BenchmarkDataset.records` is a tuple (immutable)
- `BenchmarkExporter` is `@runtime_checkable`
- Port conformance test: a minimal stub class `isinstance`-checks against `BenchmarkExporter`

**Constraints:**
- Follow existing model patterns in `models.py` (frozen, slots, tuple for sequences, dict for open metadata)
- `BenchmarkRecord.qid` uses the `candidate_id` from `ValidatedQuery` — identity is preserved, not reminted
- Import only from `benchmark.domain` — no cross-layer imports

---

### Tasks 2–4 (parallel): Adapters and stages

> These tasks are independent and can be executed in parallel.
> All depend on: Task 1.

### Task 2: LLM validator adapter

**Why:** The M3 `DeterministicValidator` only applies rule-based checks and has a pass-through `refine_evidence()`. M4 needs model-based scoring (plausibility, boundedness, ambiguity, specificity, leakage) and real per-query evidence refinement. This is a **new** adapter — the deterministic validator remains unchanged and can be composed with it.

**Files:**
- Create: `src/benchmark/adapters/validation/llm_validator.py`
- Create: `tests/benchmark/adapters/validation/test_llm_validator.py`

**Contract:**

```python
class LLMValidator:
    """Stage 5a: LLM-based validation with model-scored quality dimensions.

    Composes with DeterministicValidator — call deterministic checks first,
    then LLM checks on candidates that pass deterministic validation.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: StageConfig,
        *,
        deterministic: DeterministicValidator,
        score_thresholds: dict[str, float] | None = None,
    ) -> None: ...

    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        """Run deterministic checks first; if passed, run LLM scoring.

        LLM scores five dimensions (0.0–1.0):
        - plausibility: would a real user ask this?
        - boundedness: answerable from supplied evidence?
        - ambiguity: multiple materially different readings?
        - specificity: is the target clear?
        - leakage: mirrors statutory phrasing too directly?

        Scores are stored in ValidationResult.scores.
        Any score below its threshold adds a flag (e.g. "low_plausibility").
        """
        ...

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet:
        """Narrow unit-level evidence for a specific query via LLM.

        The LLM decides which tiers are relevant for this query class:
        - citation_lookup: may keep only critical tier
        - cross_reference: may promote contextual spans from linked units
        Returns a new EvidenceSet with refined tier assignments.
        """
        ...
```

Default score thresholds: `{"plausibility": 0.6, "boundedness": 0.6, "ambiguity": 0.7, "specificity": 0.6, "leakage": 0.7}`.

**Acceptance:**
- Implements `QueryValidator` protocol (isinstance check passes)
- Deterministic checks run first; LLM scoring only runs if deterministic checks pass (avoids wasting LLM calls on clearly invalid candidates)
- All five score dimensions present in `ValidationResult.scores` for LLM-scored candidates
- Candidates failing deterministic checks get `scores={}` (no LLM call made)
- Low scores add descriptive flags (e.g., `"low_plausibility:0.3"`)
- `refine_evidence()` returns a new `EvidenceSet` (never mutates input)
- `refine_evidence()` falls back to returning evidence unchanged if LLM response is malformed
- Malformed LLM responses for scoring fall back to flagging with `"llm_parse_error"` (not silently passing)
- Tests use a mock `LLMClient` returning canned JSON responses

**Constraints:**
- All LLM calls go through the `LLMClient` port — no direct API calls
- Must compose with `DeterministicValidator`, not replace it
- JSON prompt/response format follows the same code-fence-stripping pattern as `template_generator.py`

---

### Task 3: Unanswerable query generator

**Why:** Unanswerable queries are a required class for nuclear regulatory safety. The current `TemplateQueryGenerator` only handles `CITATION_LOOKUP`. This task adds a new generator specifically for `UNANSWERABLE` queries using three strategies: near-miss, domain-boundary, and fabricated-citation.

**Files:**
- Create: `src/benchmark/adapters/generation/unanswerable_generator.py`
- Create: `tests/benchmark/adapters/generation/test_unanswerable_generator.py`

**Contract:**

```python
class UnanswerableGenerator:
    """Generate unanswerable queries using three strategies.

    Strategies (from design doc Stage 3F):
    1. Near-miss: take a real unit, ask about a related but uncovered subsection
    2. Domain-boundary: use curated adjacent-domain topics (OSHA, EPA, DOE)
    3. Fabricated-citation: generate plausible but non-existent CFR reference
    """

    # Curated seed list for domain-boundary strategy.
    ADJACENT_DOMAINS: ClassVar[tuple[str, ...]] = (
        "29 CFR 1910",   # OSHA general industry
        "40 CFR 61",     # EPA NESHAP
        "10 CFR 830",    # DOE nuclear safety
        "49 CFR 173",    # DOT hazmat transport
    )

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
    ) -> list[QueryCandidate]:
        """Generate unanswerable query candidates.

        Raises ValueError if query_class != UNANSWERABLE.
        Produces 1-2 candidates per unit using a randomly selected strategy.
        Candidates have:
        - evidence_span_ids = () (empty — no evidence expected)
        - metadata["unanswerable_strategy"] = "near_miss" | "domain_boundary" | "fabricated_citation"
        - metadata["is_unanswerable"] = True
        - metadata["unanswerable_reason"] = strategy-specific reason string
        - corpus_snapshot_id from the unit
        """
        ...
```

**Acceptance:**
- Implements `QueryGenerator` protocol (isinstance check passes)
- Raises `ValueError` for any `query_class` other than `UNANSWERABLE`
- Produces candidates with empty `evidence_span_ids`
- Each candidate's `metadata` contains `is_unanswerable`, `unanswerable_reason`, and `unanswerable_strategy`
- `candidate_id` format: `"qc_{unit_id}_unanswerable_{strategy}_{i}"`
- Near-miss strategy: query references a plausible but non-existent subsection of the unit's parent section
- Domain-boundary strategy: query uses an adjacent-domain citation from the seed list
- Fabricated-citation strategy: query uses a non-existent CFR section number
- Falls back to template-based unanswerable if LLM response is malformed
- Tests verify each strategy independently with canned LLM responses

**Constraints:**
- All LLM calls go through `LLMClient` port
- Strategy selection must be deterministic given a seed (use unit_id hash for reproducibility)
- `ADJACENT_DOMAINS` is a class variable, not hardcoded in prompts

---

### Task 4: Stage 5b hard negative mining + EvalQuery exporter

**Why:** Hard negatives are required for valid reranker evaluation — without them, reranker eval measures easy separation, not real discrimination. The EvalQuery exporter is the bridge that lets the existing eval harness consume benchmark output. These two are combined because they are small individually and share the same dependency (Task 1 models) and test patterns.

**Files:**
- Create: `src/benchmark/stages/stage_5b_hard_negatives.py`
- Create: `src/benchmark/adapters/export/__init__.py`
- Create: `src/benchmark/adapters/export/eval_query_exporter.py`
- Create: `tests/benchmark/stages/test_stage_5b_hard_negatives.py`
- Create: `tests/benchmark/adapters/export/__init__.py`
- Create: `tests/benchmark/adapters/export/test_eval_query_exporter.py`

**Contract — Stage 5b:**

```python
def mine_hard_negatives(
    queries: list[ValidatedQuery],
    evidence_sets: dict[str, EvidenceSet],  # keyed by unit_id
    retriever: Retriever,
    *,
    top_k: int = 20,
    min_hard_negatives: int = 2,
    retriever_config: dict[str, Any],  # model, index_version — recorded for staleness detection
) -> list[HardNegativeResult]:
    """Run the retriever against each query; top-k results NOT in the
    evidence set become hard negatives.

    Sets insufficient=True when fewer than min_hard_negatives found.
    """
    ...
```

The function:
1. For each query, calls `retriever.retrieve(query.query, top_k=top_k)`
2. Collects chunk_ids from results
3. Subtracts chunk_ids present in the query's evidence set (all tiers)
4. Takes remaining chunk_ids as hard negatives
5. Flags `insufficient=True` if count < `min_hard_negatives`

**Contract — EvalQuery exporter:**

```python
class EvalQueryExporter:
    """Export BenchmarkDataset as EvalQuery-compatible JSONL.

    Maps benchmark fields to EvalQuery fields per the design doc mapping table:
    - qid → qid
    - query → query
    - critical_evidence[*].chunk_ids → relevant_chunk_ids (union)
    - source_citations → relevant_citations
    - query_class → query_type (via _QUERY_CLASS_TO_QUERY_TYPE mapping)
    - difficulty → difficulty
    - is_unanswerable → is_unanswerable
    - unanswerable_reason → unanswerable_reason

    Also populates tiered chunk ID fields:
    - critical_evidence[*].chunk_ids → critical_chunk_ids
    - supporting_evidence[*].chunk_ids → supporting_chunk_ids
    - contextual_evidence[*].chunk_ids → context_chunk_ids
    """

    def __init__(self, output_path: Path) -> None: ...

    def export(self, dataset: BenchmarkDataset) -> None:
        """Write one EvalQuery JSON object per line to output_path."""
        ...
```

Enum translation map:

```python
_QUERY_CLASS_TO_QUERY_TYPE: dict[QueryClass, QueryType] = {
    QueryClass.CITATION_LOOKUP: QueryType.FACTUAL,
    QueryClass.NARROW_FACTUAL: QueryType.FACTUAL,
    QueryClass.RULE_EXPLANATION: QueryType.PROCEDURAL,
    QueryClass.CROSS_REFERENCE: QueryType.MULTI_HOP,
    QueryClass.SCENARIO_APPLICATION: QueryType.SCENARIO,
    QueryClass.UNANSWERABLE: QueryType.FACTUAL,  # type is orthogonal to answerability
    QueryClass.ROBUSTNESS_VARIANT: QueryType.FACTUAL,
}
```

**Acceptance — Stage 5b:**
- Returns one `HardNegativeResult` per input query
- Hard negative chunk_ids exclude ALL chunk_ids from the query's evidence (critical + supporting + contextual)
- `insufficient=True` when hard negatives count < `min_hard_negatives`
- `retriever_config` is passed through to every result (for staleness detection)
- Works with an empty retriever result (returns empty hard negatives, insufficient=True)
- Tests use a mock `Retriever` returning `Candidate` objects with known chunk_ids

**Acceptance — EvalQuery exporter:**
- Implements `BenchmarkExporter` protocol (isinstance check passes)
- Output file is valid JSONL — each line parses to a dict matching `EvalQuery` fields
- `relevant_chunk_ids` is the union of all evidence tier chunk_ids
- `relevant_citations` is the set of `source_citations`
- Tiered chunk ID fields (`critical_chunk_ids`, `supporting_chunk_ids`, `context_chunk_ids`) are populated
- Unanswerable records have `is_unanswerable=True`, empty `relevant_chunk_ids`
- Output is loadable by `EvalQuery.from_dict()` (round-trip test)
- `query_type` correctly maps from `QueryClass` via the translation table

**Constraints:**
- Stage 5b imports `rag.ports.Retriever` and `rag.domain.models.Candidate` — this is the **only** place the benchmark package crosses the RAG boundary
- EvalQuery exporter imports `rag.eval.schema.EvalQuery` and `rag.eval.schema.QueryType` for the mapping — read-only, no modification of eval schema
- Stage 5b is a plain function (like Stage 0), not a port adapter — there is one sensible implementation

---

### Task 5: Runner extension for M4 stages

**Why:** The pipeline runner must orchestrate the new stages (multi-class generation, LLM validation, Stage 5b, export) while preserving checkpoint/resume semantics. This is the integration point that wires all M4 components together.

**Files:**
- Modify: `src/benchmark/pipeline/runner.py`
- Create: `src/benchmark/scripts/run_benchmark_gen.py`
- Modify: `tests/benchmark/pipeline/test_runner.py`
- Modify: `docs/operations/benchmark-pipeline-runbook.md`

**Contract:**

Entry point script (`run_benchmark_gen.py`):

```python
"""CLI entry point for the benchmark generation pipeline.

Usage:
    ./scripts/py -m benchmark.scripts.run_benchmark_gen \
        --run-id "run_20260324" \
        --output-dir benchmark_runs/ \
        --model gpt-4o \
        [--resume-from stage_3] \
        [--query-classes citation_lookup,unanswerable] \
        [--skip-hard-negatives] \
        [--export-path eval/datasets/benchmark_v1.jsonl] \
        [--valid-as-of 2026-03-24]
"""

def main() -> None:
    """Parse CLI args, wire adapters, call PipelineRunner.run().

    Wiring:
    - LLMClient: OpenAI-based adapter (from OPENAI_API_KEY env var)
    - QueryGenerators: {CITATION_LOOKUP: TemplateQueryGenerator,
                        UNANSWERABLE: UnanswerableGenerator}
    - QueryValidator: LLMValidator (wrapping DeterministicValidator)
    - Retriever: optional, from RAG container (requires Qdrant running)
    - Exporter: EvalQueryExporter writing to --export-path
    """
    ...
```

Runner changes:

```python
# Extended stage order
_STAGE_ORDER = (
    "stage_0", "stage_1a", "stage_1b", "stage_2",
    "stage_3", "stage_5a", "stage_5b", "export",
)

# New constructor parameters (added to existing ones):
class PipelineRunner:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        # ... existing params unchanged ...
        query_generators: dict[QueryClass, QueryGenerator] | None = None,
        # Replaces single query_generator param. Backward compat: if
        # query_generator is passed (singular), wrap as {CITATION_LOOKUP: gen}.
        retriever: Retriever | None = None,  # Stage 5b (optional)
        retriever_config: dict[str, Any] | None = None,  # recorded for staleness
        exporter: BenchmarkExporter | None = None,  # terminal export stage
        query_classes: tuple[QueryClass, ...] = (QueryClass.CITATION_LOOKUP,),
        valid_as_of: str = "",  # ISO date for export records
    ) -> None: ...
```

Stage 3 changes:
- Iterate over `self._query_classes` instead of hardcoding `CITATION_LOOKUP`
- For each `(unit, query_class)` pair, look up the generator in `self._query_generators`
- Skip silently if no generator registered for a class (log warning)

Stage 5a changes:
- After validation, call `refine_evidence()` on each validated query
- Write refined evidence to `stage_5a_refined_evidence.jsonl` checkpoint

Stage 5b (new):
- Skip entirely if `self._retriever is None` (log info, not error)
- Call `mine_hard_negatives()` with validated queries and evidence
- Write `stage_5b_hard_negatives.jsonl` checkpoint

Export stage (new):
- Assemble `BenchmarkRecord` from validated queries + refined evidence + hard negatives
- Build `BenchmarkDataset` wrapper
- Call `self._exporter.export(dataset)` if exporter is provided
- Write `benchmark_records.jsonl` checkpoint (full records, independent of exporter)

Backward compatibility:
- If `query_generator` (singular) is passed, wrap as `{QueryClass.CITATION_LOOKUP: query_generator}`
- If `query_generators` (plural) is passed, use directly
- Raise `ValueError` if both are passed

**Acceptance:**
- Full pipeline (0 → 1a → 1b → 2 → 3 → 5a → 5b → export) runs end-to-end with mock adapters
- Resume from any stage works (including new stages 5b, export)
- Stage 5b is skipped when no retriever provided (pipeline still completes)
- Export stage is skipped when no exporter provided (pipeline still completes)
- Multi-class generation: passing `{CITATION_LOOKUP: gen_a, UNANSWERABLE: gen_b}` produces candidates from both
- `PipelineResult` updated: add `total_hard_negatives: int` and `total_exported: int` fields
- Backward compat: existing tests pass with `query_generator` (singular) param unchanged
- New checkpoint files: `stage_5a_refined_evidence.jsonl`, `stage_5b_hard_negatives.jsonl`, `benchmark_records.jsonl`
- Checkpoint deserialization helpers added for `HardNegativeResult` and `BenchmarkRecord`
- `run_benchmark_gen.py` is runnable via `./scripts/py -m benchmark.scripts.run_benchmark_gen --help`
- Entry script uses `argparse` and wires all M4 adapters; validates `docs/operations/benchmark-pipeline-runbook.md` CLI examples
- Update `docs/operations/benchmark-pipeline-runbook.md`: add Stage 5b/export checkpoint files, new resume-from stages, updated cost estimation (LLM validator adds ~50 calls, unanswerable gen adds ~50), and correct the CLI invocation to `./scripts/py -m benchmark.scripts.run_benchmark_gen`

**Constraints:**
- Runner must NOT import adapter implementations — only port protocols
- `Retriever` import is guarded: `from rag.ports.retriever import Retriever` (the only RAG boundary crossing)
- Stage 5b function is called from runner but lives in `stages/stage_5b_hard_negatives.py`
- Checkpoint I/O follows existing `_write_checkpoint` / `_read_jsonl` patterns
- Entry script imports adapter implementations — it is the composition root (like `rag/app/container.py`)

---

### Tasks 6–7 (parallel): Documentation

> These tasks are independent of code tasks and can be executed in parallel with any group.

### Task 6: ADR — Benchmark RAG boundary crossing

**Why:** The design doc requires `docs/decisions/adr-benchmark-rag-boundary-crossing.md` by M4. Documents why hard negative mining requires a live `Retriever` port and the implications for the standalone package boundary.

**Files:**
- Create: `docs/decisions/adr-benchmark-rag-boundary-crossing.md`

**Contract:**

ADR sections (follow existing ADR format in `docs/decisions/`):
- **Status:** Accepted
- **Context:** Benchmark pipeline is standalone; Stage 5b needs live retriever
- **Decision:** Accept `Retriever` port as optional constructor arg in runner; Stage 5b is the only boundary-crossing stage
- **Consequences:** Benchmark package gains a soft dependency on `rag.ports.Retriever`; Stage 5b is skippable; hard negatives become stale when retriever changes (tracked via `retriever_config`)

**Acceptance:**
- Follows the ADR format established by `docs/decisions/adr-evidence-tier-semantics.md`
- References the design doc sections on Stage 5b
- Explains staleness detection via `retriever_config`

---

### Task 7: Benchmark review guide

**Why:** The design doc requires `docs/operations/benchmark-review-guide.md` by M4. Provides reviewer instructions, acceptance criteria per query class, and examples of common rejection reasons.

**Files:**
- Create: `docs/operations/benchmark-review-guide.md`

**Contract:**

Sections:
- **Overview:** Purpose, audience, review state machine (`pending → approved/rejected/needs_revision`)
- **Review criteria per query class:** Citation lookup, unanswerable — what to check for each
- **Common rejection reasons:** With examples (malformed citations, too-broad queries, semantic duplicates, misclassified unanswerable)
- **Review workflow:** How to edit JSONL, fields to update (`review_status`, `reviewed_by`, `reviewed_at`)

**Acceptance:**
- Covers both citation_lookup and unanswerable query classes (the two M4 classes)
- Includes at least 3 concrete rejection examples with before/after
- References the review state machine from the design doc
- Mentions the `metadata.review_status` field and `ReviewStatus` enum values
