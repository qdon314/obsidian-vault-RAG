# M5: Contamination Probe + Answer-Core Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans
> to implement this plan task-by-task.

**Goal:** Implement Stage 5c (contamination probe) and Stage 6 (gold answer synthesis), completing the answer-core v1 benchmark set and writing the contamination probe runbook.

**Architecture:** `GoldAnswerSynthesizer` (new port + LLM adapter) synthesises gold answers for all validated queries before the contamination probe runs — the probe requires a gold answer as a comparison target. Stage 5c uses `LLMClient.complete()` for both the ungrounded generation call and the judge call, routing through the existing benchmark LLM port rather than calling `openai.OpenAI` directly. All stage state is frozen `BenchmarkRecord` objects mutated via `dataclasses.replace()`.

**Tech Stack:** Python 3.11, frozen dataclasses, `rag.eval.judges.GOLD_JUDGE_PROMPT` + `GoldJudgeResult` (imported from `rag.eval`), `LLMClient` protocol, `openai` (through `LLMClient`), JSONL checkpointing.

**Design doc:** `docs/plans/2026-03-21-nrc-benchmark-generation-design.md`

---

## Rationale

M5 has four dependency layers:

1. **Domain models** — `GoldAnswer` and updated `BenchmarkRecord` fields must exist before any adapter or stage can reference them.
2. **Port + LLM adapter (gold synthesis) and Stage 5c (contamination prober)** — both depend on updated domain models but are independent of each other, so they form a parallel group.
3. **Runner + CLI + runbook** — the runner can only be extended once both the synthesiser port and the contamination stage exist.

Pipeline stage ordering clarification: even though the design doc labels contamination as "Stage 5c" and gold synthesis as "Stage 6", the pipeline must run gold synthesis *before* the contamination probe — the probe needs a `gold_answer` to judge against. The runner's `_STAGE_ORDER` will therefore be `... -> stage_5b -> stage_6 -> stage_5c -> export`.

---

### Task 1: Domain model additions

**Why:** `GoldAnswer` and the answer-core fields on `BenchmarkRecord` are referenced by every downstream component in M5. They must land first.

**Files:**
- Modify: `src/benchmark/domain/models.py`
- Test: `tests/benchmark/domain/test_models_m5.py`

**Contract:**

```python
@dataclass(frozen=True, slots=True)
class GoldAnswer:
    """Stage 6 output: synthesised gold answer and answer-core rubric.

    Attached to a BenchmarkRecord after gold synthesis and before
    contamination probing. Fields mirror the design doc dataset schema.
    """
    gold_answer: str
    acceptable_answer_variants: tuple[str, ...] = ()
    required_points: tuple[str, ...] = ()
    forbidden_errors: tuple[str, ...] = ()
```

`BenchmarkRecord` gains five new optional fields appended after the existing `validation_scores` field:

```python
# M5 additions — populated in stage_6 and stage_5c respectively.
gold_answer: str | None = None
acceptable_answer_variants: tuple[str, ...] = ()
required_points: tuple[str, ...] = ()
forbidden_errors: tuple[str, ...] = ()
contamination_probes: dict[str, bool] = field(default_factory=dict)
```

`contamination_probes` maps `model_id → bool` where `True` means contaminated (ungrounded score >= 0.7).

**Acceptance:**
- `GoldAnswer` is a frozen dataclass with slots; `dataclasses.replace()` is the only mutation path.
- All five new `BenchmarkRecord` fields have defaults so existing test fixtures still instantiate without keyword args.
- `contamination_probes` round-trips correctly through `dataclasses.asdict()` and back (dict of str→bool).
- Existing `BenchmarkRecord` tests in `test_models_m4.py` pass without modification.

**Constraints:**
- Append new fields only — do not reorder or rename existing `BenchmarkRecord` fields; checkpoint JSONL deserialisers in `runner.py` depend on field names.
- `contamination_probes` must remain a plain `dict[str, bool]`, not a frozen mapping, to match the existing pattern of `hard_negatives_retriever_config` and `metadata`.

---

### Tasks 2–3 (parallel): Gold synthesis port/adapter and contamination stage

> These tasks are independent and can be executed in parallel.
> Both depend on: Task 1.

---

### Task 2: GoldAnswerSynthesizer port and LLM adapter

**Why:** Providing a `GoldAnswerSynthesizer` protocol decouples the runner from any specific LLM backend and makes the synthesiser independently testable with a stub.

**Files:**
- Create: `src/benchmark/ports/gold_answer_synthesizer.py`
- Create: `src/benchmark/adapters/generation/llm_gold_answer_synthesizer.py`
- Test: `tests/benchmark/ports/test_gold_answer_synthesizer.py`
- Test: `tests/benchmark/adapters/generation/test_llm_gold_answer_synthesizer.py`

**Contract:**

```python
# src/benchmark/ports/gold_answer_synthesizer.py
class GoldAnswerSynthesizer(Protocol):
    def synthesize(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> GoldAnswer: ...
```

`LLMGoldAnswerSynthesizer.__init__(llm_client: LLMClient, config: StageConfig)`. The adapter:
1. Formats a prompt from `query.query` and the critical + supporting evidence entries from `evidence`.
2. Calls `llm_client.complete(prompt, config)` and parses the JSON response into `GoldAnswer` fields.
3. On parse failure, logs a warning and returns a `GoldAnswer` with `gold_answer=""` (non-raising; runner skips records with empty `gold_answer`).

The prompt instructs the model to respond with JSON matching:
```json
{
  "gold_answer": "...",
  "acceptable_answer_variants": ["..."],
  "required_points": ["..."],
  "forbidden_errors": ["..."]
}
```

**Acceptance:**
- `LLMGoldAnswerSynthesizer` satisfies the `GoldAnswerSynthesizer` protocol via structural subtyping.
- Returns a `GoldAnswer` with a non-empty `gold_answer` when the LLM returns valid JSON.
- Returns `GoldAnswer(gold_answer="")` (does not raise) on LLM error or JSON parse failure.
- Prompt includes all critical evidence entries' `.text` fields and the `.query` string.
- Test uses a stub `LLMClient` that returns canned JSON; no real API calls.

**Constraints:**
- Must import only from `benchmark.domain`, `benchmark.ports`, and stdlib — no `rag.*` imports.
- Do not log the full prompt at INFO level — it may contain regulatory text; use DEBUG.

---

### Task 3: Stage 5c contamination prober

**Why:** The contamination probe detects benchmark queries that a model can answer from training data alone, making those queries invalid for answer evaluation. It must be a standalone stage function for testability and checkpointing.

**Files:**
- Create: `src/benchmark/stages/stage_5c_contamination.py`
- Test: `tests/benchmark/stages/test_stage_5c_contamination.py`

**Contract:**

```python
def run_contamination_probe(
    records: list[BenchmarkRecord],
    llm_client: LLMClient,
    config: StageConfig,
    model_id: str,
    *,
    contamination_threshold: float = 0.7,
) -> list[BenchmarkRecord]:
    """Return records with contamination_probes[model_id] populated.

    Records without a gold_answer are passed through unchanged with a
    WARNING log. Records already probed for model_id are skipped
    (idempotent on resume).
    """
```

Implementation outline:
1. For each `record` where `record.gold_answer` is not empty and `model_id not in record.contamination_probes`:
   a. **Generation call**: Assemble a prompt: `f"Answer the following question without any additional context.\n\nQuestion: {record.query}"`. Call `llm_client.complete(generation_prompt, config)` → `ungrounded_answer`.
   b. **Judge call**: Format `GOLD_JUDGE_PROMPT` (imported from `rag.eval.judges`) with `query=record.query`, `expected_answer=record.gold_answer`, `generated_answer=ungrounded_answer`. Call `llm_client.complete(judge_prompt, config)` → judge JSON text. Parse with `GoldJudgeResult.from_llm_dict(_safe_json_loads(judge_text) or {})`.
   c. Determine contaminated: `judge.score_0_1 >= contamination_threshold` where `score_0_1 = judge.correctness / 5.0` (matching the existing harness normalisation).
   d. Build updated record: `dataclasses.replace(record, contamination_probes={**record.contamination_probes, model_id: contaminated})`.
2. Return the full list of updated records.

Imports from `rag.eval.judges`: `GOLD_JUDGE_PROMPT`, `GoldJudgeResult`, `_safe_json_loads`.

**Acceptance:**
- Records without `gold_answer` are returned unchanged (no crash, WARNING logged).
- Records already containing `model_id` in `contamination_probes` are not re-probed (idempotency).
- A record whose ungrounded answer scores >= threshold has `contamination_probes[model_id] = True`.
- A record whose ungrounded answer scores < threshold has `contamination_probes[model_id] = False`.
- Judge parse failure → treated as `False` (not contaminated) with a WARNING log; does not raise.
- All original fields of the record are preserved in the returned object.
- Tests use stub `LLMClient` returning canned generation + judge JSON; no real API calls.

**Constraints:**
- `rag.eval.judges._safe_json_loads` is a private function. If it is unexported, replicate the minimal parse logic locally rather than importing with a private name.
- Do not call `openai.OpenAI` directly — route all LLM calls through the `LLMClient` port.
- `GOLD_JUDGE_PROMPT` is a module-level constant; use it as-is, filling in the `{query}`, `{expected_answer}`, `{generated_answer}` placeholders via `.format()` or f-string substitution. Inspect the actual constant in `rag/eval/judges.py` to confirm placeholder names before use.

---

### Task 4: Runner extension, CLI update, and contamination probe runbook

**Why:** The runner needs two new pipeline stages wired in the correct order (stage_6 before stage_5c), and the CLI needs flags so operators can enable gold synthesis and contamination probing without modifying code.

**Files:**
- Modify: `src/benchmark/pipeline/runner.py`
- Modify: `src/benchmark/scripts/run_benchmark_gen.py`
- Create: `docs/operations/contamination-probe-runbook.md`
- Test: `tests/benchmark/pipeline/test_runner_m5.py`

**Contract — Runner changes:**

Update `_STAGE_ORDER` to:
```python
_STAGE_ORDER = (
    "stage_0", "stage_1a", "stage_1b", "stage_2", "stage_3",
    "stage_5a", "stage_5b",
    "stage_6",   # gold answer synthesis (new)
    "stage_5c",  # contamination probe (new)
    "export",
)
```

Update `_RESUME_INPUT_FILES`:
```python
"stage_6": "stage_5b_hard_negatives.jsonl",   # or stage_5a_queries.jsonl if 5b skipped
"stage_5c": "stage_6_gold_answers.jsonl",
```

`PipelineRunner.__init__` gains two optional parameters:
```python
gold_answer_synthesizer: GoldAnswerSynthesizer | None = None,
contamination_model_id: str | None = None,
```

New stage methods:
```python
def _run_stage_6(
    self,
    validated_queries: list[ValidatedQuery],
    evidence_sets: list[EvidenceSet],
) -> list[BenchmarkRecord]:
    """Synthesise gold answers. Returns BenchmarkRecord list with gold_answer populated."""
    ...

def _run_stage_5c(
    self,
    records: list[BenchmarkRecord],
) -> list[BenchmarkRecord]:
    """Run contamination probe. Returns records with contamination_probes populated."""
    ...
```

Stage 6 checkpoint: `stage_6_gold_answers.jsonl` (serialised `BenchmarkRecord` list).
Stage 5c checkpoint: `stage_5c_probed_records.jsonl` (serialised `BenchmarkRecord` list).

Both stages are skipped (with INFO log) when their required dependency is `None`
(`gold_answer_synthesizer` for stage_6, `contamination_model_id` for stage_5c).

**Contract — JSONL deserialiser additions:**

`_dict_to_benchmark_record(d: dict[str, Any]) -> BenchmarkRecord` — must handle the five new fields with safe defaults for backward-compat with pre-M5 checkpoint files.

**Contract — CLI changes:**

Two new optional flags:
```
--synthesize-gold-answers     Enable Stage 6 gold answer synthesis (requires LLM)
--contamination-model <id>    Model ID to use for Stage 5c probe (e.g. gpt-4o-2025-01-01)
```

When `--contamination-model` is set without `--synthesize-gold-answers`, the CLI warns:
"Stage 5c requires gold answers. Pass --synthesize-gold-answers or resume from a run that has stage_6 complete."

**Contract — Runbook:**

`docs/operations/contamination-probe-runbook.md` must cover:
1. When to re-run (model version change, new answer-core candidates).
2. How to resume from `stage_5c` using `--resume-from stage_5c`.
3. Interpreting results: how to identify contaminated queries in `stage_5c_probed_records.jsonl`.
4. Promoting clean queries to answer-core: filtering on `contamination_probes[model_id] == False`.
5. Cost and time estimate for a 75-query corpus at the threshold.

**Acceptance:**
- With both flags supplied, `run()` executes `stage_6 → stage_5c → export` in sequence.
- `--resume-from stage_5c` reads `stage_6_gold_answers.jsonl` and skips re-synthesis.
- `--resume-from stage_6` re-synthesises from validated queries.
- Records where `gold_answer` is empty are skipped by stage_5c and emitted with an empty `contamination_probes` dict.
- `PipelineResult` gains `total_contaminated: int` count.
- Runbook is valid Markdown, renders without errors, covers all five topics listed in the contract.
- All M4 runner tests continue to pass (backward-compat — no new required constructor args).

**Constraints:**
- `_STAGE_ORDER` must remain a module-level tuple (the resume index lookup depends on this).
- The `export` stage reads from `stage_5c_probed_records.jsonl` when that file exists; falls back to `stage_5a_queries.jsonl` for runs that skip M5 stages (backward compat).
- Do not break any existing pipeline tests by changing `PipelineRunner.__init__` — all new parameters must be keyword-only with `None` defaults.
