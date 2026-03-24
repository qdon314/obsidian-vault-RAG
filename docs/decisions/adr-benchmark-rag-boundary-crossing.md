# ADR: Benchmark RAG Boundary Crossing

**Status:** Accepted
**Date:** 2026-03-24
**Context:** NRC Benchmark Generation Pipeline, M4

## Context

The benchmark pipeline at `src/benchmark/` is designed as a standalone package
that imports `rag.domain` types but never touches `rag.adapters` or `rag.ports`.
This boundary preserves the benchmark's independence from RAG implementation
details.

However, Stage 5b (hard negative mining) requires running the live RAG retriever
against each benchmark query to find plausible-but-wrong chunks. Without real
hard negatives, reranker evaluation measures easy separation rather than genuine
discrimination ability.

Three alternatives were considered:

1. Pre-compute retrieval results externally and pass as static data — rejected
   because it breaks the pipeline's checkpoint/resume model and adds a manual
   step.
2. Define a benchmark-specific retriever protocol — rejected because it would
   duplicate `rag.ports.Retriever` with identical semantics.
3. Accept `rag.ports.Retriever` as an optional dependency — chosen.

## Decision

Stage 5b accepts a `Retriever` port (from `rag.ports.retriever`) as an optional
parameter in the `PipelineRunner` constructor. This is the **only** place the
benchmark package crosses the RAG boundary.

Key design choices:

- The retriever is **optional** — the runner skips Stage 5b entirely if no
  retriever is provided. The pipeline still produces valid benchmark output
  without hard negatives.
- Stage 5b is implemented as a plain function in
  `stages/stage_5b_hard_negatives.py`, not a port adapter — there is one
  sensible implementation.
- Each `HardNegativeResult` records the `retriever_config` (model name, index
  version, top_k) used at mining time, making staleness detectable when the
  retriever changes.
- Queries with fewer than 2 hard negatives are flagged with `insufficient=True`
  for manual curation, rather than relaxing the minimum.

## Consequences

- The benchmark package gains a soft import dependency on `rag.ports.Retriever`
  and `rag.domain.models.Candidate`. These are protocol/domain types only — no
  adapter implementations are imported.
- Stage 5b is the single, documented exception to the standalone boundary rule.
  All other stages remain fully independent.
- Hard negatives become stale when the retriever configuration changes (new
  embedding model, re-indexed corpus). The `retriever_config` field in
  `HardNegativeResult` enables automated staleness detection.
- The runner can operate in "retriever-less" mode for environments where Qdrant
  or the embedding model is unavailable, producing a benchmark dataset without
  hard negatives.
