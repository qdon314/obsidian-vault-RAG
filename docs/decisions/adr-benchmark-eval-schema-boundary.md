# ADR: Benchmark–Eval Schema Boundary

**Status:** Accepted
**Date:** 2026-03-22
**Context:** NRC Benchmark Generation Pipeline, M1

## Context

The benchmark pipeline maintains a richer domain schema (tiered evidence,
rubrics, contamination probes, regulatory unit provenance) than the eval
framework's `EvalQuery` from `src/rag/eval/schema.py`.

We need to decide how these two schemas relate.

## Decision

The benchmark pipeline does **not** extend or modify `EvalQuery`.

Instead, `BenchmarkExporter` (the exporter port) is responsible for emitting
`EvalQuery`-compatible JSONL as its primary output format.

The mapping is:

| Benchmark field | EvalQuery field |
|---|---|
| `qid` | `qid` |
| `query` | `query` |
| `critical_evidence[*].chunk_ids` | `relevant_chunk_ids` / `critical_chunk_ids` |
| `source_citations` | `relevant_citations` / `critical_citations` |
| `query_class` | `query_type` (mapped via enum translation) |
| `difficulty` | `difficulty` |

Fields without an `EvalQuery` counterpart (tiered evidence detail, rubric,
contamination flags) are preserved only in the full benchmark JSONL export.

## Consequences

- The existing eval harness, metrics, judges, and Streamlit app work unchanged.
- The benchmark domain retains full fidelity for benchmark-specific analysis.
- No cross-layer coupling between the benchmark package and eval internals.
- The exporter is the single point of translation — changes to either schema
  require only exporter updates.
