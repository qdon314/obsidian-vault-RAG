# Work Item Specs: RAG System Production Readiness

Implementation-ready specs focused on evaluation rigor, retrieval quality, and operational discipline.

See [revised action plan](../plans/revised-action-plan.md) for priorities and rationale.

## Current Specs

| # | Spec | Priority | Summary |
|---|------|----------|---------|
| 01 | [OpenAI Client Hygiene](01-openai-client-hygiene.md) | P2 | Client reuse, timeouts, SDK retry |
| 02 | [Hybrid Search](02-hybrid-search.md) | P1 | BM25 + vector RRF fusion, zero new deps |
| 03 | [Eval Verdict & Release Gating](03-eval-verdict-release-gating.md) | P0 | Ship/block verdicts, CI gating, failure taxonomy |
| 04 | [Regulatory Corpus Ingestion](04-regulatory-corpus-ingestion.md) | P1 | Structured regulatory docs, adversarial eval dataset |
| 05 | [Dependency Cleanup](05-dependency-cleanup.md) | P2 | Remove dead deps (~3GB), shrink Docker image |

## Implementation Sequence

1. **Client hygiene (01)** — quick win, no dependencies on other specs
2. **Eval verdict & CI gating (03)** — centerpiece; establishes the gate
3. **Hybrid search (02)** — measurable improvement via the new verdict layer
4. **Regulatory ingestion (04)** — adversarial corpus + eval dataset, gated by verdicts
5. **Dependency cleanup (05)** — housekeeping, do anytime

## Superseded Specs

The following specs from the original plan have been dropped. See [revised action plan](../plans/revised-action-plan.md) for full rationale.

- ~~Async OpenAI Embedder~~ — async solves a server concurrency problem; batching captured in spec 01
- ~~Resilience Patterns~~ — circuit breaker inappropriate for CLI; SDK retry in spec 01
- ~~Judge Calibration~~ — eval metrics are sound
- ~~Prometheus Metrics~~ — no HTTP server; QueryTrace covers observability
- ~~Load Testing~~ — no HTTP server

The file `06-load-testing.md` is an orphan from the prior plan and can be deleted.
