# ADR: Evidence Tier Semantics

**Status:** Accepted
**Date:** 2026-03-23
**Context:** NRC Benchmark Generation Pipeline, M2

## Context

The benchmark pipeline assigns evidence spans to tiers (critical, supporting,
contextual) in Stage 2. A key design question is whether these tiers should
be relative to the regulatory unit or to a specific query.

The constraint: evidence must be assigned in Stage 2, before queries exist
(Stage 3). A single regulatory unit may feed multiple query classes — citation
lookup, narrow factual, rule explanation, etc. — each with different retrieval
expectations.

## Decision

Evidence tiers are **unit-relative**: they describe how important each span is
to the regulatory unit's normative content, not to any specific query.

Tier definitions:

- **Critical:** Removing this span makes the regulatory unit's normative
  content incomprehensible.
- **Supporting:** Removing this span degrades completeness of understanding,
  but the core obligation or threshold remains clear.
- **Contextual:** Nearby material that may help interpretation but is neither
  critical nor supporting.

### Post-generation refinement

After queries are generated (Stage 3) and validated (Stage 5a), the
`QueryValidator.refine_evidence()` method may narrow the unit-level evidence
set for a specific query. For example:

- A citation-lookup query may need only the critical tier.
- A cross-reference query may promote contextual spans from a linked unit
  to supporting.

This refinement produces per-query tier assignments from the unit-level
evidence set. It does not modify the unit-level `EvidenceSet`.

## Consequences

- Evidence sets are reusable across all query classes derived from a unit.
- Retrieval eval benefits from tight relevance sets — the median critical
  evidence target is <= 2 spans, <= 4 chunk IDs.
- Per-query narrowing is a downstream concern (Stage 5a), cleanly separated
  from evidence construction (Stage 2).
- The `EvidenceSet` domain model groups entries by tier with no query
  dependency, keeping Stage 2 output stable regardless of which query
  classes are generated later.
