# ADR: Benchmark Schema Versioning

**Status:** Accepted
**Date:** 2026-03-22
**Context:** NRC Benchmark Generation Pipeline, M1

## Context

The benchmark dataset JSONL schema will evolve as new query classes, fields,
and stages are added. We need a compatibility policy so consumers know what
to expect.

## Decision

The benchmark dataset schema follows semantic versioning:

- **Minor versions** (1.0 -> 1.1): additive fields only. Backward compatible.
  Consumers must tolerate missing optional fields.
- **Major versions** (1.x -> 2.0): breaking changes. A migration script at
  `src/benchmark/scripts/migrate_schema.py` is required for each major bump.

The `schema_version` field is mandatory on every benchmark record.

Schema version is set by the pipeline runner at export time, not by
individual stages.

## Consequences

- Consumers can safely ignore unknown fields (forward compatibility for minor bumps).
- Major bumps are rare and require explicit migration tooling.
- The `schema_version` field is the single source of truth for compatibility checks.
- The exporter validates that all output records have `schema_version` set.
