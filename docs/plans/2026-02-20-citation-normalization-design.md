# Citation Normalization Design

**Date:** 2026-02-20
**Status:** Approved for Phase 1 implementation

## Problem

Citation formats are inconsistent across ingestion pipelines and eval datasets,
causing retrieval eval to report false negatives:

| Source | Example | Resolves? |
|---|---|---|
| Regulatory chunk `citation_key` | `10 CFR §50.36` | -- (index) |
| Direct-citation eval query | `10 CFR §50.2` | Yes |
| Term-mapping eval query | `10 CFR 50.48` | **No** (missing §) |
| Appendix eval query | `10 CFR 50 Appendix A` | **No** (different format) |
| Corrupted title | `0 CFR §50.55a` | **No** (wrong title) |

Root cause: the case query generator produces citations in varying formats, and
the eval harness (`_resolve_relevance_tiers`) uses exact string matching.

## Phase 1: Eval Bridge (Implement Now)

### New function: `normalize_citation_key`

**Location:** `src/rag/domain/citations.py`

Canonicalizes any CFR citation string to the human-readable form that regulatory
chunks already store (`10 CFR §{section}`).

#### Normalization rules

| Input | Output | Rule |
|---|---|---|
| `10 CFR §50.36` | `10 CFR §50.36` | Already canonical |
| `10 CFR 50.36` | `10 CFR §50.36` | Insert § after "CFR " |
| `10 CFR §50.36(c)(2)` | `10 CFR §50.36(c)(2)` | Preserve subsection markers |
| `10 CFR Part 50` | `10 CFR Part 50` | Part-level refs unchanged |
| `10 CFR 50 Appendix A` | `10 CFR 50 Appendix A` | Appendix refs unchanged |
| `10 CFR §§50.36` | `10 CFR §50.36` | Collapse double-§ |
| `0 CFR §50.55a` | `10 CFR §50.55a` | Fix title-number corruption |
| (extra whitespace) | (collapsed) | Strip and collapse |

Scope: Title 10 CFR only (this repo's domain). Does not parse the colon format
(`cfr:10:X`) -- that is Phase 2.

### Integration points

**`_build_citation_chunk_indexes` (harness.py):**
Normalize both `citation` and `citation_key` values when building the index.

**`_resolve_relevance_tiers` (harness.py):**
Normalize each query citation before looking it up in the index.

### Files changed

- `src/rag/domain/citations.py` -- add `normalize_citation_key()`
- `src/rag/eval/harness.py` -- apply normalization at index build + resolution
- `tests/domain/test_citations.py` -- unit tests for normalizer

### Test plan

- Unit tests covering every row in the normalization table above
- Passthrough test for non-CFR strings (docket numbers, ADAMS accessions)
- Integration test: query with `"10 CFR 50.48"` resolves to chunk tagged `"10 CFR §50.48"`

## Phase 2: Canonical Keys Everywhere (Future)

Goal: adopt a single canonical citation key format across both ingestion
pipelines, replacing the current asymmetry.

### Steps

1. **Choose canonical format** -- likely `cfr:10:50.46` (CitationSpan.key) for
   machine-friendliness and dedup stability.
2. **Add canonical keys to regulatory ingestion** -- store both human-readable
   `citation` and machine-friendly `citation_key` on every chunk.
3. **Run CitationSpan extraction on regulatory chunks** -- the extractors exist
   in the case pipeline; wire them into the regulatory pipeline for in-text
   cross-references.
4. **Migrate eval datasets** -- update `relevant_citations` to use canonical
   keys.
5. **Update cross-reference linking** -- wikilinks and cross-corpus resolution
   use the shared canonical form.

### Architectural note

Phase 2 would make `normalize_citation_key` the single source of truth for
citation identity across the entire system, not just the eval layer. The Phase 1
function is designed to be extended for this purpose.
