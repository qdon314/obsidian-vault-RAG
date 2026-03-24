"""Domain models for the NRC benchmark generation pipeline.

All models are frozen dataclasses. Identity (``unit_id``) is structurally
derived in Stage 1a and immutable from that point forward.

M4 additions: ``HardNegativeResult``, ``BenchmarkRecord``, ``BenchmarkDataset``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from benchmark.domain.enums import EvidenceTier, QueryClass, UnitKind


@dataclass(frozen=True, slots=True)
class BenchmarkSourceSpan:
    """Stage 0 output: a benchmark-friendly view over a corpus span.

    Maps one ``(section, subsection_chain)`` pair from ``ecfr_parser``
    to the benchmark's evidence coordinate system.
    """

    source_doc_id: str
    citation: str  # e.g. "10 CFR 50.46(b)(1)"
    citation_key: str  # stable key for dedup/linking
    section_title: str
    text: str
    char_start: int
    char_end: int
    chunk_ids_overlapping_span: tuple[str, ...]
    parent_section_id: str  # ParsedSection.section_number, e.g. "50.46"
    effective_date: str  # ISO date string
    corpus_snapshot_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RegulatoryUnit:
    """Stage 1a output: a stable regulatory unit with immutable identity.

    ``unit_id`` is minted from the subsection chain (e.g.
    ``50.46_b_1_peak_cladding_temp``) and never changes after Stage 1a.
    """

    unit_id: str
    kind: UnitKind
    spans: tuple[BenchmarkSourceSpan, ...]
    citation: str
    subsection_chain: tuple[str, ...]
    parent_section_id: str
    corpus_snapshot_id: str

    # Populated by Stage 1b (LLM classification) — left as defaults in M1.
    canonical_statement: str | None = None
    entities: tuple[str, ...] = ()
    value: str | None = None  # numeric threshold if present
    conditions: tuple[str, ...] = ()

    # Populated by Stage 1a when paragraphs contain cross-references.
    cross_references: tuple[str, ...] = ()  # target_citation values

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceEntry:
    """A single span assigned to an evidence tier.

    Created by Stage 2 (evidence builder) from ``BenchmarkSourceSpan`` data.
    The ``span_id`` is minted by the evidence builder, not carried from the
    source span.
    """

    span_id: str
    citation: str  # e.g. "10 CFR 50.46(b)(1)"
    text: str  # span text for downstream prompt construction
    char_start: int
    char_end: int
    chunk_ids: tuple[str, ...]
    tier: EvidenceTier


@dataclass(frozen=True, slots=True)
class EvidenceSet:
    """Unit-relative tiered evidence for a single ``RegulatoryUnit``.

    Evidence tiers are unit-relative — they describe how important each span
    is to the regulatory unit's normative content, not to any specific query.
    See ``docs/decisions/adr-evidence-tier-semantics.md``.
    """

    unit_id: str
    critical: tuple[EvidenceEntry, ...] = ()
    supporting: tuple[EvidenceEntry, ...] = ()
    contextual: tuple[EvidenceEntry, ...] = ()


@dataclass(frozen=True, slots=True)
class StageConfig:
    """Per-stage configuration for the benchmark pipeline.

    Deterministic by default (``temperature=0.0``) for reproducibility.
    """

    model: str
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 3
    timeout_s: float = 60.0


@dataclass(frozen=True, slots=True)
class QueryCandidate:
    """Stage 3 output: a generated query candidate.

    Produced by a ``QueryGenerator`` from a ``RegulatoryUnit`` and its
    ``EvidenceSet``.  The ``candidate_id`` is minted by the generator
    (e.g. ``qc_50.46_b_1_citation_lookup_0``).
    """

    candidate_id: str
    unit_id: str
    query: str
    query_class: QueryClass
    source_citations: tuple[str, ...]
    evidence_span_ids: tuple[str, ...]
    difficulty: str = "easy"
    corpus_snapshot_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Stage 5a output: result of validating a ``QueryCandidate``.

    ``is_valid`` is ``True`` only when ``flags`` is empty.
    """

    candidate_id: str
    is_valid: bool
    flags: tuple[str, ...]
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ValidatedQuery:
    """A ``QueryCandidate`` that has passed validation.

    Shares common field names with ``QueryCandidate`` so conversion is
    straightforward via dict unpacking.
    """

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


@dataclass(frozen=True, slots=True)
class HardNegativeResult:
    """Stage 5b output: hard negatives for a single validated query.

    Hard negatives are top-k retrieval results that are **not** in the
    query's evidence set.  Required for meaningful reranker evaluation —
    without them, reranker eval measures easy separation, not real
    discrimination.
    """

    candidate_id: str
    hard_negative_chunk_ids: tuple[str, ...]
    retriever_config: dict[str, Any]  # model, index_version, top_k
    insufficient: bool = False  # True when < min_hard_negatives found


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """A fully assembled benchmark record ready for export.

    Combines data from a validated query, its evidence set, and optional
    hard negatives into the schema defined in the design doc.  The ``qid``
    preserves the ``candidate_id`` from ``ValidatedQuery`` — identity is
    not reminted.
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
    """A collection of benchmark records with schema metadata.

    Immutable snapshot of the benchmark pipeline output, suitable for
    serialization and handoff to the exporter.
    """

    schema_version: str
    records: tuple[BenchmarkRecord, ...]
    corpus_snapshot_id: str
    created_at: str  # ISO datetime
    metadata: dict[str, Any] = field(default_factory=dict)
