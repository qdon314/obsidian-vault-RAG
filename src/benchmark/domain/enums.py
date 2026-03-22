"""Enums for the NRC benchmark generation pipeline.

These use snake_case string values, intentionally distinct from
``rag.eval.schema.QueryType``. The exporter handles translation.
"""

from __future__ import annotations

from enum import StrEnum


class UnitKind(StrEnum):
    """Classification of a regulatory unit's normative function."""

    OBLIGATION = "obligation"
    PROHIBITION = "prohibition"
    THRESHOLD = "threshold"
    EXCEPTION = "exception"
    CONDITION = "condition"
    DEFINITION = "definition"
    PROCESS = "process"
    CROSS_REFERENCE = "cross_reference"


class QueryClass(StrEnum):
    """Benchmark query class — drives scoring expectations."""

    CITATION_LOOKUP = "citation_lookup"
    NARROW_FACTUAL = "narrow_factual"
    RULE_EXPLANATION = "rule_explanation"
    CROSS_REFERENCE = "cross_reference"
    SCENARIO_APPLICATION = "scenario_application"
    UNANSWERABLE = "unanswerable"
    ROBUSTNESS_VARIANT = "robustness_variant"


class EvidenceTier(StrEnum):
    """Tier of evidence relevance (unit-relative, not query-relative)."""

    CRITICAL = "critical"
    SUPPORTING = "supporting"
    CONTEXTUAL = "contextual"


class ReviewStatus(StrEnum):
    """Human review state machine for benchmark queries."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
