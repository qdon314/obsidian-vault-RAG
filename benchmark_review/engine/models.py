from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ReviewStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"


@dataclass(frozen=True, slots=True)
class EvidenceSpan:
    span_id: str
    citation: str
    text: str
    char_start: int
    char_end: int
    tier: str  # "critical" | "supporting" | "contextual"


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    candidate_id: str
    unit_id: str
    query: str
    query_class: str
    difficulty: str
    source_citations: tuple[str, ...]
    evidence_span_ids: tuple[str, ...]
    is_valid: bool
    validation_flags: tuple[str, ...]
    critical_evidence: tuple[EvidenceSpan, ...]
    supporting_evidence: tuple[EvidenceSpan, ...]
    contextual_evidence: tuple[EvidenceSpan, ...]
    is_unanswerable: bool
    unanswerable_reason: str | None
    # Review state (populated from sidecar)
    review_status: ReviewStatus = ReviewStatus.PENDING
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    revision_notes: str | None = None
    rejection_note: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
