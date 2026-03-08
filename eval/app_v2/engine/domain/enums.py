# eval/app_v2/engine/domain/enums.py
from enum import StrEnum


class DiagnosticCode(StrEnum):
    NO_CLEAR_FAILURE             = "no_clear_failure"
    RETRIEVAL_MISS               = "retrieval_miss"
    RETRIEVAL_PARTIAL            = "retrieval_partial"
    RERANK_DROPPED_RELEVANT      = "rerank_dropped_relevant"
    RERANK_DEGRADED_RANK         = "rerank_degraded_rank"
    PACKING_OMITTED_RELEVANT     = "packing_omitted_relevant"
    PACKING_TRUNCATED_RELEVANT   = "packing_truncated_relevant"
    GROUNDED_ANSWER              = "grounded_answer"
    UNSUPPORTED_ANSWER           = "unsupported_answer"
    BAD_ABSTAIN_ON_ANSWERABLE    = "bad_abstain_on_answerable"
    FAILED_ABSTAIN_ON_UNANSWERABLE = "failed_abstain_on_unanswerable"
    TRACE_MISSING                = "trace_missing"
    DATA_INSUFFICIENT            = "data_insufficient"


class Severity(StrEnum):
    OK       = "ok"
    MINOR    = "minor"
    MODERATE = "moderate"
    CRITICAL = "critical"


class RetrievalStatus(StrEnum):
    HIT     = "hit"
    PARTIAL = "partial"
    MISS    = "miss"
    UNKNOWN = "unknown"


class RerankStatus(StrEnum):
    IMPROVED = "improved"
    NEUTRAL  = "neutral"
    DEGRADED = "degraded"
    ABSENT   = "absent"
    UNKNOWN  = "unknown"


class PackingStatus(StrEnum):
    COMPLETE  = "complete"
    TRUNCATED = "truncated"
    OMITTED   = "omitted"
    ABSENT    = "absent"
    UNKNOWN   = "unknown"


class GenerationStatus(StrEnum):
    GROUNDED          = "grounded"
    UNSUPPORTED       = "unsupported"
    ABSTAINED         = "abstained"
    FAILED_TO_ABSTAIN = "failed_to_abstain"
    ABSENT            = "absent"
    UNKNOWN           = "unknown"


class DeltaDirection(StrEnum):
    IMPROVED     = "improved"
    REGRESSED    = "regressed"
    UNCHANGED    = "unchanged"
    INSUFFICIENT = "insufficient"


class ComparisonClassification(StrEnum):
    IMPROVED          = "improved"
    REGRESSED         = "regressed"
    MIXED             = "mixed"
    UNCHANGED         = "unchanged"
    INSUFFICIENT_DATA = "insufficient_data"
