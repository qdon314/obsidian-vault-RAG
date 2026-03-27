"""Stage 5a: Deterministic auto-validation of query candidates.

Applies rule-based checks to ``QueryCandidate`` records. No LLM calls —
purely deterministic. Low-scoring items are flagged, not dropped.

``refine_evidence()`` is a pass-through stub in M3; per-query evidence
refinement is implemented by the M4 LLM validator.
"""

from __future__ import annotations

import re

from benchmark.domain.enums import QueryClass
from benchmark.domain.models import (
    EvidenceSet,
    QueryCandidate,
    ValidatedQuery,
    ValidationResult,
)

_CFR_CITATION_PATTERN = re.compile(r"\d+\s+CFR\s+\d+\.\d+")

_BANNED_PREFIXES = (
    "summarize",
    "list all",
    "describe everything",
    "what are all the rules",
)

# Query classes considered "narrow" — evidence set size is bounded.
_NARROW_QUERY_CLASSES = frozenset(
    {
        QueryClass.CITATION_LOOKUP,
        QueryClass.NARROW_FACTUAL,
    }
)


class DeterministicValidator:
    """Stage 5a: Deterministic validation of query candidates.

    Each failing check adds a descriptive flag string to the result.
    ``is_valid`` is ``True`` only when no *blocking* flags are raised.
    Flags listed in ``warn_only_flags`` are still recorded but do not
    cause ``is_valid=False``.
    """

    def __init__(
        self,
        *,
        known_queries: list[str] | None = None,
        max_query_length: int = 500,
        min_query_length: int = 10,
        max_evidence_spans: int = 6,
        warn_only_flags: frozenset[str] | None = None,
    ) -> None:
        self._warn_only_flags: frozenset[str] = warn_only_flags or frozenset()
        self._max_query_length = max_query_length
        self._min_query_length = min_query_length
        self._max_evidence_spans = max_evidence_spans
        # Normalize known queries for case-insensitive dedup.
        self._known_normalized: set[str] = set()
        for q in known_queries or []:
            self._known_normalized.add(self._normalize(q))

    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        """Run all deterministic checks on a candidate."""
        flags: list[str] = []

        self._check_citation_format(candidate, flags)
        self._check_duplicate(candidate, flags)
        self._check_banned_prefix(candidate, flags)
        self._check_length(candidate, flags)
        self._check_evidence_non_empty(candidate, flags)
        self._check_evidence_size(candidate, flags)
        self._check_snapshot_id(candidate, flags)

        # Track this query for future dedup checks.
        self._known_normalized.add(self._normalize(candidate.query))

        blocking = [f for f in flags if f not in self._warn_only_flags]
        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid=len(blocking) == 0,
            flags=tuple(flags),
        )

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet:
        """Return evidence unchanged — M3 stub."""
        return evidence

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_citation_format(
        self,
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Query must contain a well-formed CFR citation."""
        if not _CFR_CITATION_PATTERN.search(candidate.query):
            flags.append("no_cfr_citation")

    def _check_duplicate(
        self,
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Query must not duplicate a previously seen query."""
        normalized = self._normalize(candidate.query)
        if normalized in self._known_normalized:
            flags.append("duplicate_query")

    def _check_banned_prefix(
        self,
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Query must not start with a banned broad verb/phrase."""
        lower = candidate.query.lower().strip()
        for prefix in _BANNED_PREFIXES:
            if lower.startswith(prefix):
                flags.append(f"banned_prefix:{prefix}")
                break

    def _check_length(
        self,
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Query length must be within bounds."""
        length = len(candidate.query)
        if length < self._min_query_length:
            flags.append("too_short")
        if length > self._max_query_length:
            flags.append("too_long")

    @staticmethod
    def _check_evidence_non_empty(
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Evidence span IDs must be non-empty."""
        if not candidate.evidence_span_ids:
            flags.append("no_evidence")

    def _check_evidence_size(
        self,
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Narrow query classes must have bounded evidence sets."""
        if (
            candidate.query_class in _NARROW_QUERY_CLASSES
            and len(candidate.evidence_span_ids) > self._max_evidence_spans
        ):
            flags.append("too_many_evidence_spans")

    @staticmethod
    def _check_snapshot_id(
        candidate: QueryCandidate,
        flags: list[str],
    ) -> None:
        """Corpus snapshot ID must be populated."""
        if not candidate.corpus_snapshot_id:
            flags.append("missing_snapshot_id")

    @staticmethod
    def _normalize(query: str) -> str:
        """Normalize query for case-insensitive, whitespace-normalized comparison."""
        return " ".join(query.lower().split())
