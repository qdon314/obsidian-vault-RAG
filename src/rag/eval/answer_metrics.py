from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from dataclasses_json import DataClassJsonMixin

_WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)

def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))

def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

def _norm_0_5(x: float) -> float:
    return _clamp01(x / 5.0)

def _severity_to_goodness(x: float) -> float:
    # hallucination_severity: 0..5 higher worse
    return _clamp01(1.0 - (x / 5.0))


@dataclass(frozen=True, slots=True)
class AnswerQualityMetrics(DataClassJsonMixin):
    semantic_similarity: float | None = None  # 0..1 if present

    correctness: float | None = None          # 0..5 higher better
    completeness: float | None = None         # 0..5 higher better
    relevance: float | None = None            # 0..5 higher better
    hallucination_severity: float | None = None  # 0..5 higher worse

    is_correct: bool | None = None
    is_abstained: bool | None = None
    answerable_from_context: bool | None = None
    evidence_bounded: bool | None = None
    has_hallucination: bool | None = None

    supported_claims: int | None = None
    unsupported_claims: int | None = None
    citation_coverage: float | None = None

    answer_length: int | None = None  # word count
    num_citations: int | None = None

    quality_score: float | None = None  # 0..1

    @staticmethod
    def compute(
        *,
        answer_text: str,
        citations: Sequence[Any],
        is_abstained: bool,
        semantic_similarity: float | None = None,
        # gold judge (optional)
        correctness: float | None = None,
        completeness: float | None = None,
        relevance: float | None = None,
        hallucination_severity: float | None = None,
        # groundedness judge (optional)
        answerable_from_context: bool | None = None,
        evidence_bounded: bool | None = None,
        supported_claims: int | None = None,
        unsupported_claims: int | None = None,
    ) -> AnswerQualityMetrics:
        answer_len = _word_count(answer_text)
        num_cits = len(citations)

        # Derived booleans
        is_correct = (correctness >= 4.0) if correctness is not None else None
        has_hallucination = None
        if unsupported_claims is not None:
            has_hallucination = unsupported_claims > 0
        elif hallucination_severity is not None:
            has_hallucination = hallucination_severity >= 1.0

        # Citation coverage
        citation_coverage = None
        if supported_claims is not None and unsupported_claims is not None:
            total = supported_claims + unsupported_claims
            if total > 0:
                citation_coverage = _clamp01(supported_claims / total)

        # Quality score (0..1), renormalized over available components
        quality_score = _compute_quality_score(
            correctness=correctness,
            completeness=completeness,
            relevance=relevance,
            hallucination_severity=hallucination_severity,
            citation_coverage=citation_coverage,
            is_correct=is_correct,
            has_hallucination=has_hallucination,
            is_abstained=is_abstained,
            answerable_from_context=answerable_from_context,
            evidence_bounded=evidence_bounded,
        )

        return AnswerQualityMetrics(
            semantic_similarity=semantic_similarity,
            correctness=correctness,
            completeness=completeness,
            relevance=relevance,
            hallucination_severity=hallucination_severity,
            is_correct=is_correct,
            is_abstained=is_abstained,
            answerable_from_context=answerable_from_context,
            evidence_bounded=evidence_bounded,
            has_hallucination=has_hallucination,
            supported_claims=supported_claims,
            unsupported_claims=unsupported_claims,
            citation_coverage=citation_coverage,
            answer_length=answer_len,
            num_citations=num_cits,
            quality_score=quality_score,
        )


def _compute_quality_score(
    *,
    correctness: float | None,
    completeness: float | None,
    relevance: float | None,
    hallucination_severity: float | None,
    citation_coverage: float | None,
    is_correct: bool | None,
    has_hallucination: bool | None,
    is_abstained: bool,
    answerable_from_context: bool | None,
    evidence_bounded: bool | None,
) -> float | None:
    comps: list[tuple[float, float]] = []  # (value0_1, weight)

    # weights: retrieval-minded
    if correctness is not None:
        comps.append((_norm_0_5(correctness), 0.45))
    if hallucination_severity is not None:
        comps.append((_severity_to_goodness(hallucination_severity), 0.30))
    if citation_coverage is not None:
        # groundedness bonus from citations
        comps.append((_clamp01(citation_coverage), 0.15))
    if completeness is not None:
        comps.append((_norm_0_5(completeness), 0.07))
    if relevance is not None:
        comps.append((_norm_0_5(relevance), 0.03))

    if not comps:
        return None

    total_w = sum(w for _, w in comps)
    score = sum(v * (w / total_w) for v, w in comps)

    # Guardrails
    # If context was answerable but answer is not evidence-bounded, cap hard
    if answerable_from_context is True and evidence_bounded is False and not is_abstained:
        score = min(score, 0.35)

    # If context was not answerable but model answered confidently (not abstained
    # and not evidence-bounded), also cap
    if answerable_from_context is False and evidence_bounded is False and not is_abstained:
        score = min(score, 0.35)

    # Incorrect caps
    if is_correct is False:
        score = min(score, 0.35)

    # Hallucination caps (unsupported claims)
    if has_hallucination is True:
        score = min(score, 0.55)

    # Strong floors for correct+grounded
    if is_correct is True and has_hallucination is False:
        score = max(score, 0.75)

    # Bonus: evidence-bounded answers get a floor
    if evidence_bounded is True and has_hallucination is False:
        score = max(score, 0.60)

    return _clamp01(float(score))
