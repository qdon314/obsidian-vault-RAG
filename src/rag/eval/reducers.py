from __future__ import annotations

from enum import StrEnum

from rag.eval.judges import ClaimRole, GoldJudgeResult, GroundednessJudgeResult

REDUCER_VERSION = "reducers_v1"


class OutcomeLabel(StrEnum):
    SUCCESS_GROUNDED = "success_grounded"
    SUCCESS_UNGROUNDED = "success_ungrounded"
    SAFE_MISS = "safe_miss"
    UNSAFE_MISS = "unsafe_miss"
    ABSTAIN_OK = "abstain_ok"
    ABSTAIN_BAD = "abstain_bad"


def hallucination_severity(*, groundedness: GroundednessJudgeResult | None) -> float | None:
    """
    Deterministic hallucination severity 0..5 (higher worse), derived from groundedness.

    Uses claim roles when available; otherwise falls back to unsupported_claims count.
    """
    if groundedness is None:
        return None

    # Fast path if count present
    if groundedness.unsupported_claims == 0:
        return 0.0

    # Fatal: context unanswerable but model was not evidence-bounded
    if groundedness.answerable_from_context is False and groundedness.evidence_bounded is False:
        return 5.0

    claims = groundedness.claims or []
    if claims:
        unsupported = [c for c in claims if not c.supported]
        if not unsupported:
            return 0.0

        unsupported_core = sum(1 for c in unsupported if c.role == ClaimRole.CORE)
        unsupported_periph = sum(1 for c in unsupported if c.role == ClaimRole.PERIPHERAL)

        if unsupported_core >= 1:
            return 4.0

        # only peripheral unsupported
        if unsupported_periph == 1:
            return 1.0
        if 2 <= unsupported_periph <= 3:
            return 2.0

        # many peripheral issues or evidence-bounded false w/o core violations (rare)
        return 3.0

    # Fallback: counts but no claims
    if groundedness.unsupported_claims is None:
        return None

    n = groundedness.unsupported_claims
    if n <= 2:
        return 2.0
    if n <= 4:
        return 3.0
    return 4.0


def outcome_label(
    *, gold: GoldJudgeResult | None, groundedness: GroundednessJudgeResult | None
) -> OutcomeLabel | None:
    """
    Optional diagnosis label combining gold task success + groundedness boundedness.
    """
    if gold is None or groundedness is None:
        return None

    a = groundedness.answerable_from_context
    eb = groundedness.evidence_bounded
    if a is None or eb is None:
        return None

    if a is False:
        return OutcomeLabel.ABSTAIN_OK if eb is True else OutcomeLabel.ABSTAIN_BAD

    c, comp, r = gold.correctness, gold.completeness, gold.relevance
    if c is None or comp is None or r is None:
        return None

    task_success = (c >= 4.0) and (comp >= 4.0) and (r >= 4.0)

    if task_success and eb is True:
        return OutcomeLabel.SUCCESS_GROUNDED
    if task_success and eb is False:
        return OutcomeLabel.SUCCESS_UNGROUNDED
    if (not task_success) and eb is True:
        return OutcomeLabel.SAFE_MISS
    return OutcomeLabel.UNSAFE_MISS
