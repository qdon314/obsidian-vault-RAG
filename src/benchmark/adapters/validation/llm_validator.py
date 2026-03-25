"""Stage 5a: LLM-based validation adapter for query candidates.

Composes with ``DeterministicValidator`` — deterministic checks run first,
and LLM scoring only fires on candidates that pass.  This avoids wasting
LLM calls on structurally invalid candidates.

Five scoring dimensions (each 0.0-1.0):
- plausibility  : would a real user ask this?
- boundedness   : answerable from supplied evidence?
- ambiguity     : multiple materially different readings? (high = unambiguous)
- specificity   : is the target fact clear?
- leakage       : mirrors statutory phrasing too directly? (high = no leakage)
"""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmark.adapters.validation.deterministic_validator import DeterministicValidator
from benchmark.domain.models import (
    EvidenceEntry,
    EvidenceSet,
    QueryCandidate,
    ValidatedQuery,
    ValidationResult,
)
from benchmark.ports.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# -- Default thresholds -------------------------------------------------------

_DEFAULT_THRESHOLDS: dict[str, float] = {
    "plausibility": 0.6,
    "boundedness": 0.6,
    "ambiguity": 0.7,
    "specificity": 0.6,
    "leakage": 0.7,
}

_SCORE_DIMENSIONS = tuple(_DEFAULT_THRESHOLDS.keys())

# -- Prompt templates ---------------------------------------------------------

_SCORE_PROMPT = """\
You are a benchmark quality evaluator for an NRC regulatory RAG system.

Rate the following query candidate on five dimensions, each on a scale 0.0-1.0.

## Query
{query}

## Evidence span IDs available
{evidence_span_ids}

## Query class
{query_class}

## Scoring dimensions

- plausibility (0.0-1.0): Would a real NRC compliance engineer or licensee ask \
this question? 0 = nonsensical or artificial; 1 = highly realistic.
- boundedness (0.0-1.0): Is the question answerable from the listed evidence \
spans without external context? 0 = requires unlisted context; 1 = fully \
self-contained.
- ambiguity (0.0-1.0): Does the question have a single clear interpretation? \
1 = unambiguous; 0 = multiple materially different readings.
- specificity (0.0-1.0): Is the target fact or regulatory provision clear? \
1 = highly specific; 0 = vague or open-ended.
- leakage (0.0-1.0): Does the query avoid copying statutory language verbatim? \
1 = original phrasing; 0 = mirrors the regulation text too directly.

Respond ONLY with a JSON object:
{{"plausibility": 0.0, "boundedness": 0.0, "ambiguity": 0.0, "specificity": 0.0, "leakage": 0.0}}
"""

_REFINE_PROMPT = """\
You are a benchmark evidence reviewer for an NRC regulatory RAG system.

A query has passed validation. Determine which evidence tiers are relevant
for answering this specific query.

## Query
{query}

## Query class
{query_class}

## Current evidence (by tier)

Critical spans:
{critical_spans}

Supporting spans:
{supporting_spans}

Contextual spans:
{contextual_spans}

## Instructions

For the query class "{query_class}", decide whether any spans from supporting
or contextual tiers should be promoted to critical, or whether any critical
spans are actually not needed to answer this specific question.

Rules:
- citation_lookup: keep only spans that contain the specific fact being asked for
- cross_reference: contextual spans from linked units may be promoted to critical
- Other classes: use best judgment

Respond ONLY with a JSON object listing span IDs by tier:
{{"critical": ["span_id_1"], "supporting": ["span_id_2"], "contextual": ["span_id_3"]}}

Only include span IDs that appear in the input. Do not invent new spans.
"""


class LLMValidator:
    """Stage 5a: LLM-based validation with model-scored quality dimensions.

    Composes with ``DeterministicValidator`` — deterministic checks run first,
    then LLM scoring fires only for candidates that pass deterministic checks.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Any,  # StageConfig — typed as Any to avoid circular import
        *,
        deterministic: DeterministicValidator,
        score_thresholds: dict[str, float] | None = None,
    ) -> None:
        self._llm_client = llm_client
        self._config = config
        self._deterministic = deterministic
        self._thresholds = dict(_DEFAULT_THRESHOLDS)
        if score_thresholds:
            self._thresholds.update(score_thresholds)

    def validate(self, candidate: QueryCandidate) -> ValidationResult:
        """Run deterministic checks first; if passed, run LLM scoring.

        Candidates failing deterministic checks get ``scores={}`` — no LLM
        call is made.  Low LLM scores add flags like ``"low_plausibility:0.3"``.
        Malformed LLM responses add ``"llm_parse_error"`` flag.
        """
        det_result = self._deterministic.validate(candidate)

        if not det_result.is_valid:
            # Skip LLM — keep deterministic flags, no scores.
            return det_result

        # Run LLM scoring.
        prompt = _SCORE_PROMPT.format(
            query=candidate.query,
            evidence_span_ids=", ".join(candidate.evidence_span_ids) or "(none)",
            query_class=candidate.query_class.value,
        )
        response = self._llm_client.complete(prompt, self._config)
        scores, extra_flags = self._parse_scores(response, candidate.candidate_id)

        flags: list[str] = list(det_result.flags) + extra_flags
        for dim, score in scores.items():
            threshold = self._thresholds.get(dim, 0.6)
            if score < threshold:
                flags.append(f"low_{dim}:{score:.2f}")

        return ValidationResult(
            candidate_id=candidate.candidate_id,
            is_valid=len(flags) == 0,
            flags=tuple(flags),
            scores=scores,
        )

    def refine_evidence(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> EvidenceSet:
        """Narrow unit-level evidence for a specific query via LLM.

        Falls back to returning evidence unchanged if the LLM response is
        malformed or omits a tier.
        """
        critical_ids = {e.span_id for e in evidence.critical}
        supporting_ids = {e.span_id for e in evidence.supporting}
        contextual_ids = {e.span_id for e in evidence.contextual}

        prompt = _REFINE_PROMPT.format(
            query=query.query,
            query_class=query.query_class.value,
            critical_spans=_format_spans(evidence.critical),
            supporting_spans=_format_spans(evidence.supporting),
            contextual_spans=_format_spans(evidence.contextual),
        )
        response = self._llm_client.complete(prompt, self._config)
        refined = self._parse_refinement(
            response, query.candidate_id, critical_ids | supporting_ids | contextual_ids
        )

        if refined is None:
            return evidence

        # Build span lookup by span_id from all tiers.
        all_spans: dict[str, EvidenceEntry] = {
            e.span_id: e for e in (*evidence.critical, *evidence.supporting, *evidence.contextual)
        }

        def _pick(ids: list[str]) -> tuple[EvidenceEntry, ...]:
            return tuple(all_spans[sid] for sid in ids if sid in all_spans)

        return EvidenceSet(
            unit_id=evidence.unit_id,
            critical=_pick(refined.get("critical", [])),
            supporting=_pick(refined.get("supporting", [])),
            contextual=_pick(refined.get("contextual", [])),
        )

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scores(
        response: LLMResponse,
        candidate_id: str,
    ) -> tuple[dict[str, float], list[str]]:
        """Parse LLM score response. Returns (scores, extra_flags)."""
        cleaned = _strip_code_fence(response.text)
        try:
            result = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse LLM score JSON for candidate %s: %.200s",
                candidate_id,
                response.text,
            )
            return {}, ["llm_parse_error"]

        if not isinstance(result, dict):
            logger.warning(
                "Expected JSON object for candidate %s scores, got %s",
                candidate_id,
                type(result).__name__,
            )
            return {}, ["llm_parse_error"]

        scores: dict[str, float] = {}
        for dim in _SCORE_DIMENSIONS:
            raw = result.get(dim)
            if isinstance(raw, (int, float)):
                scores[dim] = float(raw)
            else:
                logger.warning(
                    "Missing/invalid score for dimension %r in candidate %s",
                    dim,
                    candidate_id,
                )
                # Treat missing dimension as parse error.
                return {}, ["llm_parse_error"]

        return scores, []

    @staticmethod
    def _parse_refinement(
        response: LLMResponse,
        candidate_id: str,
        valid_span_ids: set[str],
    ) -> dict[str, list[str]] | None:
        """Parse LLM evidence refinement response."""
        cleaned = _strip_code_fence(response.text)
        try:
            result = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse refinement JSON for candidate %s: %.200s",
                candidate_id,
                response.text,
            )
            return None

        if not isinstance(result, dict):
            return None

        # Filter out any hallucinated span IDs.
        refined: dict[str, list[str]] = {}
        for tier in ("critical", "supporting", "contextual"):
            raw_ids = result.get(tier, [])
            if isinstance(raw_ids, list):
                refined[tier] = [sid for sid in raw_ids if sid in valid_span_ids]
            else:
                refined[tier] = []

        return refined


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _strip_code_fence(text: str) -> str:
    """Strip leading/trailing markdown code fences from LLM output."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)
    return cleaned


def _format_spans(entries: tuple[EvidenceEntry, ...]) -> str:
    """Format evidence entries for prompt inclusion."""
    if not entries:
        return "(none)"
    return "\n".join(f"- {e.span_id}: {e.text[:100]}" for e in entries)
