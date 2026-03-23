"""Stage 2: LLM-based evidence tier assignment.

For each ``RegulatoryUnit``, classifies each span into a tier (critical,
supporting, contextual) based on its importance to the unit's normative
content.  Also considers neighboring spans from the same parent section
as candidates for the contextual tier.

Evidence tiers are **unit-relative** — see
``docs/decisions/adr-evidence-tier-semantics.md``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmark.domain.enums import EvidenceTier
from benchmark.domain.models import (
    BenchmarkSourceSpan,
    EvidenceEntry,
    EvidenceSet,
    RegulatoryUnit,
    StageConfig,
)
from benchmark.ports.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_VALID_TIERS = frozenset(t.value for t in EvidenceTier)

_EVIDENCE_PROMPT_TEMPLATE = """\
You are a regulatory evidence analyst. Given a regulatory unit and its spans,
assign each span to an evidence tier.

## Tier definitions

- **critical**: Removing this span makes the unit's normative content \
incomprehensible.
- **supporting**: Removing this span degrades completeness but the core \
obligation/threshold remains clear.
- **contextual**: Nearby material that aids interpretation but is neither \
critical nor supporting.

## Regulatory unit

Unit ID: {unit_id}
Citation: {citation}
Kind: {kind}

## Spans to classify

{spans_block}

## Neighboring spans (same section, not part of this unit)

{neighbors_block}

## Instructions

Respond with a JSON object mapping span indices to tier assignments.
Unit spans are indexed 0..N-1 under "unit_spans".
Neighbor spans (if any) that are contextually relevant should be listed
under "contextual_neighbors" as a list of indices.

Example:
{{"unit_spans": {{"0": "critical", "1": "supporting"}}, \
"contextual_neighbors": [0]}}

Respond ONLY with the JSON object, no other text.
"""


class LLMEvidenceBuilder:
    """Stage 2: LLM-based evidence tier assignment.

    For each unit, asks the LLM to classify each span into a tier based
    on its importance to the unit's normative content.  Also considers
    neighboring spans (from the same parent section) as candidates for
    the contextual tier.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: StageConfig,
        all_spans: list[BenchmarkSourceSpan],
    ) -> None:
        self._llm_client = llm_client
        self._config = config
        # Index spans by parent_section_id for neighbor lookup.
        self._section_spans: dict[str, list[BenchmarkSourceSpan]] = {}
        for span in all_spans:
            self._section_spans.setdefault(span.parent_section_id, []).append(span)

    def build(self, unit: RegulatoryUnit) -> EvidenceSet:
        """Build tiered evidence for a single regulatory unit."""
        neighbors = self._find_neighbors(unit)
        prompt = self._build_prompt(unit, neighbors)
        response = self._llm_client.complete(prompt, self._config)
        return self._parse_evidence(unit, neighbors, response)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _find_neighbors(
        self,
        unit: RegulatoryUnit,
    ) -> list[BenchmarkSourceSpan]:
        """Find spans in the same section that are not part of this unit."""
        unit_keys = {(s.citation_key, s.char_start, s.char_end) for s in unit.spans}
        section_spans = self._section_spans.get(unit.parent_section_id, [])
        return [
            s for s in section_spans if (s.citation_key, s.char_start, s.char_end) not in unit_keys
        ]

    def _build_prompt(
        self,
        unit: RegulatoryUnit,
        neighbors: list[BenchmarkSourceSpan],
    ) -> str:
        """Build the evidence classification prompt."""
        spans_block = "\n".join(
            f"[{i}] ({span.citation}) {span.text}" for i, span in enumerate(unit.spans)
        )
        if neighbors:
            neighbors_block = "\n".join(
                f"[{i}] ({span.citation}) {span.text}" for i, span in enumerate(neighbors)
            )
        else:
            neighbors_block = "(none)"

        return _EVIDENCE_PROMPT_TEMPLATE.format(
            unit_id=unit.unit_id,
            citation=unit.citation,
            kind=unit.kind.value,
            spans_block=spans_block,
            neighbors_block=neighbors_block,
        )

    def _parse_evidence(
        self,
        unit: RegulatoryUnit,
        neighbors: list[BenchmarkSourceSpan],
        response: LLMResponse,
    ) -> EvidenceSet:
        """Parse LLM response into an EvidenceSet.

        Falls back to all-critical if parsing fails.
        """
        parsed = self._parse_response_json(response.text, unit.unit_id)

        if parsed is None:
            logger.warning(
                "Malformed LLM response for unit %s; defaulting all spans to critical",
                unit.unit_id,
            )
            return self._fallback_all_critical(unit)

        return self._build_evidence_set(unit, neighbors, parsed)

    def _build_evidence_set(
        self,
        unit: RegulatoryUnit,
        neighbors: list[BenchmarkSourceSpan],
        parsed: dict[str, Any],
    ) -> EvidenceSet:
        """Construct EvidenceSet from parsed LLM tier assignments."""
        critical: list[EvidenceEntry] = []
        supporting: list[EvidenceEntry] = []
        contextual: list[EvidenceEntry] = []

        unit_assignments = parsed.get("unit_spans", {})

        for i, span in enumerate(unit.spans):
            raw_tier = unit_assignments.get(str(i))
            tier = self._resolve_tier(raw_tier, fallback=EvidenceTier.CRITICAL)
            entry = self._span_to_entry(
                span=span,
                span_id=f"{unit.unit_id}_{i}",
                tier=tier,
            )
            if tier == EvidenceTier.CRITICAL:
                critical.append(entry)
            elif tier == EvidenceTier.SUPPORTING:
                supporting.append(entry)
            else:
                contextual.append(entry)

        # Add neighbor spans classified as contextual.
        contextual_indices = parsed.get("contextual_neighbors", [])
        for idx in contextual_indices:
            if isinstance(idx, int) and 0 <= idx < len(neighbors):
                entry = self._span_to_entry(
                    span=neighbors[idx],
                    span_id=f"{unit.unit_id}_n{idx}",
                    tier=EvidenceTier.CONTEXTUAL,
                )
                contextual.append(entry)

        return EvidenceSet(
            unit_id=unit.unit_id,
            critical=tuple(critical),
            supporting=tuple(supporting),
            contextual=tuple(contextual),
        )

    @staticmethod
    def _fallback_all_critical(unit: RegulatoryUnit) -> EvidenceSet:
        """Fallback: assign all unit spans to the critical tier."""
        entries = tuple(
            EvidenceEntry(
                span_id=f"{unit.unit_id}_{i}",
                citation=span.citation,
                text=span.text,
                char_start=span.char_start,
                char_end=span.char_end,
                chunk_ids=span.chunk_ids_overlapping_span,
                tier=EvidenceTier.CRITICAL,
            )
            for i, span in enumerate(unit.spans)
        )
        return EvidenceSet(unit_id=unit.unit_id, critical=entries)

    @staticmethod
    def _span_to_entry(
        *,
        span: BenchmarkSourceSpan,
        span_id: str,
        tier: EvidenceTier,
    ) -> EvidenceEntry:
        """Map a BenchmarkSourceSpan to an EvidenceEntry."""
        return EvidenceEntry(
            span_id=span_id,
            citation=span.citation,
            text=span.text,
            char_start=span.char_start,
            char_end=span.char_end,
            chunk_ids=span.chunk_ids_overlapping_span,
            tier=tier,
        )

    @staticmethod
    def _resolve_tier(
        raw: Any,
        *,
        fallback: EvidenceTier = EvidenceTier.CRITICAL,
    ) -> EvidenceTier:
        """Resolve a tier string to an EvidenceTier enum value."""
        if isinstance(raw, str) and raw in _VALID_TIERS:
            return EvidenceTier(raw)
        return fallback

    @staticmethod
    def _parse_response_json(text: str, unit_id: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response, stripping code fences."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning("Failed to parse JSON for unit %s: %.200s", unit_id, text)
            return None

        if not isinstance(parsed, dict):
            return None

        return parsed
