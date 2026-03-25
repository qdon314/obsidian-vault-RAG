"""Stage 1b: LLM-based semantic classification of regulatory units.

Takes Stage 1a ``RegulatoryUnit`` records and returns enriched copies with
semantic fields populated.  Does **not** alter ``unit_id``, ``spans``, or
span boundaries — identity is immutable after Stage 1a.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from typing import Any

from benchmark.domain.enums import UnitKind
from benchmark.domain.models import RegulatoryUnit, StageConfig
from benchmark.ports.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_VALID_KINDS = frozenset(k.value for k in UnitKind)

_CLASSIFICATION_PROMPT_TEMPLATE = """\
You are a regulatory classification assistant. Given a regulatory text unit,
classify it and extract structured fields.

## Regulatory unit

Section: {section_id}
Subsection chain: {subsection_chain}
Text:
{text}

## Instructions

Respond with a JSON object containing exactly these fields:

- "kind": one of {valid_kinds}
- "canonical_statement": a single normalized sentence stating the regulatory fact
- "entities": list of named regulatory entities (e.g., "licensee", "Commission")
- "value": numeric threshold if present (as string), or null
- "conditions": list of qualifying conditions

Respond ONLY with the JSON object, no other text.
"""

# Confidence threshold: below this, flag the unit for human review.
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Safety cap on combined span text sent to the LLM.
# 1 token ≈ 4 chars; 100 000 chars ≈ 25 000 tokens, leaving headroom for the
# system prompt and response within a 30 000-TPM budget.
DEFAULT_MAX_CONTEXT_CHARS = 100_000


class LLMExtractor:
    """Stage 1b: LLM-based semantic classification of regulatory units.

    Takes Stage 1a ``RegulatoryUnit`` records and returns enriched copies
    with populated semantic fields.  Units where classification confidence
    is below threshold are returned with ``metadata["low_confidence"] = True``.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: StageConfig,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ) -> None:
        self._llm_client = llm_client
        self._config = config
        self._confidence_threshold = confidence_threshold
        self._max_context_chars = max_context_chars

    def classify(self, units: list[RegulatoryUnit]) -> list[RegulatoryUnit]:
        """Classify each unit and return enriched copies.

        Uses ``dataclasses.replace()`` to produce new instances with
        populated semantic fields.
        """
        return [self._classify_one(unit) for unit in units]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _classify_one(self, unit: RegulatoryUnit) -> RegulatoryUnit:
        """Classify a single regulatory unit via LLM."""
        prompt, truncated = self._build_prompt(unit)
        if truncated:
            unit = replace(unit, metadata={**unit.metadata, "truncated": True})
        response = self._llm_client.complete(prompt, self._config)
        return self._apply_classification(unit, response)

    def _build_prompt(self, unit: RegulatoryUnit) -> tuple[str, bool]:
        """Build the classification prompt for a single unit.

        Returns:
            (prompt, truncated) — truncated is True when span text exceeded
            ``max_context_chars`` and was cut to fit.
        """
        combined_text = "\n\n".join(span.text for span in unit.spans)
        truncated = False
        if len(combined_text) > self._max_context_chars:
            combined_text = combined_text[: self._max_context_chars]
            truncated = True
            logger.warning(
                "Unit %s span text truncated to %d chars for LLM classification "
                "(original ~%d chars). Consider using a higher-context model.",
                unit.unit_id,
                self._max_context_chars,
                len("\n\n".join(span.text for span in unit.spans)),
            )
        prompt = _CLASSIFICATION_PROMPT_TEMPLATE.format(
            section_id=unit.parent_section_id,
            subsection_chain=" > ".join(unit.subsection_chain) or "(root)",
            text=combined_text,
            valid_kinds=", ".join(sorted(_VALID_KINDS)),
        )
        return prompt, truncated

    def _apply_classification(
        self,
        unit: RegulatoryUnit,
        response: LLMResponse,
    ) -> RegulatoryUnit:
        """Parse LLM response and apply classification to the unit."""
        parsed = self._parse_response(response.text, unit.unit_id)

        if parsed is None:
            logger.warning(
                "Malformed LLM response for unit %s; flagging low_confidence",
                unit.unit_id,
            )
            return replace(
                unit,
                metadata={**unit.metadata, "low_confidence": True},
            )

        kind = self._resolve_kind(parsed.get("kind"), unit)
        metadata = {**unit.metadata}

        # Flag low confidence if the LLM returned an unexpected kind
        # relative to the structural cue from Stage 1a.
        if kind != unit.kind and unit.kind == UnitKind.CROSS_REFERENCE:
            metadata["low_confidence"] = True

        return replace(
            unit,
            kind=kind,
            canonical_statement=parsed.get("canonical_statement"),
            entities=tuple(parsed.get("entities") or ()),
            value=parsed.get("value"),
            conditions=tuple(parsed.get("conditions") or ()),
            metadata=metadata,
        )

    @staticmethod
    def _parse_response(text: str, unit_id: str) -> dict[str, Any] | None:
        """Parse JSON from LLM response text.

        Returns ``None`` if the response is not valid JSON.
        """
        # Strip markdown code fences if present.
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            # Remove first and last lines (fences).
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            parsed = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse JSON for unit %s: %.200s",
                unit_id,
                text,
            )
            return None

        if not isinstance(parsed, dict):
            logger.warning(
                "LLM response for unit %s is not a JSON object",
                unit_id,
            )
            return None

        return parsed

    @staticmethod
    def _resolve_kind(raw_kind: Any, unit: RegulatoryUnit) -> UnitKind:
        """Resolve a kind string to a ``UnitKind`` enum value.

        Falls back to the unit's existing kind if the value is invalid.
        """
        if isinstance(raw_kind, str) and raw_kind in _VALID_KINDS:
            return UnitKind(raw_kind)
        logger.warning(
            "Invalid kind '%s' for unit %s; keeping existing kind %s",
            raw_kind,
            unit.unit_id,
            unit.kind,
        )
        return unit.kind
