"""Stage 3: Template-based query generation with LLM paraphrasing.

Generates citation-lookup queries from ``RegulatoryUnit`` records by
building template questions from unit metadata and asking the LLM to
produce realistic paraphrases.  Only ``CITATION_LOOKUP`` is supported
in M3; other query classes raise ``ValueError``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmark.domain.enums import QueryClass
from benchmark.domain.models import (
    EvidenceSet,
    QueryCandidate,
    RegulatoryUnit,
    StageConfig,
)
from benchmark.ports.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

_PARAPHRASE_PROMPT_TEMPLATE = """\
You are a regulatory question writer. Given a regulatory citation and its
normative content, produce realistic paraphrases of a lookup question.

## Regulatory unit

Citation: {citation}
Normative statement: {canonical_statement}
Key entities: {entities}

## Template question

{template_question}

## Instructions

Produce 2-3 realistic paraphrases of the template question. Each
paraphrase should:
- Reference the citation (e.g. "10 CFR 50.46(b)(1)")
- Ask about the same normative content
- Use different wording, sentence structure, or framing
- Sound like a question a regulatory professional would ask

Respond with a JSON array of strings, each being one paraphrased question.
Respond ONLY with the JSON array, no other text.
"""


class TemplateQueryGenerator:
    """Stage 3: Template-based citation-lookup query generator.

    Constructs a template question from the unit's citation and
    canonical statement, then uses the LLM to produce realistic
    paraphrases.  Falls back to the template itself if the LLM
    response is malformed.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: StageConfig,
    ) -> None:
        self._llm_client = llm_client
        self._config = config

    def generate(
        self,
        unit: RegulatoryUnit,
        evidence: EvidenceSet,
        query_class: QueryClass,
    ) -> list[QueryCandidate]:
        """Generate query candidates for a regulatory unit.

        Only ``CITATION_LOOKUP`` is supported in M3.
        """
        if query_class != QueryClass.CITATION_LOOKUP:
            msg = (
                f"Only CITATION_LOOKUP is supported in M3, "
                f"got {query_class.value!r}"
            )
            raise ValueError(msg)

        template = self._build_template(unit)
        prompt = self._build_prompt(unit, template)
        response = self._llm_client.complete(prompt, self._config)
        queries = self._parse_paraphrases(response, template, unit.unit_id)

        critical_span_ids = tuple(e.span_id for e in evidence.critical)

        return [
            QueryCandidate(
                candidate_id=f"qc_{unit.unit_id}_{query_class.value}_{i}",
                unit_id=unit.unit_id,
                query=q,
                query_class=query_class,
                source_citations=(unit.citation,),
                evidence_span_ids=critical_span_ids,
                corpus_snapshot_id=unit.corpus_snapshot_id,
            )
            for i, q in enumerate(queries)
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_template(unit: RegulatoryUnit) -> str:
        """Build a template question from unit metadata."""
        topic = unit.canonical_statement or "the regulatory requirements"
        return f"What does {unit.citation} require regarding {topic}?"

    @staticmethod
    def _build_prompt(unit: RegulatoryUnit, template: str) -> str:
        """Build the paraphrase prompt for the LLM."""
        return _PARAPHRASE_PROMPT_TEMPLATE.format(
            citation=unit.citation,
            canonical_statement=unit.canonical_statement or "(not classified)",
            entities=", ".join(unit.entities) if unit.entities else "(none)",
            template_question=template,
        )

    @staticmethod
    def _parse_paraphrases(
        response: LLMResponse,
        template: str,
        unit_id: str,
    ) -> list[str]:
        """Parse LLM response into a list of query strings.

        Falls back to the template question if parsing fails.
        """
        parsed = _parse_json_response(response.text, unit_id)

        if parsed is None or not isinstance(parsed, list):
            logger.warning(
                "Malformed LLM response for unit %s; falling back to template",
                unit_id,
            )
            return [template]

        queries: list[str] = []
        for item in parsed:
            if isinstance(item, str) and item.strip():
                queries.append(item.strip())

        if not queries:
            logger.warning(
                "Empty paraphrase list for unit %s; falling back to template",
                unit_id,
            )
            return [template]

        return queries


def _parse_json_response(text: str, unit_id: str) -> Any | None:
    """Parse JSON from LLM response, stripping code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines)

    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        logger.warning(
            "Failed to parse JSON for unit %s: %.200s",
            unit_id,
            text,
        )
        return None
