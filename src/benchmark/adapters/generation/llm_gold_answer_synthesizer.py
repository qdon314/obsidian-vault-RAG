"""Stage 6: LLM-based gold answer synthesiser.

Generates a gold answer and answer-core rubric (acceptable variants,
required points, forbidden errors) for a validated query, grounded in its
critical and supporting evidence spans.

On LLM error or JSON parse failure, returns ``GoldAnswer(gold_answer="")``
without raising. The runner skips records with an empty ``gold_answer``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from benchmark.domain.models import (
    EvidenceEntry,
    EvidenceSet,
    GoldAnswer,
    StageConfig,
    ValidatedQuery,
)
from benchmark.ports.llm_client import LLMClient

logger = logging.getLogger(__name__)

_SYNTHESIZE_PROMPT = """\
You are a benchmark curator for an NRC nuclear regulatory RAG system.

Given a query and its supporting evidence, produce a concise gold answer and
answer-core rubric. The gold answer must be grounded exclusively in the
evidence provided — do not introduce information from outside the evidence.

## Query
{query}

## Query class
{query_class}

## Critical evidence
{critical_evidence}

## Supporting evidence
{supporting_evidence}

Respond ONLY with a JSON object in this exact format:
{{
  "gold_answer": "<one or two sentence factual answer>",
  "acceptable_answer_variants": ["<paraphrase 1>", "<paraphrase 2>"],
  "required_points": ["<fact that must be present>"],
  "forbidden_errors": ["<common wrong answer>"]
}}
"""

_EMPTY = GoldAnswer(gold_answer="")


def _format_entries(entries: tuple[EvidenceEntry, ...]) -> str:
    if not entries:
        return "(none)"
    return "\n".join(f"- [{e.citation}] {e.text}" for e in entries)


def _parse_response(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening and closing fence lines.
        inner = [l for l in lines if not l.startswith("```")]
        text = "\n".join(inner).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


class LLMGoldAnswerSynthesizer:
    """Gold answer synthesiser backed by an ``LLMClient``.

    Structural subtype of ``GoldAnswerSynthesizer`` protocol.
    """

    def __init__(self, llm_client: LLMClient, config: StageConfig) -> None:
        self._client = llm_client
        self._config = config

    def synthesize(
        self,
        query: ValidatedQuery,
        evidence: EvidenceSet,
    ) -> GoldAnswer:
        prompt = _SYNTHESIZE_PROMPT.format(
            query=query.query,
            query_class=query.query_class.value,
            critical_evidence=_format_entries(evidence.critical),
            supporting_evidence=_format_entries(evidence.supporting),
        )
        logger.debug("Synthesising gold answer for candidate_id=%s", query.candidate_id)
        try:
            response = self._client.complete(prompt, self._config)
        except Exception as exc:
            logger.warning(
                "LLM call failed for gold answer synthesis (candidate_id=%s): %s",
                query.candidate_id,
                exc,
            )
            return _EMPTY

        data = _parse_response(response.text)
        if data is None:
            logger.warning(
                "Gold answer JSON parse failed for candidate_id=%s; raw: %.200s",
                query.candidate_id,
                response.text,
            )
            return _EMPTY

        gold_answer = str(data.get("gold_answer", "")).strip()
        if not gold_answer:
            logger.warning(
                "Empty gold_answer in LLM response for candidate_id=%s",
                query.candidate_id,
            )
            return _EMPTY

        return GoldAnswer(
            gold_answer=gold_answer,
            acceptable_answer_variants=tuple(
                str(v) for v in data.get("acceptable_answer_variants", [])
            ),
            required_points=tuple(str(p) for p in data.get("required_points", [])),
            forbidden_errors=tuple(str(e) for e in data.get("forbidden_errors", [])),
        )
