"""Stage 3: Unanswerable query generation for benchmark safety evaluation.

Generates queries that *cannot* be answered from the NRC corpus.  These are
safety-critical for nuclear regulatory evaluation — the system must know when
to abstain rather than hallucinate.

Three strategies:
1. **Near-miss** — ask about a related but uncovered subsection.
2. **Domain-boundary** — reference adjacent-domain regulations (OSHA, EPA, DOE).
3. **Fabricated-citation** — use a plausible but non-existent CFR reference.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Any, ClassVar

from benchmark.domain.enums import QueryClass
from benchmark.domain.models import (
    EvidenceSet,
    QueryCandidate,
    RegulatoryUnit,
    StageConfig,
)
from benchmark.ports.llm_client import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

# -- Strategy names -----------------------------------------------------------

STRATEGY_NEAR_MISS = "near_miss"
STRATEGY_DOMAIN_BOUNDARY = "domain_boundary"
STRATEGY_FABRICATED_CITATION = "fabricated_citation"

_STRATEGIES = (STRATEGY_NEAR_MISS, STRATEGY_DOMAIN_BOUNDARY, STRATEGY_FABRICATED_CITATION)

# -- Prompt templates ---------------------------------------------------------

_NEAR_MISS_PROMPT = """\
You are a regulatory question writer creating unanswerable questions for a
benchmark dataset.  The goal is to produce a realistic question that is
closely related to a real regulatory unit but asks about content that does
NOT exist in the corpus.

## Real regulatory unit

Citation: {citation}
Section: {parent_section_id}
Subsection chain: {subsection_chain}
Content summary: {canonical_statement}

## Instructions

Write 1 realistic question that:
- References the same section ({parent_section_id}) or a plausible adjacent
  subsection that does NOT exist (e.g. a paragraph letter/number beyond what
  the section actually contains).
- Sounds like something a regulatory professional would genuinely ask.
- Cannot be answered from the provided unit or its section.

Also provide a brief reason why the question is unanswerable.

Respond ONLY with a JSON object:
{{"query": "...", "reason": "..."}}
"""

_DOMAIN_BOUNDARY_PROMPT = """\
You are a regulatory question writer creating unanswerable questions for an
NRC (10 CFR) benchmark dataset.  The corpus only contains NRC regulations
from 10 CFR.

## Adjacent regulation for reference

{adjacent_citation}

## Real NRC unit for context

Citation: {citation}
Topic: {canonical_statement}

## Instructions

Write 1 realistic question that:
- Asks about the adjacent-domain regulation ({adjacent_citation}), NOT the
  NRC corpus.
- Sounds like a natural question from someone working in nuclear energy who
  might confuse regulatory jurisdictions.
- Cannot be answered from 10 CFR.

Also provide a brief reason why the question is unanswerable.

Respond ONLY with a JSON object:
{{"query": "...", "reason": "..."}}
"""

_FABRICATED_CITATION_PROMPT = """\
You are a regulatory question writer creating unanswerable questions for an
NRC (10 CFR) benchmark dataset.

## Real regulatory unit for context

Citation: {citation}
Section: {parent_section_id}
Subsection chain: {subsection_chain}

## Instructions

Write 1 realistic question that:
- References a plausible but NON-EXISTENT 10 CFR citation.  For example,
  if the real section is 10 CFR 50.46, you might reference 10 CFR 50.46(b)(6)
  when only (b)(1)-(5) exist, or 10 CFR 50.48(f) when 50.48 only goes to (e).
- Sounds like a real regulatory lookup question.
- Cannot be answered because the cited provision does not exist.

Also provide the fabricated citation and a brief reason.

Respond ONLY with a JSON object:
{{"query": "...", "fabricated_citation": "...", "reason": "..."}}
"""


class UnanswerableGenerator:
    """Generate unanswerable queries using three strategies.

    Implements ``QueryGenerator`` protocol for ``QueryClass.UNANSWERABLE``
    only.  Strategy selection is deterministic given a unit_id (hash-based)
    for reproducibility.
    """

    ADJACENT_DOMAINS: ClassVar[tuple[str, ...]] = (
        "29 CFR 1910",   # OSHA general industry
        "40 CFR 61",     # EPA NESHAP
        "10 CFR 830",    # DOE nuclear safety
        "49 CFR 173",    # DOT hazmat transport
    )

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
        """Generate unanswerable query candidates.

        Raises ``ValueError`` if *query_class* is not ``UNANSWERABLE``.
        """
        if query_class != QueryClass.UNANSWERABLE:
            msg = (
                f"UnanswerableGenerator only supports UNANSWERABLE, "
                f"got {query_class.value!r}"
            )
            raise ValueError(msg)

        strategy = self._select_strategy(unit.unit_id)
        prompt = self._build_prompt(unit, strategy)
        response = self._llm_client.complete(prompt, self._config)
        parsed = self._parse_response(response, unit.unit_id, strategy)

        if parsed is None:
            # Fallback: produce a template-based unanswerable query
            return [self._fallback_candidate(unit, strategy, 0)]

        candidates: list[QueryCandidate] = []
        query_text = parsed.get("query", "")
        reason = parsed.get("reason", f"unanswerable via {strategy}")

        if not query_text or not isinstance(query_text, str):
            return [self._fallback_candidate(unit, strategy, 0)]

        candidates.append(
            QueryCandidate(
                candidate_id=f"qc_{unit.unit_id}_unanswerable_{strategy}_0",
                unit_id=unit.unit_id,
                query=query_text.strip(),
                query_class=QueryClass.UNANSWERABLE,
                source_citations=(),
                evidence_span_ids=(),
                corpus_snapshot_id=unit.corpus_snapshot_id,
                metadata={
                    "is_unanswerable": True,
                    "unanswerable_reason": reason,
                    "unanswerable_strategy": strategy,
                },
            )
        )
        return candidates

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _select_strategy(self, unit_id: str) -> str:
        """Deterministically select a strategy based on unit_id hash."""
        h = int(hashlib.sha256(unit_id.encode()).hexdigest(), 16)
        return _STRATEGIES[h % len(_STRATEGIES)]

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _build_prompt(self, unit: RegulatoryUnit, strategy: str) -> str:
        canonical = unit.canonical_statement or "the regulatory requirements"
        subsection_str = " > ".join(unit.subsection_chain)

        if strategy == STRATEGY_NEAR_MISS:
            return _NEAR_MISS_PROMPT.format(
                citation=unit.citation,
                parent_section_id=unit.parent_section_id,
                subsection_chain=subsection_str,
                canonical_statement=canonical,
            )
        if strategy == STRATEGY_DOMAIN_BOUNDARY:
            adjacent = self._select_adjacent_domain(unit.unit_id)
            return _DOMAIN_BOUNDARY_PROMPT.format(
                adjacent_citation=adjacent,
                citation=unit.citation,
                canonical_statement=canonical,
            )
        # STRATEGY_FABRICATED_CITATION
        return _FABRICATED_CITATION_PROMPT.format(
            citation=unit.citation,
            parent_section_id=unit.parent_section_id,
            subsection_chain=subsection_str,
        )

    def _select_adjacent_domain(self, unit_id: str) -> str:
        """Deterministically pick an adjacent domain from the seed list."""
        h = int(hashlib.sha256(f"domain_{unit_id}".encode()).hexdigest(), 16)
        return self.ADJACENT_DOMAINS[h % len(self.ADJACENT_DOMAINS)]

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(
        response: LLMResponse,
        unit_id: str,
        strategy: str,
    ) -> dict[str, Any] | None:
        """Parse JSON object from LLM response, stripping code fences."""
        cleaned = response.text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines)

        try:
            result = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            logger.warning(
                "Failed to parse JSON for unanswerable query "
                "(unit=%s, strategy=%s): %.200s",
                unit_id,
                strategy,
                response.text,
            )
            return None

        if not isinstance(result, dict):
            logger.warning(
                "Expected JSON object for unanswerable query "
                "(unit=%s, strategy=%s), got %s",
                unit_id,
                strategy,
                type(result).__name__,
            )
            return None

        return result

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback_candidate(
        unit: RegulatoryUnit,
        strategy: str,
        index: int,
    ) -> QueryCandidate:
        """Produce a template-based unanswerable query as fallback."""
        if strategy == STRATEGY_NEAR_MISS:
            query = (
                f"What requirements apply to subsection (z) of "
                f"{unit.parent_section_id} regarding additional criteria?"
            )
            reason = f"Subsection (z) of {unit.parent_section_id} does not exist"
        elif strategy == STRATEGY_DOMAIN_BOUNDARY:
            query = (
                f"What does 29 CFR 1910 require for nuclear facility "
                f"safety procedures related to {unit.parent_section_id}?"
            )
            reason = "29 CFR 1910 (OSHA) is outside the NRC 10 CFR corpus"
        else:
            query = (
                f"What does 10 CFR {unit.parent_section_id}(z)(99) "
                f"require regarding special conditions?"
            )
            reason = (
                f"10 CFR {unit.parent_section_id}(z)(99) is a "
                f"fabricated citation that does not exist"
            )

        return QueryCandidate(
            candidate_id=f"qc_{unit.unit_id}_unanswerable_{strategy}_{index}",
            unit_id=unit.unit_id,
            query=query,
            query_class=QueryClass.UNANSWERABLE,
            source_citations=(),
            evidence_span_ids=(),
            corpus_snapshot_id=unit.corpus_snapshot_id,
            metadata={
                "is_unanswerable": True,
                "unanswerable_reason": reason,
                "unanswerable_strategy": strategy,
                "is_fallback": True,
            },
        )
