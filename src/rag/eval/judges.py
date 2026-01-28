from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from dataclasses_json import DataClassJsonMixin
from openai import OpenAI

from rag.utils.enum_parse import parse_enum

logger = logging.getLogger(__name__)


class ClaimRole(StrEnum):
    """Classification of a claim's importance to the answer."""

    CORE = "core"
    PERIPHERAL = "peripheral"

# !--------  Update version when prompt changes! -----------!
GOLD_JUDGE_VERSION = "gold_v1"
GOLD_JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

Given a query, the expected answer (ground truth), and the system's generated answer, evaluate the quality on these dimensions:

1. CORRECTNESS (0-5): Is the generated answer factually correct compared to the expected answer?
   - 5: Completely correct, all key facts match
   - 3: Mostly correct, minor errors or omissions
   - 0: Completely incorrect or unrelated

2. COMPLETENESS (0-5): Does the generated answer cover all important points from the expected answer?
   - 5: Covers all key points comprehensively
   - 3: Covers main points but misses some details
   - 0: Misses most or all key points

3. RELEVANCE (0-5): Is the answer relevant to the query?
   - 5: Directly answers the query
   - 3: Partially answers the query
   - 0: Completely off-topic

4. HALLUCINATION (0-5): Does the answer contain information not supported by the expected answer?
   - 0: No hallucinations, all info is grounded
   - 3: Some unsupported claims
   - 5: Significant fabricated information

QUERY: {query}

EXPECTED ANSWER:
{expected_answer}

GENERATED ANSWER:
{generated_answer}

Respond with ONLY a JSON object:
{{
  "correctness": <0-5>,
  "completeness": <0-5>,
  "relevance": <0-5>,
  "hallucination_severity": <0-5>,
  "reasoning": "<brief explanation>"
}}"""


# !--------  Update version when prompt changes! -----------!
GROUNDEDNESS_JUDGE_VERSION = "groundedness_v3"
GROUNDEDNESS_JUDGE_PROMPT = """You are an expert groundedness evaluator for a Retrieval-Augmented Generation (RAG) system.

You will be given:
- a QUERY
- RETRIEVED CONTEXT CHUNKS (each has a chunk_id)
- a GENERATED ANSWER

Your task is to analyze whether the GENERATED ANSWER is supported by the RETRIEVED CONTEXT.

Definitions:

1) answerable_from_context:
   True only if the retrieved context contains enough information to directly answer the query as asked.

2) evidence_bounded:
   True only if EVERY substantive claim in the answer is either:
     (a) explicitly supported by the retrieved context, OR
     (b) an explicit and accurate statement of uncertainty or insufficiency
         (e.g., "the context does not say", "cannot be determined from the provided context").

3) Claim:
   A concise factual assertion made by the answer.
   Ignore stylistic, hedging, or conversational language.

4) Claim role:
   - CORE:
     A claim that directly answers the query or is essential for the answer to be correct.
     If false or unsupported, the answer would be materially wrong or misleading.
   - PERIPHERAL:
     A claim that provides additional detail, explanation, or framing.
     If false or unsupported, the core answer would still be intact.

Role assignment rules:
- Every claim MUST be assigned exactly one role.
- Prefer CORE if unsure.
- Claims that enumerate items, dates, quantities, or statuses requested by the query
  are almost always CORE.

Evaluation steps:

1) Determine answerable_from_context.
2) Extract all substantive claims from the generated answer.
3) Assign each claim a role (CORE or PERIPHERAL).
4) For each claim, determine whether it is SUPPORTED by the retrieved context.
   - Statements of insufficiency are SUPPORTED if the context indeed lacks the information.
5) If any CORE claim is UNSUPPORTED, evidence_bounded MUST be false.
6) Count supported and unsupported claims.

Important constraints:
- Judge support ONLY using the retrieved context.
- Do NOT use outside knowledge.
- Do NOT infer or assume missing facts.
- Be conservative: if support is ambiguous, mark UNSUPPORTED.

QUERY:
{query}

RETRIEVED CONTEXT:
{context_chunks}

GENERATED ANSWER:
{generated_answer}

Respond with ONLY a JSON object:

{{
  "answerable_from_context": <true|false>,
  "evidence_bounded": <true|false>,
  "supported_claims": <int>,
  "unsupported_claims": <int>,
  "claims": [
    {{
      "claim": "<string>",
      "role": "<core|peripheral>",
      "supported": <true|false>,
      "chunk_id": "<chunk_id or null>",
      "quote": "<short quote (≤20 words) or null>",
      "note": "<brief explanation>"
    }}
  ]
}}"""


@dataclass(frozen=True, slots=True)
class GoldJudgeResult(DataClassJsonMixin):
    correctness: float | None = None
    completeness: float | None = None
    relevance: float | None = None
    hallucination_severity: float | None = None
    reasoning: str | None = None

    @classmethod
    def from_llm_dict(cls, data: dict[str, Any]) -> GoldJudgeResult:
        """Parse from LLM output dict which may use alternate keys."""
        return cls(
            correctness=_to_float(data.get("correctness")),
            completeness=_to_float(data.get("completeness")),
            relevance=_to_float(data.get("relevance")),
            hallucination_severity=_to_float(
                data.get("hallucination_severity") or data.get("hallucination")
            ),
            reasoning=_to_str(data.get("reasoning")),
        )


@dataclass(frozen=True, slots=True)
class AnswerClaim(DataClassJsonMixin):
    claim: str = ""
    role: ClaimRole = ClaimRole.CORE
    supported: bool = False
    chunk_id: str | None = None
    quote: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class GroundednessJudgeResult(DataClassJsonMixin):
    answerable_from_context: bool | None = None
    evidence_bounded: bool | None = None
    supported_claims: int | None = None
    unsupported_claims: int | None = None
    claims: list[AnswerClaim] | None = None

    @classmethod
    def from_llm_dict(cls, data: dict[str, Any]) -> GroundednessJudgeResult:
        """Parse from LLM output dict with lenient type coercion."""
        raw_claims = data.get("claims")
        parsed_claims: list[AnswerClaim] | None = None
        if isinstance(raw_claims, list):
            parsed_claims = [
                _parse_answer_claim(c) for c in raw_claims if isinstance(c, dict)
            ]

        return cls(
            answerable_from_context=_to_bool(data.get("answerable_from_context")),
            evidence_bounded=_to_bool(data.get("evidence_bounded")),
            supported_claims=_to_int(data.get("supported_claims")),
            unsupported_claims=_to_int(data.get("unsupported_claims")),
            claims=parsed_claims,
        )

def make_gold_prompt(*, query: str, expected_answer: str, generated_answer: str) -> str:
    return GOLD_JUDGE_PROMPT.format(
        query=query,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

def make_groundedness_prompt(*, query: str, context_chunks: str, generated_answer: str) -> str:
    return GROUNDEDNESS_JUDGE_PROMPT.format(
        query=query,
        context_chunks=context_chunks,
        generated_answer=generated_answer,
    )

def _safe_json_loads(text: str) -> dict[str, Any] | None:
    """
    Attempts to parse JSON even if the model wrapped it in code fences.
    """
    text = text.strip()
    if text.startswith("```"):
        # strip leading/trailing fences
        text = text.strip("`")
        # sometimes starts with json\n
        text = text.replace("json\n", "", 1).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def evaluate_vs_expected_answer(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
) -> GoldJudgeResult:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        data = _safe_json_loads(content) or {}
        return GoldJudgeResult.from_llm_dict(data)
    except Exception as e:
        logger.error("Gold-judge error: %s", e)
        return GoldJudgeResult()


def evaluate_groundedness(
    *,
    client: OpenAI,
    model: str,
    prompt: str,
) -> GroundednessJudgeResult:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator. Output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content or ""
        data = _safe_json_loads(content) or {}
        return GroundednessJudgeResult.from_llm_dict(data)
    except Exception as e:
        logger.error("Groundedness-judge error: %s", e)
        return GroundednessJudgeResult()


def _to_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _to_int(x: Any) -> int | None:
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None


def _to_bool(x: Any) -> bool | None:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0"}:
            return False
    return None


def _to_str(x: Any) -> str | None:
    return x if isinstance(x, str) else None


def _parse_answer_claim(data: dict[str, Any]) -> AnswerClaim:
    """Parse an AnswerClaim from LLM output dict with role enum conversion."""
    return AnswerClaim(
        claim=data.get("claim", ""),
        role=parse_enum(ClaimRole, data.get("role")) or ClaimRole.CORE,
        supported=_to_bool(data.get("supported")) or False,
        chunk_id=data.get("chunk_id"),
        quote=data.get("quote"),
        note=data.get("note"),
    )
