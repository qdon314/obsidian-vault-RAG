from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

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
GROUNDEDNESS_JUDGE_VERSION = "groundedness_v1"
GROUNDEDNESS_JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

You will be given:
- a QUERY
- RETRIEVED CONTEXT CHUNKS (each has a chunk_id)
- a GENERATED ANSWER

Your job:
1) Determine whether the context contains enough information to answer the query (SHOULD_ABSTAIN).
2) Extract the key factual claims from the generated answer (be concise; focus on substantive claims).
3) For each claim, mark it as SUPPORTED or UNSUPPORTED by the provided context.
4) For each SUPPORTED claim, provide exactly one chunk_id as evidence and a short quote (<= 20 words) from that chunk.
5) For each UNSUPPORTED claim, explain briefly why it is unsupported (missing from context, contradicts context, etc).

Important:
- You must judge support ONLY using the retrieved context below.
- Do not use outside knowledge.
- If the answer includes extra facts not in context, those are hallucinations (UNSUPPORTED claims).

QUERY:
{query}

RETRIEVED CONTEXT:
{context_chunks}

GENERATED ANSWER:
{generated_answer}

Respond with ONLY a JSON object:
{{
  "should_abstain": <true|false>,
  "supported_claims": <int>,
  "unsupported_claims": <int>,
  "claims": [
    {{
      "claim": "<string>",
      "supported": <true|false>,
      "chunk_id": "<chunk_id or null>",
      "quote": "<short quote or null>",
      "note": "<brief explanation>"
    }}
  ]
}}"""

@dataclass(frozen=True, slots=True)
class GoldJudgeResult:
    correctness: float | None = None
    completeness: float | None = None
    relevance: float | None = None
    hallucination_severity: float | None = None
    reasoning: str | None = None


@dataclass(frozen=True, slots=True)
class GroundednessJudgeResult:
    should_abstain: bool | None = None
    supported_claims: int | None = None
    unsupported_claims: int | None = None
    claims: list[dict[str, Any]] | None = None

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
        return GoldJudgeResult(
            correctness=_to_float(data.get("correctness")),
            completeness=_to_float(data.get("completeness")),
            relevance=_to_float(data.get("relevance")),
            hallucination_severity=_to_float(data.get("hallucination_severity") or data.get("hallucination")),
            reasoning=_to_str(data.get("reasoning")),
        )
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
        return GroundednessJudgeResult(
            should_abstain=_to_bool(data.get("should_abstain")),
            supported_claims=_to_int(data.get("supported_claims")),
            unsupported_claims=_to_int(data.get("unsupported_claims")),
            claims=data.get("claims") if isinstance(data.get("claims"), list) else None,
        )
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
