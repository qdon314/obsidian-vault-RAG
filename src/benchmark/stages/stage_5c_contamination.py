"""Stage 5c: Contamination probe for benchmark records.

For each ``BenchmarkRecord`` with a non-empty ``gold_answer``, runs the
production generator model with an empty context window and judges the
ungrounded answer against the gold answer using the eval harness judge.

Records where the model achieves ``correctness / 5.0 >= threshold`` are
flagged as contaminated for the given ``model_id``.  Contamination is
model-version-specific: ``contamination_probes`` stores one entry per
model so that a model upgrade does not invalidate prior probes.

All LLM calls route through the ``LLMClient`` port.  The gold judge prompt
and ``GoldJudgeResult`` are imported from ``rag.eval.judges`` (eval layer
is accessible from the benchmark package; ``rag.adapters`` and ``rag.ports``
are not).
"""

from __future__ import annotations

import dataclasses
import json
import logging
from typing import Any

from benchmark.domain.models import BenchmarkRecord, StageConfig
from benchmark.ports.llm_client import LLMClient
from rag.eval.judges import GoldJudgeResult, make_gold_prompt

logger = logging.getLogger(__name__)

_GENERATION_PROMPT = """\
Answer the following question without any additional context.

Question: {query}"""

_CONTAMINATION_THRESHOLD_DEFAULT = 0.7


def _safe_json_loads(text: str) -> dict[str, Any] | None:
    """Parse JSON, tolerating markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = [line for line in lines if not line.startswith("```")]
        text = "\n".join(inner).strip()
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _score_0_1(result: GoldJudgeResult) -> float:
    """Normalise correctness from 0-5 scale to 0-1."""
    if result.correctness is None:
        return 0.0
    return max(0.0, min(1.0, result.correctness / 5.0))


def run_contamination_probe(
    records: list[BenchmarkRecord],
    llm_client: LLMClient,
    config: StageConfig,
    model_id: str,
    *,
    contamination_threshold: float = _CONTAMINATION_THRESHOLD_DEFAULT,
) -> list[BenchmarkRecord]:
    """Return records with ``contamination_probes[model_id]`` populated.

    Records without a ``gold_answer`` are passed through unchanged (WARNING
    logged).  Records already probed for ``model_id`` are skipped (idempotent
    on resume).  Judge parse failure → treated as ``False`` (not contaminated)
    with a WARNING log; does not raise.

    Args:
        records: BenchmarkRecords from stage_6 (gold answers populated).
        llm_client: LLM client routed through the benchmark port.
        config: StageConfig used for both the generation and judge calls.
        model_id: Identifier for the model being probed (e.g. ``"gpt-4o-2025-01-01"``).
        contamination_threshold: Score (0-1) at or above which a record is
            considered contaminated.  Default 0.7 per the design doc.

    Returns:
        The same records list with ``contamination_probes[model_id]`` set on
        every record that has a ``gold_answer``.
    """
    updated: list[BenchmarkRecord] = []

    for record in records:
        # Skip records without a gold answer.
        if not record.gold_answer:
            logger.warning(
                "Skipping contamination probe for qid=%s: gold_answer is empty",
                record.qid,
            )
            updated.append(record)
            continue

        # Idempotency: skip already-probed records.
        if model_id in record.contamination_probes:
            logger.debug(
                "qid=%s already probed for model_id=%s; skipping",
                record.qid,
                model_id,
            )
            updated.append(record)
            continue

        contaminated = _probe_record(
            record,
            llm_client,
            config,
            model_id,
            contamination_threshold,
        )
        updated.append(
            dataclasses.replace(
                record,
                contamination_probes={
                    **record.contamination_probes,
                    model_id: contaminated,
                },
            )
        )

    return updated


def _probe_record(
    record: BenchmarkRecord,
    llm_client: LLMClient,
    config: StageConfig,
    model_id: str,
    threshold: float,
) -> bool:
    """Run generation + judge calls for one record. Returns True if contaminated."""
    # Step 1: generate an ungrounded answer.
    generation_prompt = _GENERATION_PROMPT.format(query=record.query)
    try:
        gen_response = llm_client.complete(generation_prompt, config)
        ungrounded_answer = gen_response.text.strip()
    except Exception as exc:
        logger.warning(
            "Generation call failed for qid=%s (model_id=%s): %s — "
            "treating as not contaminated",
            record.qid,
            model_id,
            exc,
        )
        return False

    # Step 2: judge the ungrounded answer against the gold answer.
    assert record.gold_answer is not None  # narrowed above in caller
    judge_prompt = make_gold_prompt(
        query=record.query,
        expected_answer=record.gold_answer,
        generated_answer=ungrounded_answer,
        query_type=record.query_class.value,
        difficulty=record.difficulty,
    )
    try:
        judge_response = llm_client.complete(judge_prompt, config)
        judge_data = _safe_json_loads(judge_response.text) or {}
        result = GoldJudgeResult.from_llm_dict(judge_data)
    except Exception as exc:
        logger.warning(
            "Judge call failed for qid=%s (model_id=%s): %s — "
            "treating as not contaminated",
            record.qid,
            model_id,
            exc,
        )
        return False

    score = _score_0_1(result)
    contaminated = score >= threshold
    logger.info(
        "qid=%s model_id=%s correctness=%.2f score_0_1=%.3f contaminated=%s",
        record.qid,
        model_id,
        result.correctness if result.correctness is not None else float("nan"),
        score,
        contaminated,
    )
    return contaminated
