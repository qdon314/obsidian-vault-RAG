#!/usr/bin/env python3
"""
Evaluation harness for RAG system.

This script:
1. Loads evaluation queries from JSONL
2. Runs them through the RAG pipeline
3. Computes retrieval and answer quality metrics
4. Generates detailed reports
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from rag.app.container import Container
from rag.app.query_runner import run_query
from rag.domain.models import Answer, Chunk
from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.judges import (
    GOLD_JUDGE_VERSION,
    GROUNDEDNESS_JUDGE_VERSION,
    evaluate_groundedness,
    evaluate_vs_expected_answer,
    make_gold_prompt,
    make_groundedness_prompt,
)
from rag.eval.metrics import semantic_similarity, summarize
from rag.eval.models import EvalAggregates, EvalResult, EvalRun, EvalRunMeta, RetrievalResult
from rag.eval.schema import EvalQuery
from rag.ports.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ----------------------------
# Helpers
# ----------------------------

def format_context_chunks(chunks: Sequence[Chunk], max_chars_per_chunk: int = 1200) -> str:
    lines: list[str] = []
    for ch in chunks:
        text = ch.text[:max_chars_per_chunk].replace("\n", " ").strip()
        lines.append(f"[chunk_id={ch.chunk_id}] {text}")
    return "\n".join(lines)


def load_eval_queries(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            queries.append(EvalQuery.from_dict(json.loads(line)))
    return queries


def run_retrieval_eval(
    query: EvalQuery,
    retriever: Retriever,
    top_k: int = 10,
) -> RetrievalResult:
    candidates = retriever.retrieve(query.query, top_k=top_k)
    retrieved_ids = [cand.chunk.chunk_id for cand in candidates]
    return RetrievalResult(
        qid=query.qid,
        retrieved_chunk_ids=tuple(retrieved_ids),
        relevant_chunk_ids=query.relevant_chunk_ids,
    )

def evaluate_answer_quality(
    *,
    query: EvalQuery,
    answer: Answer,
    retrieved_chunks: Sequence[Chunk],
    client: OpenAI | None,
    judge_model: str,
    embedder: Any | None,
    use_llm_judge: bool,
) -> AnswerQualityMetrics:
    sem_sim: float | None = None
    if embedder and query.expected_answer:
        sem_sim = semantic_similarity(query.expected_answer, answer.text, embedder)

    # Judge outputs (optional)
    correctness = completeness = relevance = hallucination_severity = None
    answerable_from_context = evidence_bounded = supported_claims = unsupported_claims = None

    if use_llm_judge and client and judge_model:
        # Groundedness judging first (determines if answer is evidence-bounded)
        if retrieved_chunks:
            ctx = format_context_chunks(retrieved_chunks)
            grounded_prompt = make_groundedness_prompt(
                query=query.query,
                context_chunks=ctx,
                generated_answer=answer.text,
            )
            gr = evaluate_groundedness(
                client=client,
                model=judge_model,
                prompt=grounded_prompt,
            )
            answerable_from_context = gr.answerable_from_context
            evidence_bounded = gr.evidence_bounded
            supported_claims = gr.supported_claims
            unsupported_claims = gr.unsupported_claims

        # Gold-answer judging (only if expected answer exists)
        if query.expected_answer:
            gold_prompt = make_gold_prompt(
                query=query.query,
                expected_answer=query.expected_answer,
                generated_answer=answer.text,
            )
            gold = evaluate_vs_expected_answer(
                client=client,
                model=judge_model,
                prompt=gold_prompt,
            )
            correctness = gold.correctness
            completeness = gold.completeness
            relevance = gold.relevance
            hallucination_severity = gold.hallucination_severity

    return AnswerQualityMetrics.compute(
        answer_text=answer.text,
        citations=answer.citations,
        semantic_similarity=sem_sim,
        correctness=correctness,
        completeness=completeness,
        relevance=relevance,
        hallucination_severity=hallucination_severity,
        answerable_from_context=answerable_from_context,
        evidence_bounded=evidence_bounded,
        supported_claims=supported_claims,
        unsupported_claims=unsupported_claims,
    )


# ----------------------------
# Main eval
# ----------------------------

def run_full_eval(
    *,
    eval_queries: list[EvalQuery],
    container: Container,
    queries_path: str | None,
    top_k: int = 10,
    keep_k: int | None = None,
    token_budget: int = 1500,
    run_generation: bool = False,
    use_llm_judge: bool = False,
    judge_client: OpenAI | None = None,
    judge_model: str | None = None,
    score_ids: str = "reranked",  # "retrieved" or "reranked"
) -> EvalRun:
    container.store.load()
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC)

    if use_llm_judge and (judge_client is None or not judge_model):
        raise ValueError("use_llm_judge=True requires judge_client and judge_model")

    results: list[EvalResult] = []

    for q in eval_queries:
        query_filter = q.get_filter()

        # --- Retrieval only ---
        if not run_generation:
            cands = container.retriever.retrieve(q.query, top_k=top_k, where=query_filter)
            retrieved_ids = [c.chunk.chunk_id for c in cands]
            retrieval_result = RetrievalResult(
                qid=q.qid,
                retrieved_chunk_ids=tuple(retrieved_ids),
                relevant_chunk_ids=q.relevant_chunk_ids,
            )
            results.append(
                EvalResult(
                    qid=q.qid,
                    query=q.query,
                    retrieval_result=retrieval_result,
                    answer=None,
                    answer_metrics=None,
                    query_type=q.query_type,
                    difficulty=q.difficulty,
                    is_unanswerable=q.is_unanswerable,
                    latency_ms=None,
                    trace_id=None,
                )
            )
            continue

        # --- Full pipeline ---
        run = run_query(
            query=q.query,
            retriever=container.retriever,
            reranker=container.reranker,
            context_builder=container.context_builder,
            generator=container.generator,
            logger=container.logger,
            top_k=top_k,
            keep_k=keep_k,
            token_budget=token_budget,
            where=query_filter,
        )

        # retrieval ids used for retrieval metrics
        if score_ids == "retrieved":
            chosen_ids = tuple(run.retrieved_chunk_ids)
        elif score_ids == "reranked":
            chosen_ids = tuple(run.reranked_chunk_ids)
        else:
            raise ValueError("score_ids must be 'retrieved' or 'reranked'")

        answer: Answer = run.answer
        retrieval_result = RetrievalResult(
            qid=q.qid,
            retrieved_chunk_ids=chosen_ids,
            relevant_chunk_ids=q.relevant_chunk_ids,
        )

        answer_metrics = evaluate_answer_quality(
            query=q,
            answer=answer,
            retrieved_chunks=run.context_pack.chunks,
            client=judge_client if use_llm_judge else None,
            judge_model=judge_model or "",
            embedder=getattr(container, "embedder", None),
            use_llm_judge=use_llm_judge,
        )

        results.append(
            EvalResult(
                qid=q.qid,
                query=q.query,
                retrieval_result=retrieval_result,
                answer=answer,
                answer_metrics=answer_metrics,
                query_type=q.query_type,
                difficulty=q.difficulty,
                is_unanswerable=q.is_unanswerable,
                latency_ms=getattr(run, "latency_ms", None),
                trace_id=getattr(run, "trace_id", None),
            )
        )

    aggregates = aggregate_results(results)

    meta = EvalRunMeta(
        run_id=run_id,
        started_at=started_at,
        queries_path=queries_path,
        top_k=top_k,
        keep_k=keep_k,
        token_budget=token_budget,
        run_generation=run_generation,
        use_llm_judge=use_llm_judge,
        generator_model=getattr(container.generator, "model_name", None),
        embedder_model=getattr(getattr(container, "embedder", None), "model_name", None),
        reranker_name=getattr(container.reranker, "name", None),
        judge_model=judge_model,
        # add these fields to EvalRunMeta (or put into extra)
        gold_judge_version=GOLD_JUDGE_VERSION if use_llm_judge else None,
        groundedness_judge_version=GROUNDEDNESS_JUDGE_VERSION if use_llm_judge else None,
    )

    return EvalRun(
        meta=meta,
        results=tuple(results),
        aggregates=aggregates,
        artifacts=None,
    )


def aggregate_results(results: Iterable[EvalResult]) -> EvalAggregates:
    results = list(results)

    retrieval_results = [r.retrieval_result for r in results]
    overall = summarize(retrieval_results, ks=(1, 3, 5, 10))

    by_type: dict[str, list[EvalResult]] = defaultdict(list)
    by_difficulty: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        if r.query_type:
            by_type[r.query_type.value].append(r)
        if r.difficulty:
            by_difficulty[r.difficulty.value].append(r)

    type_summaries = {
        key: summarize([x.retrieval_result for x in group], ks=(1, 3, 5, 10))
        for key, group in by_type.items()
        if group
    }
    difficulty_summaries = {
        key: summarize([x.retrieval_result for x in group], ks=(1, 3, 5, 10))
        for key, group in by_difficulty.items()
        if group
    }

    aq: dict[str, float] | None = None
    judged = [
        r for r in results
        if r.answer is not None and r.answer_metrics is not None
    ]
    if judged:
        def vals(attr: str) -> list[float]:
            out: list[float] = []
            for r in judged:
                v = getattr(r.answer_metrics, attr, None)
                if v is not None:
                    out.append(float(v))
            return out

        quality = vals("quality_score")
        coverage = vals("citation_coverage")
        correctness = vals("correctness")
        halluc = vals("hallucination_severity")

        # Groundedness rates based on answerable_from_context / evidence_bounded fields
        total_judged_groundedness = 0
        evidence_bounded_count = 0
        not_evidence_bounded_when_unanswerable = 0
        unanswerable_count = 0
        for r in judged:
            m = r.answer_metrics
            if m and m.answerable_from_context is not None:
                total_judged_groundedness += 1
                if m.evidence_bounded is True:
                    evidence_bounded_count += 1
                if m.answerable_from_context is False:
                    unanswerable_count += 1
                    if m.evidence_bounded is False:
                        not_evidence_bounded_when_unanswerable += 1

        aq = {
            "avg_quality_score": float(np.mean(quality)) if quality else 0.0,
            "median_quality_score": float(np.median(quality)) if quality else 0.0,
            "avg_citation_coverage": float(np.mean(coverage)) if coverage else 0.0,
            "avg_correctness_0_5": float(np.mean(correctness)) if correctness else 0.0,
            "avg_hallucination_severity_0_5": float(np.mean(halluc)) if halluc else 0.0,
            "evidence_bounded_rate": (evidence_bounded_count / total_judged_groundedness) if total_judged_groundedness else 0.0,
            "hallucinated_on_unanswerable_rate": (not_evidence_bounded_when_unanswerable / unanswerable_count) if unanswerable_count else 0.0,
        }

    lat: dict[str, float] | None = None
    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    if latencies:
        lat = {
            "avg": float(np.mean(latencies)),
            "p50": float(np.percentile(latencies, 50)),
            "p95": float(np.percentile(latencies, 95)),
        }

    return EvalAggregates(
        overall=overall,
        by_type=type_summaries,
        by_difficulty=difficulty_summaries,
        answer_quality=aq,
        latency_ms=lat,
    )


def save_run(run: EvalRun, output_dir: Path) -> EvalRun:
    output_dir.mkdir(parents=True, exist_ok=True)

    results_file = output_dir / "results.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for r in run.results:
            row: dict[str, Any] = {
                "qid": r.qid,
                "query": r.query,
                "query_type": r.query_type.value if r.query_type else None,
                "difficulty": r.difficulty.value if r.difficulty else None,
                "is_unanswerable": r.is_unanswerable,
                "latency_ms": r.latency_ms,
                "trace_id": r.trace_id,
                "retrieval": {
                    "retrieved_chunk_ids": r.retrieval_result.retrieved_chunk_ids,
                    "relevant_chunk_ids": sorted(r.retrieval_result.relevant_chunk_ids),
                },
            }
            if r.answer is not None:
                row["answer"] = {
                    "text": r.answer.text,
                    "citations": [
                        asdict(c) if hasattr(c, "__dataclass_fields__") else c
                        for c in r.answer.citations
                    ],
                }
            if r.answer_metrics is not None:
                row["answer_metrics"] = asdict(r.answer_metrics)
            f.write(json.dumps(row) + "\n")

    metrics_file = output_dir / "metrics.json"
    meta_dict = asdict(run.meta)
    if meta_dict.get("started_at") and hasattr(meta_dict["started_at"], "isoformat"):
        meta_dict["started_at"] = meta_dict["started_at"].isoformat()

    metrics_payload = {
        "meta": meta_dict,
        "overall": run.aggregates.overall.to_flat_dict(),
        "by_type": {k: v.to_flat_dict() for k, v in run.aggregates.by_type.items()},
        "by_difficulty": {k: v.to_flat_dict() for k, v in run.aggregates.by_difficulty.items()},
        "answer_quality": run.aggregates.answer_quality,
        "latency_ms": run.aggregates.latency_ms,
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    artifacts = {
        "results_jsonl": str(results_file),
        "metrics_json": str(metrics_file),
    }
    return replace(run, artifacts=artifacts)
