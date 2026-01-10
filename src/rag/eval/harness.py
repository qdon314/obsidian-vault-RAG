#!/usr/bin/env python3
"""
Evaluation harness for RAG system.

This script:
1. Loads evaluation queries from JSONL
2. Runs them through the RAG pipeline
3. Computes retrieval and answer quality metrics
4. Generates detailed reports

Usage:
    python -m experiments.eval_harness --queries experiments/eval_queries.jsonl --output experiments/results/
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from rag.app.container import Container
from rag.app.query_runner import run_query
from rag.domain.models import Answer
from rag.eval.metrics import RetrievalResult, summarize
from rag.eval.models import AnswerQualityMetrics, EvalAggregates, EvalResult, EvalRun, EvalRunMeta
from rag.eval.schema import EvalQuery
from rag.ports.retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

LLM_JUDGE_PROMPT = """You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.

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
  "hallucination": <0-5>,
  "reasoning": "<brief explanation>"
}}"""


def load_eval_queries(path: Path) -> list[EvalQuery]:
    """Load evaluation queries from JSONL file."""
    queries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            queries.append(EvalQuery.from_dict(data))
    return queries


def run_retrieval_eval(
    query: EvalQuery,
    retriever: Retriever,
    top_k: int = 10,
) -> RetrievalResult:
    """
    Run retrieval for a single query and compute metrics.

    Args:
        query: Evaluation query with ground truth
        retriever: Retriever instance
        top_k: Number of results to retrieve

    Returns:
        RetrievalResult with metrics
    """
    candidates = retriever.retrieve(query.query, top_k=top_k)
    retrieved_ids = [cand.chunk.chunk_id for cand in candidates]

    return RetrievalResult(
        qid=query.qid,
        retrieved_chunk_ids=tuple(retrieved_ids),
        relevant_chunk_ids=query.relevant_chunk_ids,
    )


def evaluate_answer_with_llm(
    query: str,
    expected_answer: str,
    generated_answer: str,
    client: OpenAI,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Use LLM to evaluate answer quality.

    Args:
        query: The query
        expected_answer: Ground truth answer
        generated_answer: System-generated answer
        client: OpenAI client
        model: Model to use for evaluation

    Returns:
        Dictionary with scores and reasoning
    """
    prompt = LLM_JUDGE_PROMPT.format(
        query=query,
        expected_answer=expected_answer,
        generated_answer=generated_answer,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert evaluator."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )

        content = response.choices[0].message.content
        if not content:
            return {}

        result = json.loads(content)
        return result

    except Exception as e:
        logger.error(f"Error in LLM evaluation: {e}")
        return {}


def compute_semantic_similarity(text1: str, text2: str, embedder: Any) -> float:
    """
    Compute cosine similarity between two texts using embeddings.

    Args:
        text1: First text
        text2: Second text
        embedder: Embedder instance

    Returns:
        Cosine similarity (-1 to 1)
    """
    try:
        emb1 = embedder.embed(text1)
        emb2 = embedder.embed(text2)

        # Cosine similarity
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    except Exception as e:
        logger.error(f"Error computing similarity: {e}")
        return 0.0


def evaluate_answer_quality(
    query: EvalQuery,
    answer: Answer,
    client: OpenAI | None = None,
    embedder: Any | None = None,
    use_llm_judge: bool = True,
) -> AnswerQualityMetrics:
    """
    Comprehensive answer quality evaluation.

    Args:
        query: Evaluation query with ground truth
        answer: Generated answer
        client: OpenAI client for LLM-as-judge
        embedder: Embedder for semantic similarity
        use_llm_judge: Whether to use LLM for evaluation

    Returns:
        AnswerQualityMetrics
    """
    is_abstained=answer.abstained
    answer_length=len(answer.text)
    num_citations=len(answer.citations)

    # If no expected answer, we can't evaluate quality
    if not query.expected_answer:
        return AnswerQualityMetrics(
            is_abstained=is_abstained,
            answer_length=answer_length,
            num_citations=num_citations,
        )

    # Semantic similarity
    if embedder:
        semantic_similarity = compute_semantic_similarity(
            query.expected_answer,
            answer.text,
            embedder,
        )

    # LLM-as-judge evaluation
    if use_llm_judge and client and not answer.abstained:
        llm_scores = evaluate_answer_with_llm(
            query=query.query,
            expected_answer=query.expected_answer,
            generated_answer=answer.text,
            client=client,
        )

        correctness = llm_scores.get("correctness")
        completeness = llm_scores.get("completeness")
        relevance = llm_scores.get("relevance")
        hallucination_score = llm_scores.get("hallucination")

        # Derive binary metrics
        if correctness is not None:
            is_correct = correctness >= 4.0
        if hallucination_score is not None:
            has_hallucination = hallucination_score >= 3.0

    return AnswerQualityMetrics(
        semantic_similarity=semantic_similarity if embedder else None,
        correctness=correctness if use_llm_judge else None,
        completeness=completeness if use_llm_judge else None,
        relevance=relevance if use_llm_judge else None,
        hallucination_score=hallucination_score if use_llm_judge else None,
        is_correct=is_correct if use_llm_judge else None,
        has_hallucination=has_hallucination if use_llm_judge else None,
        is_abstained=is_abstained,
        answer_length=answer_length,
        num_citations=num_citations,
    )

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
    score_ids: str = "reranked", # "retrieved" or "reranked"
) -> EvalRun:
    """
    Run a complete evaluation over a set of queries.

    Supports two modes:
        - Retrieval-only (run_generation=False): Only retrieves candidates and scores retrieval metrics.
        - Full pipeline (run_generation=True): Runs retrieve → rerank → context → generate, then
          optionally evaluates answer quality with an LLM judge.

    Args:
        eval_queries: List of EvalQuery objects with ground truth annotations.
        container: Dependency injection container with retriever, reranker, generator, etc.
        queries_path: Path to the source queries file (recorded in run metadata).
        top_k: Number of candidates to retrieve from the vector store.
        keep_k: If set, truncate reranked candidates to this count before context building.
        token_budget: Maximum tokens for context window during generation.
        run_generation: If True, run the full RAG pipeline; otherwise retrieval-only.
        use_llm_judge: If True, use an LLM to score answer quality against expected answers.
        judge_client: OpenAI client instance for LLM-as-judge evaluation.
        judge_model: Model name for LLM judge (e.g., "gpt-4o-mini").
        score_ids: Which chunk IDs to use for retrieval metrics: "retrieved" or "reranked".

    Returns:
        EvalRun containing per-query results, aggregate metrics, and run metadata.
    """
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC)

    results: list[EvalResult] = []

    for _, q in enumerate(eval_queries, 1):
        # --- Retrieval only ---
        if not run_generation:
            cands = container.retriever.retrieve(q.query, top_k=top_k)
            retrieved_ids = [c.chunk.chunk_id for c in cands]  # <- make Candidate expose chunk_id
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

        # --- Full pipeline via query runner ---
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
        )
        
        if score_ids == "retrieved":
            chosen_ids = tuple(run.retrieved_chunk_ids)
        elif score_ids == "reranked":
            chosen_ids = tuple(run.reranked_chunk_ids)
        else:
            raise ValueError("score_ids must be 'retrieved' or 'reranked'")
        
        answer: Answer = run.answer
        retrieved_ids = list(run.reranked_chunk_ids)  # or run.retrieved_chunk_ids
        retrieval_result = RetrievalResult(
            qid=q.qid,
            retrieved_chunk_ids=chosen_ids,
            relevant_chunk_ids=q.relevant_chunk_ids,
        )

        answer_metrics = None
        if q.expected_answer and not answer.abstained:
            answer_metrics = evaluate_answer_quality(
                query=q,
                answer=answer,
                client=judge_client if use_llm_judge else None,
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
    )

    return EvalRun(
        meta=meta,
        results=tuple(results),
        aggregates=aggregates,
        artifacts=None,
    )

def aggregate_results(results: Iterable[EvalResult]) -> EvalAggregates:
    """
    Aggregate per-query evaluation results into summary statistics.

    Computes:
        - Overall retrieval metrics (recall, precision, MRR, NDCG, etc.)
        - Breakdowns by query type and difficulty
        - Answer quality rollups (avg/median correctness) if LLM judge was used
        - Latency statistics (avg, p50, p95) if generation was run

    Args:
        results: Iterable of EvalResult objects from run_full_eval.

    Returns:
        EvalAggregates containing overall, by_type, by_difficulty summaries,
        plus optional answer_quality and latency_ms rollups.
    """
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

    # Optional answer-quality rollups (keep simple for now)
    aq: dict[str, float] | None = None
    with_aq = [r for r in results if r.answer_metrics and r.answer_metrics.correctness is not None]
    if with_aq:
        correctness = [r.answer_metrics.correctness for r in with_aq if r.answer_metrics and r.answer_metrics.correctness is not None]
        aq = {
            "avg_correctness": float(np.mean(correctness)) if correctness else 0.0,
            "median_correctness": float(np.median(correctness)) if correctness else 0.0,
        }

    # Optional latency rollups
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
    
def save_run(run: EvalRun, output_dir: Path, run_name: str | None = None) -> EvalRun:
    """
    Persist an evaluation run to disk as JSONL results and JSON metrics.

    Creates two files in output_dir:
        - results_{name}.jsonl: Per-query results with retrieval, answer, and metrics
        - metrics_{name}.json: Aggregated metrics and run metadata

    Args:
        run: The completed EvalRun to save.
        output_dir: Directory to write artifacts to (created if it doesn't exist).
        run_name: Optional name for this run; defaults to run_id or timestamp.

    Returns:
        Updated EvalRun with artifacts dict containing file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    name = run_name or run.meta.run_id or timestamp

    results_file = output_dir / f"results_{name}.jsonl"
    with results_file.open("w", encoding="utf-8") as f:
        for r in run.results:
            row = {
                "qid": r.qid,
                "query": r.query,
                "query_type": r.query_type.value if r.query_type else None,
                "difficulty": r.difficulty.value if r.difficulty else None,
                "is_unanswerable": r.is_unanswerable,
                "latency_ms": r.latency_ms,
                "trace_id": getattr(r, "trace_id", None),
                "retrieval": {
                    "retrieved_chunk_ids": r.retrieval_result.retrieved_chunk_ids,
                    "relevant_chunk_ids": sorted(r.retrieval_result.relevant_chunk_ids),
                },
            }
            if r.answer is not None:
                row["answer"] = {
                    "text": r.answer.text,
                    "abstained": r.answer.abstained,
                    "citations": [asdict(c) if hasattr(c, "__dataclass_fields__") else c for c in r.answer.citations],
                }
            if r.answer_metrics is not None:
                row["answer_metrics"] = asdict(r.answer_metrics)

            f.write(json.dumps(row) + "\n")

    metrics_file = output_dir / f"metrics_{name}.json"
    # summarize() now returns RetrievalSummary dataclass; use .to_dict() for flat output if you want
    metrics_payload = {
        "meta": asdict(run.meta),
        "overall": run.aggregates.overall.to_dict(),
        "by_type": {k: v.to_dict() for k, v in run.aggregates.by_type.items()},
        "by_difficulty": {k: v.to_dict() for k, v in run.aggregates.by_difficulty.items()},
        "answer_quality": run.aggregates.answer_quality,
        "latency_ms": run.aggregates.latency_ms,
    }
    metrics_file.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    artifacts = {
        "results_jsonl": str(results_file),
        "metrics_json": str(metrics_file),
    }
    return replace(run, artifacts=artifacts)