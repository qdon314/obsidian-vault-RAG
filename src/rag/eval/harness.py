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
import shutil
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

from rag.app.container import Container
from rag.app.manifest_validation import validate_index
from rag.app.query_runner import run_query
from rag.domain.citations import normalize_citation_key
from rag.domain.index_manifest import IndexManifest
from rag.domain.models import Answer, Chunk
from rag.eval import reducers
from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.judges import (
    GOLD_JUDGE_VERSION,
    GROUNDEDNESS_JUDGE_VERSION,
    GoldJudgeResult,
    GroundednessJudgeResult,
    evaluate_groundedness,
    evaluate_vs_expected_answer,
    make_gold_prompt,
    make_groundedness_prompt,
)
from rag.eval.metrics import semantic_similarity, summarize
from rag.eval.models import EvalAggregates, EvalResult, EvalRun, EvalRunMeta, RetrievalResult
from rag.eval.schema import EvalQuery
from rag.ports import Retriever
from rag.ports.chunk_store import ChunkStore

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


def _store_chunks_for_eval(
    container: Container,
    *,
    corpus_id: str | None = None,
) -> list[Chunk]:
    try:
        return container.store.all_chunks()
    except NotImplementedError:
        logger.info("Store does not support all_chunks(); trying chunk_store fallback")

    # Fallback: hydrate all chunks from the ChunkStore (distributed mode)
    chunk_store: ChunkStore | None = container.chunk_store
    if chunk_store is not None:
        meta = {"corpus_id": corpus_id} if corpus_id else None
        all_ids = chunk_store.list_all_chunk_ids(metadata=meta)
        logger.info("ChunkStore reports %d chunk IDs (corpus_id=%s)", len(all_ids), corpus_id)
        if all_ids:
            chunks = list(chunk_store.get_chunks(all_ids).values())
            logger.info("Hydrated %d chunks from chunk store", len(chunks))
            return chunks
    else:
        logger.warning("No chunk_store configured on container")

    logger.warning("No chunks available for citation index — retrieval metrics will be zero")
    return []


def _build_citation_chunk_indexes(
    chunks: Sequence[Chunk],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    citation_to_ids: dict[str, set[str]] = defaultdict(set)
    citation_key_to_ids: dict[str, set[str]] = defaultdict(set)

    # Diagnostic counters
    has_citation = 0
    has_citation_key = 0
    corpus_counts: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        metadata = chunk.metadata
        corpus_counts[str(metadata.get("corpus", "unknown"))] += 1

        citation = metadata.get("citation")
        if isinstance(citation, str) and citation.strip():
            has_citation += 1
            citation_to_ids[normalize_citation_key(citation)].add(chunk.chunk_id)

        citation_key = metadata.get("citation_key")
        if isinstance(citation_key, str) and citation_key.strip():
            has_citation_key += 1
            citation_key_to_ids[normalize_citation_key(citation_key)].add(chunk.chunk_id)

    logger.info(
        "Citation index built: %d chunks total, %d with citation, %d with citation_key, "
        "%d unique citations, %d unique citation_keys. Corpus breakdown: %s",
        len(chunks),
        has_citation,
        has_citation_key,
        len(citation_to_ids),
        len(citation_key_to_ids),
        dict(corpus_counts),
    )
    if citation_key_to_ids:
        sample = sorted(citation_key_to_ids)[:10]
        logger.info("Sample citation_key index entries: %s", sample)

    return dict(citation_to_ids), dict(citation_key_to_ids)


@dataclass(frozen=True, slots=True)
class ResolvedRelevance:
    critical_chunk_ids: set[str]
    supporting_chunk_ids: set[str]
    context_chunk_ids: set[str]

    @property
    def all_chunk_ids(self) -> set[str]:
        return self.critical_chunk_ids | self.supporting_chunk_ids | self.context_chunk_ids


def _resolve_relevance_tiers(
    query: EvalQuery,
    *,
    citation_to_ids: dict[str, set[str]],
    citation_key_to_ids: dict[str, set[str]],
) -> ResolvedRelevance:
    # Legacy relevant_chunk_ids / relevant_citations map to critical relevance.
    critical_ids: set[str] = set(query.relevant_chunk_ids) | set(query.critical_chunk_ids)
    supporting_ids: set[str] = set(query.supporting_chunk_ids)
    context_ids: set[str] = set(query.context_chunk_ids)

    unresolved_critical_citations: list[str] = []
    for citation in query.relevant_citations | query.critical_citations:
        normalized = normalize_citation_key(citation)
        chunk_ids = citation_to_ids.get(normalized) or citation_key_to_ids.get(normalized)
        if chunk_ids:
            critical_ids.update(chunk_ids)
        else:
            unresolved_critical_citations.append(citation)

    unresolved_supporting_citations: list[str] = []
    for citation in query.supporting_citations:
        normalized = normalize_citation_key(citation)
        chunk_ids = citation_to_ids.get(normalized) or citation_key_to_ids.get(normalized)
        if chunk_ids:
            supporting_ids.update(chunk_ids)
        else:
            unresolved_supporting_citations.append(citation)

    unresolved_context_doc_citations: list[str] = []
    for citation_key in query.relevant_doc_citations | query.context_doc_citations:
        normalized = normalize_citation_key(citation_key)
        chunk_ids = citation_key_to_ids.get(normalized)
        if chunk_ids:
            context_ids.update(chunk_ids)
        else:
            unresolved_context_doc_citations.append(citation_key)

    if unresolved_critical_citations:
        logger.warning(
            "Unresolved critical citations for %s: %s",
            query.qid,
            ", ".join(sorted(unresolved_critical_citations)),
        )
    if unresolved_supporting_citations:
        logger.warning(
            "Unresolved supporting citations for %s: %s",
            query.qid,
            ", ".join(sorted(unresolved_supporting_citations)),
        )
    if unresolved_context_doc_citations:
        logger.warning(
            "Unresolved context doc citations for %s: %s",
            query.qid,
            ", ".join(sorted(unresolved_context_doc_citations)),
        )

    # Enforce strict tier precedence: critical > supporting > context.
    supporting_ids -= critical_ids
    context_ids -= critical_ids | supporting_ids

    return ResolvedRelevance(
        critical_chunk_ids=critical_ids,
        supporting_chunk_ids=supporting_ids,
        context_chunk_ids=context_ids,
    )


def _copy_attrs_if_present(
    dest: dict[str, Any],
    source: Any,
    *,
    attr_to_key: dict[str, str],
) -> None:
    for attr, key in attr_to_key.items():
        if hasattr(source, attr):
            dest[key] = getattr(source, attr)


def _build_retrieval_meta_extra(retriever: Retriever, *, score_ids: str) -> dict[str, Any]:
    extra: dict[str, Any] = {
        "score_ids": score_ids,
        "retriever_class": type(retriever).__name__,
        "retrieval_backend": "vector",
    }

    if not (hasattr(retriever, "primary") and hasattr(retriever, "secondary")):
        return extra

    extra["retrieval_backend"] = "hybrid"
    _copy_attrs_if_present(
        extra,
        retriever,
        attr_to_key={
            "primary_weight": "hybrid_primary_weight",
            "secondary_weight": "hybrid_secondary_weight",
            "rrf_k": "hybrid_rrf_k",
        },
    )

    secondary = getattr(retriever, "secondary")  # noqa: B009
    extra["hybrid_secondary_class"] = type(secondary).__name__
    _copy_attrs_if_present(
        extra,
        secondary,
        attr_to_key={
            "k1": "hybrid_bm25_k1",
            "b": "hybrid_bm25_b",
        },
    )
    return extra


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
        critical_chunk_ids=query.relevant_chunk_ids | query.critical_chunk_ids,
        supporting_chunk_ids=query.supporting_chunk_ids,
        context_chunk_ids=query.context_chunk_ids,
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
) -> tuple[AnswerQualityMetrics, GroundednessJudgeResult | None, GoldJudgeResult | None]:
    sem_sim: float | None = None
    if embedder and query.expected_answer:
        sem_sim = semantic_similarity(query.expected_answer, answer.text, embedder)

    # Judge outputs (optional)
    correctness = completeness = relevance = hallucination_severity = None
    answerable_from_context = evidence_bounded = supported_claims = unsupported_claims = None
    gr: GroundednessJudgeResult | None = None
    gold: GoldJudgeResult | None = None

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
                query_type=query.query_type.value if query.query_type else "unknown",
                difficulty=query.difficulty.value if query.difficulty else "unknown",
                requires_synthesis=query.requires_synthesis,
            )
            gold = evaluate_vs_expected_answer(
                client=client,
                model=judge_model,
                prompt=gold_prompt,
            )
            hallucination_severity = (
                reducers.hallucination_severity(
                    groundedness=gr,
                )
                if gr
                else None
            )
            correctness = gold.correctness
            completeness = gold.completeness
            relevance = gold.relevance

    metrics = AnswerQualityMetrics.compute(
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
    return metrics, gr, gold


# ----------------------------
# Main eval
# ----------------------------


def _evaluate_single_query(
    *,
    q: EvalQuery,
    container: Container,
    citation_to_ids: dict[str, set[str]],
    citation_key_to_ids: dict[str, set[str]],
    top_k: int,
    keep_k: int | None,
    token_budget: int,
    run_generation: bool,
    use_llm_judge: bool,
    judge_client: OpenAI | None,
    judge_model: str,
    score_ids: str,
) -> EvalResult:
    """Evaluate a single query. Thread-safe — no shared mutable state."""
    relevance = _resolve_relevance_tiers(
        q,
        citation_to_ids=citation_to_ids,
        citation_key_to_ids=citation_key_to_ids,
    )
    query_filter = q.get_filter()

    # --- Retrieval only ---
    if not run_generation:
        cands = container.retriever.retrieve(q.query, top_k=top_k, where=query_filter)
        retrieved_ids = [c.chunk.chunk_id for c in cands]
        retrieval_result = RetrievalResult(
            qid=q.qid,
            retrieved_chunk_ids=tuple(retrieved_ids),
            relevant_chunk_ids=relevance.all_chunk_ids,
            critical_chunk_ids=relevance.critical_chunk_ids,
            supporting_chunk_ids=relevance.supporting_chunk_ids,
            context_chunk_ids=relevance.context_chunk_ids,
        )
        return EvalResult(
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
        relevant_chunk_ids=relevance.all_chunk_ids,
        critical_chunk_ids=relevance.critical_chunk_ids,
        supporting_chunk_ids=relevance.supporting_chunk_ids,
        context_chunk_ids=relevance.context_chunk_ids,
    )

    answer_metrics, groundedness_result, gold_result = evaluate_answer_quality(
        query=q,
        answer=answer,
        retrieved_chunks=run.context_pack.chunks,
        client=judge_client if use_llm_judge else None,
        judge_model=judge_model or "",
        embedder=getattr(container, "embedder", None),
        use_llm_judge=use_llm_judge,
    )

    outcome_label = reducers.outcome_label(
        gold=gold_result,
        groundedness=groundedness_result,
    )

    return EvalResult(
        qid=q.qid,
        query=q.query,
        retrieval_result=retrieval_result,
        answer=answer,
        answer_metrics=answer_metrics,
        groundedness_result=groundedness_result,
        outcome_label=outcome_label,
        query_type=q.query_type,
        difficulty=q.difficulty,
        is_unanswerable=q.is_unanswerable,
        latency_ms=getattr(run, "latency_ms", None),
        trace_id=getattr(run, "trace_id", None),
    )


def run_full_eval(
    *,
    eval_queries: list[EvalQuery],
    container: Container,
    queries_path: str | None,
    index_dir: Path | None = None,
    manifest: IndexManifest | None = None,
    top_k: int = 10,
    keep_k: int | None = None,
    token_budget: int = 1500,
    run_generation: bool = False,
    use_llm_judge: bool = False,
    judge_client: OpenAI | None = None,
    judge_model: str | None = None,
    score_ids: str = "reranked",  # "retrieved" or "reranked"
    run_name: str | None = None,
    max_workers: int = 1,
    corpus_id: str | None = None,
) -> EvalRun:
    container.store.load()

    # Diagnostic: log store point count to surface empty-collection issues early
    store_count = container.store.count()
    logger.info(
        "Vector store loaded: %s, %d points",
        type(container.store).__name__,
        store_count,
    )
    if store_count == 0:
        logger.warning(
            "Vector store has 0 points — all retrieval results will be empty. "
            "Verify that ingestion completed and the store is accessible."
        )

    store_chunks = _store_chunks_for_eval(container, corpus_id=corpus_id)
    citation_to_ids, citation_key_to_ids = _build_citation_chunk_indexes(store_chunks)

    # Validate index manifest (explicit manifest takes precedence over index_dir)
    if manifest is None and index_dir is not None:
        manifest_path = index_dir / "manifest.json"
        if manifest_path.exists():
            manifest = IndexManifest.load(index_dir)
        else:
            logger.warning("No manifest.json in %s — skipping validation", index_dir)
    if manifest is not None:
        validate_index(manifest, container.embedder)
    run_id = uuid.uuid4().hex
    started_at = datetime.now(UTC)

    if use_llm_judge and (judge_client is None or not judge_model):
        raise ValueError("use_llm_judge=True requires judge_client and judge_model")

    results: list[EvalResult] = []
    empty_retrieval_count = 0

    if max_workers <= 1:
        # Sequential path — preserves exact legacy behavior and stack traces.
        for q in eval_queries:
            result = _evaluate_single_query(
                q=q,
                container=container,
                citation_to_ids=citation_to_ids,
                citation_key_to_ids=citation_key_to_ids,
                top_k=top_k,
                keep_k=keep_k,
                token_budget=token_budget,
                run_generation=run_generation,
                use_llm_judge=use_llm_judge,
                judge_client=judge_client,
                judge_model=judge_model or "",
                score_ids=score_ids,
            )
            results.append(result)
            if not result.retrieval_result.retrieved_chunk_ids:
                empty_retrieval_count += 1
                if empty_retrieval_count <= 3:
                    logger.warning(
                        "Query %s returned 0 candidates", result.qid,
                    )
    else:
        # Concurrent path — fan out queries across threads.
        logger.info("Running eval with %d workers across %d queries", max_workers, len(eval_queries))
        future_to_idx: dict[Any, int] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for idx, q in enumerate(eval_queries):
                future = executor.submit(
                    _evaluate_single_query,
                    q=q,
                    container=container,
                    citation_to_ids=citation_to_ids,
                    citation_key_to_ids=citation_key_to_ids,
                    top_k=top_k,
                    keep_k=keep_k,
                    token_budget=token_budget,
                    run_generation=run_generation,
                    use_llm_judge=use_llm_judge,
                    judge_client=judge_client,
                    judge_model=judge_model or "",
                    score_ids=score_ids,
                )
                future_to_idx[future] = idx

            # Collect results, logging progress as futures complete.
            indexed_results: list[tuple[int, EvalResult]] = []
            for completed, future in enumerate(as_completed(future_to_idx), 1):
                idx = future_to_idx[future]
                result = future.result()  # propagates exceptions
                indexed_results.append((idx, result))
                if completed % 10 == 0 or completed == len(eval_queries):
                    logger.info("Eval progress: %d/%d queries", completed, len(eval_queries))

        # Sort by original index to preserve deterministic output order.
        indexed_results.sort(key=lambda x: x[0])
        for _, result in indexed_results:
            results.append(result)
            if not result.retrieval_result.retrieved_chunk_ids:
                empty_retrieval_count += 1
                if empty_retrieval_count <= 3:
                    logger.warning(
                        "Query %s returned 0 candidates", result.qid,
                    )

    if empty_retrieval_count > 0:
        logger.warning(
            "Retrieval returned 0 candidates for %d/%d queries",
            empty_retrieval_count,
            len(eval_queries),
        )

    aggregates = aggregate_results(results)

    # Capture retrieval configuration so run artifacts explain scoring context.
    extra = _build_retrieval_meta_extra(container.retriever, score_ids=score_ids)

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
        run_name=run_name,
        gold_judge_version=GOLD_JUDGE_VERSION if use_llm_judge else None,
        groundedness_judge_version=GROUNDEDNESS_JUDGE_VERSION if use_llm_judge else None,
        extra=extra,
        index_name=manifest.index_name if manifest else None,
        index_id=manifest.index_id if manifest else None,
        index_build_id=manifest.build_id if manifest else None,
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
    judged = [r for r in results if r.answer is not None and r.answer_metrics is not None]
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
            "evidence_bounded_rate": (evidence_bounded_count / total_judged_groundedness)
            if total_judged_groundedness
            else 0.0,
            "hallucinated_on_unanswerable_rate": (
                not_evidence_bounded_when_unanswerable / unanswerable_count
            )
            if unanswerable_count
            else 0.0,
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
                    "critical_chunk_ids": sorted(r.retrieval_result.critical_chunk_ids),
                    "supporting_chunk_ids": sorted(r.retrieval_result.supporting_chunk_ids),
                    "context_chunk_ids": sorted(r.retrieval_result.context_chunk_ids),
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
                row["answer_metrics"]["reducer_version"] = reducers.REDUCER_VERSION

            if r.groundedness_result is not None:
                row["groundedness_result"] = asdict(r.groundedness_result)

            if r.outcome_label is not None:
                row["outcome_label"] = r.outcome_label.value

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

    # Maintain a stable "latest" symlink so verdict scripts can use
    # --current eval/runs/latest/ without knowing the timestamp.
    latest = output_dir.parent / "latest"
    if latest.is_symlink():
        latest.unlink()
    elif latest.is_dir():
        shutil.rmtree(latest)
    elif latest.exists():
        latest.unlink()
    latest.symlink_to(output_dir.name)

    artifacts = {
        "results_jsonl": str(results_file),
        "metrics_json": str(metrics_file),
    }
    return replace(run, artifacts=artifacts)
