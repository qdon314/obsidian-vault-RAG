from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any

from rag.domain.filters import Where
from rag.domain.models import QueryRunResult, QueryTrace
from rag.ports import ContextBuilder, Generator, PromptCompressor, QueryLogger, Retriever
from rag.ports.reranker import Reranker


def run_query(
    query: str,
    *,
    retriever: Retriever,
    reranker: Reranker,
    context_builder: ContextBuilder,
    generator: Generator,
    logger: QueryLogger,
    top_k: int,
    keep_k: int | None,
    token_budget: int,
    where: Where = None,
    metadata: Mapping[str, object] | None = None,
    compressor: PromptCompressor | None = None,
) -> QueryRunResult:
    """
    Execute a full RAG query pipeline: retrieve, rerank, build context, and generate.

    Pipeline stages:
        1. Retrieval: Fetch top_k candidate chunks from the vector store
        2. Reranking: Re-score candidates and optionally truncate to keep_k
        3. Context building: Pack chunks into a prompt within token_budget
        4. Generation: Generate an answer using the LLM

    Args:
        query: The user's natural language query.
        retriever: Retriever implementation for vector similarity search.
        reranker: Reranker implementation to re-score retrieved candidates.
        context_builder: Builds the final context string from candidates.
        generator: LLM generator to produce the answer.
        logger: QueryLogger to record the trace for observability.
        top_k: Number of candidates to retrieve from the vector store.
        keep_k: If set, truncate reranked candidates to this count before context building.
        token_budget: Maximum tokens allowed for the context window.
        where: Optional filter to constrain retrieval (Filter from domain.filters).
        metadata: Optional metadata passed through the pipeline for logging/customization.

    Returns:
        A QueryRunResult containing the generated response, chunk IDs, and timing.
    """
    trace_id = uuid.uuid4().hex
    started = time.perf_counter()

    trace = QueryTrace(
        trace_id=trace_id,
        query=query,
        created_at=datetime.now(UTC),
        top_k=top_k,
        token_budget=token_budget,
        metadata=dict(metadata or {}),
    )

    # Retrieval
    t0 = time.perf_counter()
    retrieved_candidates = retriever.retrieve(query, top_k=top_k, where=where, metadata=metadata)
    t_retrieval_ms = int((time.perf_counter() - t0) * 1000)

    # Rerank - if reranking is disabled, this is a no-op
    t0 = time.perf_counter()
    reranked_candidates = reranker.rerank(query, retrieved_candidates, metadata=metadata)
    t_rerank_ms = int((time.perf_counter() - t0) * 1000)
    if keep_k is not None:
        reranked_candidates = reranked_candidates[:keep_k]

    # Context build - pass reranked candidates (already ordered by reranker)
    t1 = time.perf_counter()
    context = context_builder.build(
        query, candidates=reranked_candidates, token_budget=token_budget, metadata=metadata
    )
    t_context_ms = int((time.perf_counter() - t1) * 1000)

    # Compression (optional — skipped when no compressor is provided)
    t_compress_ms = 0
    compression_meta: dict[str, Any] | None = None
    if compressor is not None:
        compression_result = compressor.compress(context, query=query, metadata=metadata)
        t_compress_ms = compression_result.latency_ms
        context = compression_result.context_pack
        compression_meta = compression_result.to_metadata_dict()

    # Generation
    t2 = time.perf_counter()
    answer = generator.generate(query, context, metadata=metadata)
    t_gen_ms = int((time.perf_counter() - t2) * 1000)

    total_ms = int((time.perf_counter() - started) * 1000)

    retrieved_ids = tuple(c.chunk.chunk_id for c in retrieved_candidates)
    reranked_ids = tuple(c.chunk.chunk_id for c in reranked_candidates)

    packed_ids = tuple(
        getattr(c, "chunk_id", None) or c.chunk.chunk_id for c in getattr(context, "chunks", [])
    )

    # Fill trace (immutably)
    extra_meta: dict[str, Any] = {}
    if compression_meta is not None:
        extra_meta["compression"] = compression_meta

    trace = replace(
        trace,
        retrieved=tuple(retrieved_candidates),
        reranked=tuple(reranked_candidates),
        packed_chunk_ids=packed_ids,
        model=getattr(generator, "model_name", None),
        latency_ms=total_ms,
        keep_k=keep_k,
        reranker=getattr(reranker, "name", None),
        metadata={
            **trace.metadata,
            **extra_meta,
            "timing_ms": {
                "retrieval": t_retrieval_ms,
                "rerank": t_rerank_ms,
                "context": t_context_ms,
                "compress": t_compress_ms,
                "generation": t_gen_ms,
                "total": total_ms,
            },
        },
        answer=answer,
    )

    logger.log(trace)

    return QueryRunResult(
        trace_id=trace_id,
        answer=answer,
        context_pack=context,
        retrieved_chunk_ids=retrieved_ids,
        reranked_chunk_ids=reranked_ids,
        packed_chunk_ids=packed_ids,
        latency_ms=total_ms,
        compression=compression_meta,
    )
