from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime

from rag.domain.models import Answer, QueryTrace
from rag.ports import ContextBuilder, Generator, QueryLogger, Retriever
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
    filters: Mapping[str, object] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Answer:
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
    candidates = retriever.retrieve(query, top_k=top_k, filters=filters, metadata=metadata)
    t_retrieval_ms = int((time.perf_counter() - t0) * 1000)

    # Rerank
    t0 = time.perf_counter()
    reranked_candidates = reranker.rerank(query, candidates)
    t_rerank_ms = int((time.perf_counter() - t0) * 1000)
    if keep_k is not None:
        reranked_candidates = reranked_candidates[:keep_k]
    candidates = reranked_candidates

    # Context build
    t1 = time.perf_counter()
    context = context_builder.build(query, candidates, token_budget=token_budget, metadata=metadata)
    t_context_ms = int((time.perf_counter() - t1) * 1000)

    # Generation
    t2 = time.perf_counter()
    answer = generator.generate(query, context, metadata=metadata)
    t_gen_ms = int((time.perf_counter() - t2) * 1000)

    total_ms = int((time.perf_counter() - started) * 1000)

    # Fill trace (immutably)
    trace = replace(
        trace,
        retrieved=tuple(candidates),
        packed_chunk_ids=tuple(
            getattr(c, "chunk_id", None) or c.chunk.chunk_id
            for c in getattr(context, "chunks", [])
        ),
        model=getattr(generator, "model_name", None),
        latency_ms=total_ms,
        reranked=reranked_candidates,
        keep_k=keep_k,
        reranker=getattr(reranker, "name", None),
        metadata={
            **trace.metadata,
            "timing_ms": {
                "retrieval": t_retrieval_ms,
                "rerank": t_rerank_ms,
                "context": t_context_ms,
                "generation": t_gen_ms,
                "total": total_ms,
            },
        },
        answer=answer,
    )

    logger.log(trace)
    return answer
