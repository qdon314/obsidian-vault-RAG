from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from time import perf_counter

from rag.domain.filters import Where
from rag.domain.models import Answer, Candidate, Chunk, Document
from rag.ports import (
    Chunker,
    ChunkStore,
    ContextBuilder,
    Embedder,
    Generator,
    Retriever,
    VectorStore,
)


def index_document(
    doc: Document,
    *,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    chunk_store: ChunkStore | None = None,
    metadata: Mapping[str, object] | None = None,
) -> int:
    chunks: list[Chunk] = chunker.chunk(doc, metadata=metadata)
    if not chunks:
        return 0
    vectors = embedder.embed_texts([c.text for c in chunks], metadata=metadata)
    store.upsert(chunks=chunks, vectors=vectors, metadata=metadata)
    if chunk_store is not None:
        chunk_store.store_chunks(chunks, metadata=metadata)
    return len(chunks)


def index_documents(
    docs: Sequence[Document],
    *,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    chunk_store: ChunkStore | None = None,
    embed_batch_size: int = 512,
    on_doc_chunked: Callable[[Document, int, float], None] | None = None,
    on_batch_embedded: Callable[[int, float], None] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> int:
    """Index multiple documents with cross-document embedding batching.

    Accumulates chunks across documents and embeds them in large batches,
    reducing the number of API round-trips compared to per-document embedding.

    Args:
        docs: Documents to index.
        chunker: Chunker to split documents into chunks.
        embedder: Embedder to convert chunk texts into vectors.
        store: VectorStore to persist chunks and vectors.
        chunk_store: Optional ChunkStore for dual-write (distributed mode).
        embed_batch_size: Max chunks per embedding API call.
        on_doc_chunked: Called after each document is chunked.
            Signature: (doc, n_chunks, elapsed_seconds).
        on_batch_embedded: Called after each embedding batch completes.
            Signature: (batch_size, elapsed_seconds).
        metadata: Optional metadata passed through to adapters.

    Returns:
        Total number of chunks indexed.
    """
    chunk_buffer: list[Chunk] = []
    total_chunks = 0

    def _flush() -> None:
        nonlocal chunk_buffer, total_chunks
        if not chunk_buffer:
            return
        t0 = perf_counter()
        vectors = embedder.embed_texts([c.text for c in chunk_buffer], metadata=metadata)
        store.upsert(chunks=chunk_buffer, vectors=vectors, metadata=metadata)
        if chunk_store is not None:
            chunk_store.store_chunks(chunk_buffer, metadata=metadata)
        elapsed = perf_counter() - t0
        if on_batch_embedded:
            on_batch_embedded(len(chunk_buffer), elapsed)
        total_chunks += len(chunk_buffer)
        chunk_buffer = []

    for doc in docs:
        t0 = perf_counter()
        chunks = chunker.chunk(doc, metadata=metadata)
        chunk_dt = perf_counter() - t0

        if on_doc_chunked:
            on_doc_chunked(doc, len(chunks), chunk_dt)

        chunk_buffer.extend(chunks)

        if len(chunk_buffer) >= embed_batch_size:
            _flush()

    _flush()
    return total_chunks


def rag_answer(
    query: str,
    *,
    retriever: Retriever,
    context_builder: ContextBuilder,
    generator: Generator,
    top_k: int = 10,
    token_budget: int = 1800,
    where: Where = None,
    metadata: Mapping[str, object] | None = None,
) -> Answer:
    candidates: list[Candidate] = retriever.retrieve(
        query, top_k=top_k, where=where, metadata=metadata
    )
    context = context_builder.build(query, candidates, token_budget=token_budget, metadata=metadata)
    return generator.generate(query, context, metadata=metadata)


def retrieve_candidates(
    q: str,
    *,
    retriever: Retriever,
    top_k: int = 10,
    where: Where = None,
    metadata: Mapping[str, object] | None = None,
) -> list[Candidate]:
    return retriever.retrieve(q, top_k=top_k, where=where, metadata=metadata)
