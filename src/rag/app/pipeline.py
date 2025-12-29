from __future__ import annotations

from typing import Mapping, Optional

from rag.domain.models import Candidate, Chunk, Document, Answer
from rag.ports import Chunker, Embedder, Retriever, VectorStore, Generator, ContextBuilder


def index_document(
    doc: Document,
    *,
    chunker: Chunker,
    embedder: Embedder,
    store: VectorStore,
    metadata: Optional[Mapping[str, object]] = None,
) -> int:
    chunks: list[Chunk] = chunker.chunk(doc, metadata=metadata)
    if not chunks:
        return 0
    vectors = embedder.embed_texts([c.text for c in chunks], metadata=metadata)
    store.upsert(chunks=chunks, vectors=vectors, metadata=metadata)
    return len(chunks)

def rag_answer(
    query: str,
    *,
    retriever: Retriever,
    context_builder: ContextBuilder,
    generator: Generator,
    top_k: int = 10,
    token_budget: int = 1800,
    filters: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> Answer:
    candidates: list[Candidate] = retriever.retrieve(query, top_k=top_k, filters=filters, metadata=metadata)
    context = context_builder.build(query, candidates, token_budget=token_budget, metadata=metadata)
    return generator.generate(query, context, metadata=metadata)

def retrieve_candidates(
    q: str,
    *,
    retriever: Retriever,
    top_k: int = 10,
    filters: Optional[Mapping[str, object]] = None,
    metadata: Optional[Mapping[str, object]] = None,
) -> list[Candidate]:
    return retriever.retrieve(q, top_k=top_k, filters=filters, metadata=metadata)