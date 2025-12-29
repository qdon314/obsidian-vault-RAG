from __future__ import annotations

from dataclasses import dataclass
from os import getenv

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
# from rag.adapters.embedding.openai_embedder import OpenAIEmbedder --- IGNORE ---
from rag.adapters.generation.openai_chat import OpenAIChatGenerator
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore

from rag.ports import Chunker, Embedder, Retriever, VectorStore, Generator, ContextBuilder


@dataclass(frozen=True, slots=True)
class Container:
    """
    Lightweight dependency container.
    Holds instances of adapters implementing various ports.
    """
    chunker: Chunker
    embedder: Embedder
    store: VectorStore
    retriever: Retriever
    context_builder: ContextBuilder
    generator: Generator


def build_container() -> Container:
    # Later: read these from config/env
    chunker = FixedChunker(chunk_size=1200, overlap=150)
    embedder = DummyEmbedder(dim=128)
    
    # If you want OpenAI embeddings, swap to:
    # embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
    
    context_builder = SimpleContextBuilder(min_score=None, max_chunks=10, dedupe=True, include_scores=False)
    
    api_key = getenv("OPENAI_API_KEY", "")
    generator = OpenAIChatGenerator(api_key=api_key, model=getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    store = InMemoryVectorStore()
    retriever = VectorRetriever(embedder=embedder, store=store)
    return Container(
        chunker=chunker,
        embedder=embedder,
        store=store,
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
    )