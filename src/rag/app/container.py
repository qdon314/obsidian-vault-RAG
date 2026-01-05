from __future__ import annotations

from dataclasses import dataclass
from rag import settings
from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.embedding.openai_embedder import OpenAIEmbedder
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
    _settings = settings.load_settings()



    context_builder = SimpleContextBuilder(min_score=None, max_chunks=10, dedupe=True, include_scores=False)
    
    
    api_key = str(_settings.secrets.openai_api_key)
    embedder = OpenAIEmbedder(api_key=api_key, model=str(_settings.embeddings.model))
    generator = OpenAIChatGenerator(api_key=api_key, model=str(_settings.llm.model))
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