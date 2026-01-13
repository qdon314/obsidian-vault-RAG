from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rag import settings
from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.embedding.openai_embedder import OpenAIEmbedder
from rag.adapters.generation.openai_chat import OpenAIChatGenerator
from rag.adapters.ingestion.filesystem import FilesystemIngestor
from rag.adapters.ingestion.loaders.obsidian_markdown_loader import ObsidianMarkdownLoader
from rag.adapters.ingestion.loaders.text_loader import TextLoader
from rag.adapters.logging.jsonl_logger import JsonlQueryLogger
from rag.adapters.reranking.rerank_heuristic import HeuristicReranker
from rag.adapters.reranking.rerank_noop import NoOpReranker
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.adapters.vectorstores.jsonl_store import JsonlVectorStore
from rag.ports import (
    Chunker,
    ContextBuilder,
    Embedder,
    Generator,
    Ingestor,
    QueryLogger,
    Reranker,
    Retriever,
    VectorStore,
)


@dataclass(frozen=True, slots=True)
class Container:
    chunker: Chunker
    context_builder: ContextBuilder
    embedder: Embedder
    generator: Generator
    ingestor: Ingestor
    store: VectorStore = field(repr=False)
    retriever: Retriever = field(repr=False)
    logger: QueryLogger = field(repr=False)
    reranker: Reranker = field(repr=False)


@dataclass(frozen=True, slots=True)
class ContainerOverrides:
    embedder_backend: Literal["openai", "dummy"] | None = None
    dummy_embed_dim: int | None = None

    store_backend: Literal["memory", "jsonl", "qdrant"] | None = None
    jsonl_index_dir: Path | None = None

    # Qdrant overrides
    qdrant_collection: str | None = None
    qdrant_url: str | None = None
    qdrant_path: Path | None = None

    chunk_size: int | None = None
    chunk_overlap: int | None = None

    vault_dir: Path | None = None

    top_k: int | None = None
    rerank_backend: Literal["heuristic", "noop"] | None = None
    rerank_enabled: bool | None = None


def build_container(
    *,
    cfg: settings.Settings | None = None,
    overrides: ContainerOverrides | None = None,
) -> Container:
    cfg = cfg or settings.load_settings()
    ovrds = overrides or ContainerOverrides()

    # ----- chunking (defaults from settings, overridden by CLI)
    chunk_size = ovrds.chunk_size if ovrds.chunk_size is not None else cfg.chunking.chunk_size
    overlap = ovrds.chunk_overlap if ovrds.chunk_overlap is not None else cfg.chunking.overlap
    chunker = FixedChunker(chunk_size=chunk_size, overlap=overlap)

    # ----- context building
    context_builder = SimpleContextBuilder(
        min_score=cfg.context.min_score,
        max_chunks=cfg.context.max_chunks,
        dedupe=cfg.context.dedupe,
        include_scores=cfg.context.include_scores,
    )

    # ----- ingestion + loaders
    vault_dir = ovrds.vault_dir if ovrds.vault_dir is not None else cfg.paths.vault_dir
    md_loader = ObsidianMarkdownLoader(
        vault_dir=vault_dir,
        expand_embeds=cfg.ingestion.expand_embeds,
        max_embed_depth=cfg.ingestion.max_embed_depth,
    )
    ingestor = FilesystemIngestor(
        text_loader=TextLoader(),
        markdown_loader=md_loader,
    )

    # ----- embedder selection
    api_key = str(cfg.secrets.openai_api_key)
    embedder_backend = ovrds.embedder_backend or cfg.embeddings.backend  # e.g. "openai"
    if embedder_backend == "dummy":
        dim = (
            ovrds.dummy_embed_dim if ovrds.dummy_embed_dim is not None else cfg.embeddings.dummy_dim
        )
        embedder = DummyEmbedder(dim=dim)
    else:
        embedder = OpenAIEmbedder(api_key=api_key, model=str(cfg.embeddings.model))

    # ----- generator (usually always OpenAI for now, but can also be made configurable)
    generator = OpenAIChatGenerator(api_key=api_key, model=str(cfg.llm.model))

    # ----- store selection
    store_backend = ovrds.store_backend or cfg.vectorstore.backend  # "memory", "jsonl", or "qdrant"
    if store_backend == "memory":
        store = InMemoryVectorStore()
    elif store_backend == "qdrant":
        from rag.adapters.vectorstores.qdrant_store import QdrantVectorStore

        # Determine vector size based on embedder
        if embedder_backend == "dummy":
            vector_size = ovrds.dummy_embed_dim or cfg.embeddings.dummy_dim
        else:
            # OpenAI embedding dimensions by model
            model = cfg.embeddings.model
            if "3-large" in model:
                vector_size = 3072
            elif "3-small" in model or "ada" in model:
                vector_size = 1536
            else:
                vector_size = 1536  # default fallback

        store = QdrantVectorStore(
            collection_name=ovrds.qdrant_collection or cfg.vectorstore.qdrant_collection,
            vector_size=vector_size,
            url=ovrds.qdrant_url or cfg.vectorstore.qdrant_url,
            path=str(ovrds.qdrant_path or cfg.vectorstore.qdrant_path) if (ovrds.qdrant_path or cfg.vectorstore.qdrant_path) else None,
            api_key=cfg.vectorstore.qdrant_api_key,
        )
    else:
        # jsonl backend
        index_dir = ovrds.jsonl_index_dir or cfg.vectorstore.jsonl_dir or cfg.paths.index_dir
        if index_dir is None:
            raise ValueError(
                "jsonl_index_dir (override) or vectorstore.jsonl_dir (settings) is required"
            )
        store = JsonlVectorStore(path=index_dir)

    # IMPORTANT: retriever must be built from the chosen embedder+store
    retriever = VectorRetriever(embedder=embedder, store=store)
    
    # ----- reranker (optional)
    if not cfg.rerank.enabled or cfg.rerank.backend == "noop":
        reranker = NoOpReranker()
    else:
        reranker = HeuristicReranker()
        
    # ----- logger (for query tracing)
    logger = JsonlQueryLogger(path=cfg.paths.artifacts_dir / "logs" / "queries.jsonl")

    return Container(
        chunker=chunker,
        context_builder=context_builder,
        embedder=embedder,
        generator=generator,
        ingestor=ingestor,
        store=store,
        reranker=reranker,
        retriever=retriever,
        logger=logger,
    )
