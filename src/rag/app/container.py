from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from rag import settings
from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.chunking.obsidian_structural import ObsidianStructuralChunker
from rag.adapters.chunking.proposition.backends import T5Propositionizer
from rag.adapters.chunking.proposition.chunker import ObsidianPropositionChunker
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
from rag.domain.models import Chunk
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
    cache_embeddings: bool | None = None  # None = use settings, True/False = override
    dummy_embed_dim: int | None = None

    store_backend: Literal["memory", "jsonl", "qdrant"] | None = None
    jsonl_index_dir: Path | None = None

    # Qdrant overrides
    qdrant_collection: str | None = None
    qdrant_url: str | None = None
    qdrant_path: Path | None = None

    # Fixed chunker overrides
    chunk_size: int | None = None
    chunk_overlap: int | None = None

    # Obsidian structural chunker overrides
    target_chars: int | None = None
    hard_max_chars: int | None = None
    overlap_blocks: int | None = None
    include_heading_preamble: bool | None = None

    vault_dir: Path | None = None

    retrieval_backend: Literal["vector", "hybrid"] | None = None
    top_k: int | None = None
    rerank_backend: Literal["heuristic", "noop"] | None = None
    rerank_enabled: bool | None = None

    logs_directory: Path | None = None


def _get_store_chunks(store: VectorStore) -> Sequence[Chunk]:
    """Access loaded chunks from an in-memory or JSONL vector store.

    Hybrid retrieval requires direct chunk access for BM25 indexing.
    Only stores that hold chunks in memory (InMemory, JSONL) are supported.

    TODO: Replace private-attribute access with a public ChunkLoader port
    or a ``list_chunks()`` method on VectorStore to respect the hexagonal
    architecture boundary.
    """
    chunks = getattr(store, "_chunks", None)
    if chunks is None:
        raise TypeError(
            "Hybrid retrieval requires a store with in-memory chunk access "
            "(memory or jsonl). The current store does not expose chunks."
        )
    return chunks


def build_container(
    *,
    cfg: settings.Settings | None = None,
    overrides: ContainerOverrides | None = None,
) -> Container:
    cfg = cfg or settings.load_settings()
    ovrds = overrides or ContainerOverrides()

    # ----- chunking (defaults from settings, overridden by CLI)
    chunker: Chunker
    if cfg.chunking.backend == "obsidian_structural":
        target_chars = (
            ovrds.target_chars if ovrds.target_chars is not None else cfg.chunking.target_chars
        )
        hard_max_chars = (
            ovrds.hard_max_chars
            if ovrds.hard_max_chars is not None
            else cfg.chunking.hard_max_chars
        )
        overlap_blocks = (
            ovrds.overlap_blocks
            if ovrds.overlap_blocks is not None
            else cfg.chunking.overlap_blocks
        )
        include_heading_preamble = (
            ovrds.include_heading_preamble
            if ovrds.include_heading_preamble is not None
            else cfg.chunking.include_heading_preamble
        )
        chunker = ObsidianStructuralChunker(
            target_chars=target_chars,
            hard_max_chars=hard_max_chars,
            overlap_blocks=overlap_blocks,
            include_heading_preamble=include_heading_preamble,
        )
    elif cfg.chunking.backend == "obsidian_proposition":
        target_chars = (
            ovrds.target_chars if ovrds.target_chars is not None else cfg.chunking.target_chars
        )
        hard_max_chars = (
            ovrds.hard_max_chars
            if ovrds.hard_max_chars is not None
            else cfg.chunking.hard_max_chars
        )
        overlap_blocks = (
            ovrds.overlap_blocks
            if ovrds.overlap_blocks is not None
            else cfg.chunking.overlap_blocks
        )
        include_heading_preamble = (
            ovrds.include_heading_preamble
            if ovrds.include_heading_preamble is not None
            else cfg.chunking.include_heading_preamble
        )
        chunker = ObsidianPropositionChunker(
            propositionizer=T5Propositionizer(batch_size=cfg.chunking.proposition_batch_size),
            overlap_blocks=overlap_blocks,
            include_heading_preamble=include_heading_preamble,
        )
    else:
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
    embedder: Embedder
    embedder_backend = ovrds.embedder_backend or cfg.embeddings.backend  # e.g. "openai"
    if embedder_backend == "dummy":
        dim = (
            ovrds.dummy_embed_dim if ovrds.dummy_embed_dim is not None else cfg.embeddings.dummy_dim
        )
        embedder = DummyEmbedder(dim=dim)
    else:
        embedder = OpenAIEmbedder(
            api_key=api_key,
            model=str(cfg.embeddings.model),
            timeout=cfg.embeddings.timeout,
            max_retries=cfg.embeddings.max_retries,
        )

    # ----- optional embedding cache (wraps the embedder with SQLite disk cache)
    cache_enabled = (
        cfg.embeddings.cache_embeddings
        if ovrds.cache_embeddings is None
        else ovrds.cache_embeddings
    )
    if cache_enabled:
        from rag.adapters.embedding.sqlite_cache import CachedEmbedder

        cache_db_path = cfg.embeddings.cache_db_path or (
            cfg.paths.artifacts_dir / "embedding_cache.db"
        )
        embedder = CachedEmbedder(embedder=embedder, db_path=cache_db_path)

    # ----- generator (usually always OpenAI for now, but can also be made configurable)
    generator = OpenAIChatGenerator(
        api_key=api_key,
        model=str(cfg.llm.model),
        temperature=cfg.llm.temperature,
        timeout=cfg.llm.timeout,
        max_retries=cfg.llm.max_retries,
    )

    # ----- store selection
    store: VectorStore
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
            path=str(ovrds.qdrant_path or cfg.vectorstore.qdrant_path)
            if (ovrds.qdrant_path or cfg.vectorstore.qdrant_path)
            else None,
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

    # ----- retriever selection
    retriever: Retriever
    retriever_backend = ovrds.retrieval_backend or cfg.retrieval.backend
    if retriever_backend == "hybrid":
        from rag.adapters.retrieval.bm25_retriever import BM25Retriever
        from rag.adapters.retrieval.hybrid_retriever import HybridRetriever

        vector_retriever = VectorRetriever(embedder=embedder, store=store)
        bm25 = BM25Retriever(
            k1=cfg.retrieval.hybrid_bm25_k1,
            b=cfg.retrieval.hybrid_bm25_b,
            chunk_source=lambda: _get_store_chunks(store),
        )
        retriever = HybridRetriever(
            primary=vector_retriever,
            secondary=bm25,
            primary_weight=cfg.retrieval.hybrid_primary_weight,
            secondary_weight=cfg.retrieval.hybrid_secondary_weight,
            rrf_k=cfg.retrieval.hybrid_rrf_k,
        )
    else:
        retriever = VectorRetriever(embedder=embedder, store=store)

    # ----- reranker (optional)
    reranker: Reranker
    if not cfg.rerank.enabled or cfg.rerank.backend == "noop":
        reranker = NoOpReranker()
    else:
        reranker = HeuristicReranker()

    # ----- logger (for query tracing)
    log_path = ovrds.logs_directory or (cfg.paths.artifacts_dir / "logs")
    logger = JsonlQueryLogger(path=log_path / "traces.jsonl")

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
