from __future__ import annotations

from typing import List, Tuple
from llama_index.core import VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilters, ExactMatchFilter
from llama_index.core.schema import NodeWithScore
from llama_index.core.settings import Settings

from src.rag.profiles import RetrievalConfig
from src.rag.retrieval import dedupe_by_source, mmr_select, llm_rerank

def retrieve_nodes(index: VectorStoreIndex, query: str, cfg: RetrievalConfig) -> tuple[list[NodeWithScore], list[NodeWithScore]]:
    """
    Returns:
      raw_nodes: initial retriever output (size ~= cfg.retrieve_k)
      final_nodes: nodes actually used for context (size <= cfg.context_k)
    """
    # Retriever + query engine
    filters = []
    if cfg.ai_only:
        filters.append(ExactMatchFilter(key="is_ai", value="True"))
    
    if not cfg.include_moc:
        filters.append(ExactMatchFilter(key="classification", value="note"))

    retriever = index.as_retriever(similarity_top_k=cfg.retrieve_k, filters=MetadataFilters(filters=filters))
    raw_nodes = retriever.retrieve(query)
    nodes = raw_nodes
    
    # Sort by score (defensive)
    nodes = sorted(nodes, key=lambda n: (n.score is None, -(n.score or 0.0)))

    # Dedupe by file
    if cfg.dedupe:
        nodes = dedupe_by_source(nodes, key="source_path")

    # MMR diversity
    if cfg.mmr:
        # Ensure embed_model exists if you rely on MMR
        if Settings.embed_model is None:
            # If you want MMR, set Settings.embed_model in your runtime too.
            pass
        nodes = mmr_select(query, nodes, k=min(len(nodes), cfg.retrieve_k), lambda_mult=cfg.mmr_lambda)

    # Rerank (LLM)
    if cfg.rerank:
        candidates = nodes[: cfg.rerank_candidates]
        nodes = llm_rerank(query, candidates, keep_k=cfg.context_k)

    final_nodes = nodes[: cfg.context_k]
    return raw_nodes, final_nodes
