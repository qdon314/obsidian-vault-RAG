from __future__ import annotations

import json
import math
import re

from typing import Annotated, Any

from pydantic import BaseModel, Field
from llama_index.core.settings import Settings
from llama_index.core.schema import NodeWithScore

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

_WORD_RE = re.compile(r"[A-Za-z0-9_/-]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(s: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def lexical_best_sentence(query: str, text: str, max_len: int = 260) -> str:
    """Cheap 'rationale': sentence with highest token overlap w.r.t. query."""
    query_token_set = set(_tokenize(query))
    best = ""
    best_score = -1
    for sentence in _SENT_SPLIT_RE.split(text or ""):
        toks = set(_tokenize(sentence))
        
        # Set intersection size is a proxy for relevance.
        score = len(query_token_set & toks)
        if score > best_score:
            best_score = score
            best = sentence.strip()
    if len(best) > max_len:
        best = best[: max_len - 1] + "…"
    return best


def dedupe_by_source(nodes: list[NodeWithScore], key: str = "source_path") -> list[NodeWithScore]:
    """Keep best-scoring node per source_path."""
    nodes = sorted(nodes, key=lambda n: (n.score is None, -(n.score or 0.0)))
    best: dict[str, NodeWithScore] = {}
    for n in nodes:
        meta = n.node.metadata or {}
        k = meta.get(key) or f"__missing__::{getattr(n.node, 'node_id', id(n.node))}"
        if k not in best:
            best[k] = n
    return list(best.values())


def _cosine(a: list[float], b: list[float]) -> float:
    """
    Cosine similarity of two vectors.
    """
    dot_product = sum(x*y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x*x for x in a))
    norm_b = math.sqrt(sum(y*y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)

def mmr_select(
    query: str,
    candidates: list[NodeWithScore],
    k: int,
    lambda_mult: float = 0.7,
) -> list[NodeWithScore]:
    """
    Max Marginal Relevance selection using embeddings (query + node texts).
    Requires Settings.embed_model to be set.
    
    Args:
        query: The query string.
        candidates: List of NodeWithScore objects.
        k: Number of top candidates to return.
        lambda_mult: MMR lambda multiplier (0.0 = no diversity, 1.0 = full diversity).
    
    Returns:
        List of top-k selected NodeWithScore objects.
    """
    if not candidates or k <= 0:
        return []

    embed = Settings.embed_model
    if embed is None:
        # fallback: just return top-k by score
        return sorted(candidates, key=lambda n: (n.score is None, -(n.score or 0.0)))[:k]

    # Prepare embeddings
    qv = embed.get_text_embedding(query)
    texts = []
    for n in candidates:
        t = getattr(n.node, "text", None) or n.node.get_content()
        texts.append(t)

    node_text_embeddings = [embed.get_text_embedding(t[:2000]) for t in texts]  # cap to keep it quick

    # Similarities to query
    sim_q = [_cosine(qv, dv) for dv in node_text_embeddings]

    selected: list[int] = []
    remaining = set(range(len(candidates)))

    while remaining and len(selected) < min(k, len(candidates)):
        best_i = None
        best_val = -1e9
        for i in list(remaining):
            # diversity penalty: similarity to closest selected doc
            if not selected:
                div = 0.0
            else:
                div = max(_cosine(node_text_embeddings[i], node_text_embeddings[j]) for j in selected)

            val = lambda_mult * sim_q[i] - (1.0 - lambda_mult) * div
            if val > best_val:
                best_val = val
                best_i = i

        assert best_i is not None
        selected.append(best_i)
        remaining.remove(best_i)

    return [candidates[i] for i in selected]


def llm_rerank(
    query: str,
    candidates: list[NodeWithScore],
    keep_k: int,
    preview_chars: int = 700,
) -> list[NodeWithScore]:
    """
    LLM rerank: asks the configured Settings.llm to order candidates by relevance.
    """
    if not candidates or keep_k <= 0:
        return []

    llm = Settings.llm
    if llm is None:
        return sorted(candidates, key=lambda n: (n.score is None, -(n.score or 0.0)))[:keep_k]

    items = []
    for i, n in enumerate(candidates):
        meta = n.node.metadata or {}
        text = getattr(n.node, "text", None) or n.node.get_content()
        items.append({
            "i": i,
            "file": meta.get("file_name"),
            "section": meta.get("section_heading"),
            "preview": (text or "")[:preview_chars],
        })

    prompt = (
        "You are reranking retrieval chunks for a RAG system.\n"
        "Given a user query and a list of chunks, return JSON: "
        '{"ranked": [indices...]} with the most relevant first.\n'
        "Do not include any extra keys.\n\n"
        f"Query:\n{query}\n\n"
        f"Chunks:\n{json.dumps(items, ensure_ascii=False)}\n"
    )

    raw = llm.complete(prompt).text
    
    # Try to extract JSON
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return sorted(candidates, key=lambda n: (n.score is None, -(n.score or 0.0)))[:keep_k]

    try:
        obj = json.loads(m.group(0))
        ranked = obj.get("ranked", [])
        ranked = [int(x) for x in ranked if isinstance(x, int) or (isinstance(x, str) and x.isdigit())]
        ranked = [i for i in ranked if 0 <= i < len(candidates)]
        if not ranked:
            raise ValueError("empty ranked")
        # preserve order, unique
        seen = set()
        ordered = []
        for i in ranked:
            if i not in seen:
                ordered.append(candidates[i])
                seen.add(i)
        return ordered[:keep_k]
    except Exception:
        return sorted(candidates, key=lambda n: (n.score is None, -(n.score or 0.0)))[:keep_k]


_RERANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are reranking retrieval chunks for a RAG system. "
            "Return ONLY valid JSON that matches the provided schema. "
            "No extra keys. No commentary.",
        ),
        (
            "user",
            "Query:\n{query}\n\n"
            "Chunks (JSON list):\n{items_json}\n\n"
            "Return JSON like: {{\"ranked\": [0, 2, 1]}}",
        ),
    ]
)


class RerankOut(BaseModel):
    ranked: Annotated[list[int], Field(min_length=1)] = Field(
        ..., description="Indices of chunks ranked most relevant first"
    )
    
def llm_rerank_lcel(
    query: str,
    candidates: list[NodeWithScore],
    keep_k: int,
    *,
    preview_chars: int = 700,
    model: str = "gpt-4.1-mini-2025-04-14",
    temperature: float = 0.0,
    max_retries: int = 1,
) -> list[NodeWithScore]:
    """
    Rerank candidates using LangChain + structured output (Pydantic).
    Falls back to score ordering if model output is invalid.
    """
    if not candidates or keep_k <= 0:
        return []

    # Build candidate items (same as your current function)
    items: list[dict[str, Any]] = []
    for i, n in enumerate(candidates):
        meta = n.node.metadata or {}
        text = getattr(n.node, "text", None) or n.node.get_content()
        items.append(
            {
                "i": i,
                "file": meta.get("file_name"),
                "section": meta.get("section_heading"),
                "preview": (text or "")[:preview_chars],
            }
        )

    # Deterministic fallback
    fallback = sorted(
        candidates, 
        key=lambda n: (n.score is None, -(n.score or 0.0))
    )[:keep_k]

    llm = ChatOpenAI(model=model, temperature=temperature)

    # Structured output wrapper
    chain = _RERANK_PROMPT | llm.with_structured_output(RerankOut)

    items_json = json.dumps(items, ensure_ascii=False)

    for attempt in range(max_retries + 1):
        try:
            result: Any = chain.invoke({"query": query, "items_json": items_json})
            if isinstance(result, RerankOut):
                out = result
            else:
                out = RerankOut.model_validate(result)
            
            ranked = [i for i in out.ranked if 0 <= i < len(candidates)]

            if not ranked:
                return fallback

            # Preserve order, unique indices
            seen = set()
            ordered: list[NodeWithScore] = []
            for idx in ranked:
                if idx not in seen:
                    ordered.append(candidates[idx])
                    seen.add(idx)

            return ordered[:keep_k] if ordered else fallback
        except Exception:
            # On failure, try once more; then fall back
            if attempt >= max_retries:
                return fallback

    return fallback