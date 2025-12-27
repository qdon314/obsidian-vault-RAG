from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Sequence


# -------------------------
# Core content objects
# -------------------------

@dataclass(frozen=True, slots=True)
class Document:
    """
    A raw source unit before chunking.

    doc_id should be stable (same input -> same doc_id), e.g. hash(source + path + content).
    """
    doc_id: str
    text: str
    source: str  # e.g. "filesystem", "web", "notion", "github"
    uri: str     # path or URL
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass(frozen=True, slots=True)
class Chunk:
    """
    A piece of a Document used for embedding/retrieval.

    chunk_id should be stable given (doc_id, chunk_index, chunking_strategy, offsets).
    """
    chunk_id: str
    doc_id: str
    text: str

    # Provenance within the document
    chunk_index: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None

    # Helpful for markdown/code corpora
    section_heading: Optional[str] = None
    section_path: Optional[str] = None  # e.g. "H1 > H2 > H3"
    language: Optional[str] = None      # e.g. "python", "markdown"

    metadata: Mapping[str, Any] = field(default_factory=dict)


# -------------------------
# Retrieval / ranking objects
# -------------------------

@dataclass(frozen=True, slots=True)
class Candidate:
    """
    A retrieved chunk plus scores from retrieval and optional reranking.

    score: retrieval similarity score (higher is better).
    rerank_score: optional reranker score (higher is better).
    """
    chunk: Chunk
    score: float
    rerank_score: Optional[float] = None

    # Optional: store "why" for debugging (LLM reranker rationale, match highlights, etc.)
    debug: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation:
    """
    A pointer to a source used in the final answer.
    """
    chunk_id: str
    doc_id: str
    uri: str
    quote: Optional[str] = None  # small excerpt used/displayed
    section_heading: Optional[str] = None
    section_path: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ContextPack:
    """
    The final set of evidence given to the generator, plus the exact rendered context string.
    """
    query: str
    chunks: Sequence[Chunk]
    rendered_context: str
    citations: Sequence[Citation]
    token_budget: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


# -------------------------
# Output objects
# -------------------------

@dataclass(frozen=True, slots=True)
class Answer:
    """
    Final model output (or abstention).
    """
    query: str
    text: str
    citations: Sequence[Citation] = field(default_factory=tuple)
    abstained: bool = False
    confidence: Optional[float] = None  # optional; only if you compute one
    metadata: Mapping[str, Any] = field(default_factory=dict)


# -------------------------
# Query tracing
# -------------------------

@dataclass(frozen=True, slots=True)
class QueryTrace:
    """
    A structured record for observability + evaluation.
    Log this per query (JSONL), and you can debug everything.
    """
    trace_id: str
    query: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    # Retrieval
    top_k: int = 10
    retrieved: Sequence[Candidate] = field(default_factory=tuple)

    # Context build
    token_budget: int = 0
    packed_chunk_ids: Sequence[str] = field(default_factory=tuple)

    # Generation
    model: Optional[str] = None
    latency_ms: Optional[int] = None
    estimated_cost_usd: Optional[float] = None

    # Final
    answer: Optional[Answer] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
