from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from dataclasses_json import DataClassJsonMixin

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
class Chunk(DataClassJsonMixin):
    """
    A piece of a Document used for embedding/retrieval.

    chunk_id should be stable given (doc_id, chunk_index, chunking_strategy, offsets).
    """
    chunk_id: str
    doc_id: str
    text: str

    # Provenance within the document
    chunk_index: int
    start_char: int | None = None
    end_char: int | None = None

    # Helpful for markdown/code corpora
    section_heading: str | None = None
    section_path: str | None = None  # e.g. "H1 > H2 > H3"
    language: str | None = None      # e.g. "python", "markdown"

    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def to_record(cls, ch: Chunk) -> dict[str, object]:
        """Convert Chunk to a flat record for filtering."""
        # Start with "first-class" fields
        rec: dict[str, object] = {
            "chunk_id": ch.chunk_id,
            "doc_id": ch.doc_id,
            "chunk_index": ch.chunk_index,
            "start_char": ch.start_char,
            "end_char": ch.end_char,
            "section_heading": ch.section_heading,
            "section_path": ch.section_path,
            "language": ch.language,
        }
        
        # Add metadata fields
        for k, v in ch.metadata.items():
            rec[k] = v
        
        return rec

# --------------------------
# Core Content Metadata Objects
# --------------------------

@dataclass(frozen=True, slots=True)
class IngestReport:
    """
    Summary statistics from an ingestion run.
    """
    scanned: int
    loaded: int
    skipped_hidden: int
    skipped_extension: int
    skipped_too_large: int
    skipped_empty: int
    failed: int
    by_extension: Mapping[str, int] = field(default_factory=dict)

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
    rerank_score: float | None = None

    # Optional: store "why" for debugging (LLM reranker rationale, match highlights, etc.)
    debug: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Citation(DataClassJsonMixin):
    """
    A pointer to a source used in the final answer.
    """
    chunk_id: str
    doc_id: str
    uri: str | None = None
    quote: str | None = None  # small excerpt used/displayed
    section_heading: str | None = None
    section_path: str | None = None
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


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
class Answer(DataClassJsonMixin):
    """
    Final model output (or abstention).
    """
    query: str
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    abstained: bool = False
    confidence: float | None = None  # optional; only if you compute one
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_query_dict(cls, query: str, data: dict[str, Any]) -> Answer:
        return cls.from_dict({**data, "query": query})
    
@dataclass(frozen=True, slots=True)
class QueryRunResult:
    trace_id: str
    answer: Answer
    context_pack: ContextPack
    retrieved_chunk_ids: tuple[str, ...]
    reranked_chunk_ids: tuple[str, ...]
    packed_chunk_ids: tuple[str, ...]
    latency_ms: int

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
    created_at: datetime = field(
        default_factory=lambda: datetime.now(UTC)
    )

    # Retrieval
    top_k: int = 10
    retrieved: Sequence[Candidate] = field(default_factory=tuple)
    
    # Rerank
    reranked: Sequence[Candidate] = field(default_factory=tuple)
    keep_k: int | None = None
    reranker: str | None = None


    # Context build
    token_budget: int = 0
    packed_chunk_ids: Sequence[str] = field(default_factory=tuple)

    # Generation
    model: str | None = None
    latency_ms: int | None = None
    estimated_cost_usd: float | None = None

    # Final
    answer: Answer | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
