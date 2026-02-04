from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from dataclasses_json import DataClassJsonMixin, config

from rag.domain.filter_serde import deserialize_filter
from rag.domain.filters import Where
from rag.utils.dataclass_json_utils import (
    enum_encoder,
    make_enum_decoder,
    set_decoder,
    set_encoder,
)

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Types of queries in the evaluation set."""

    FACTUAL = "factual"  # Simple fact lookup
    COMPARISON = "comparison"  # Comparing two or more concepts
    AGGREGATION = "aggregation"  # Requires synthesizing multiple chunks
    PROCEDURAL = "procedural"  # How-to questions
    DEFINITION = "definition"  # What is X?
    CAUSAL = "causal"  # Why/how questions requiring reasoning
    TEMPORAL = "temporal"  # Time-based queries
    NEGATION = "negation"  # Questions about what is NOT in the vault
    MULTI_HOP = "multi_hop"  # Requires connecting multiple pieces of information


class Difficulty(str, Enum):
    """Difficulty levels for queries."""

    EASY = "easy"  # Direct match, single chunk
    MEDIUM = "medium"  # Requires 2-3 chunks or some reasoning
    HARD = "hard"  # Multi-hop reasoning, synthesis across many chunks


# Pre-create decoders with defaults for this module
_query_type_decoder = make_enum_decoder(QueryType, default=QueryType.FACTUAL)
_difficulty_decoder = make_enum_decoder(Difficulty, default=Difficulty.EASY)


@dataclass(frozen=True, slots=True)
class QuerySuggestion:
    """A suggested query generated from a chunk by an LLM."""

    query: str
    query_type: QueryType
    difficulty: Difficulty
    requires_synthesis: bool
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class EvalQuery(DataClassJsonMixin):
    """
    A single evaluation query with ground truth annotations.

    This is the core evaluation unit. Each query should have:
    - A unique ID for tracking
    - The actual query string
    - Ground truth relevant chunks
    - Optional expected answer and metadata
    """

    qid: str
    query: str
    relevant_chunk_ids: set[str] = field(
        default_factory=set,
        metadata=config(encoder=set_encoder, decoder=set_decoder),
    )

    # Optional: expected answer for answer quality evaluation
    expected_answer: str | None = None
    expected_answer_alternatives: list[str] = field(default_factory=list)

    # Query characteristics
    query_type: QueryType = field(
        default=QueryType.FACTUAL,
        metadata=config(encoder=enum_encoder, decoder=_query_type_decoder),
    )
    difficulty: Difficulty = field(
        default=Difficulty.EASY,
        metadata=config(encoder=enum_encoder, decoder=_difficulty_decoder),
    )
    requires_synthesis: bool = False  # Does answer require combining multiple chunks?

    # Additional context
    notes: str | None = None  # Free-form notes for evaluators
    tags: list[str] = field(default_factory=list)  # Topical tags
    created_at: str | None = None
    created_by: str | None = None  # "manual", "synthetic", or evaluator name

    # Negative examples
    is_unanswerable: bool = False  # Should the system abstain?
    unanswerable_reason: str | None = None  # "not_in_corpus", "ambiguous", etc.

    metadata: dict[str, Any] = field(default_factory=dict)

    def get_filter(self) -> Where:
        """
        Extract and deserialize the retrieval filter from metadata.

        The filter should be stored in metadata['filter'] as a serialized
        dictionary matching the Filter DSL format (e.g., {'type': 'Eq', 'field': 'doc_id', 'value': 'abc'}).

        Returns None if no filter is present. Logs a warning and returns None
        if deserialization fails, allowing graceful degradation.

        Returns:
            Filter object for use with Retriever.retrieve(where=...), or None

        Example:
            >>> query = EvalQuery(
            ...     qid="q1", query="test", relevant_chunk_ids={"c1"},
            ...     metadata={"filter": {"type": "Eq", "field": "doc_id", "value": "doc123"}}
            ... )
            >>> query.get_filter()
            Eq(field='doc_id', value='doc123')
        """
        filter_data = self.metadata.get("filter")
        if filter_data is None:
            return None

        try:
            return deserialize_filter(filter_data)
        except (ValueError, KeyError, TypeError) as e:
            logger.warning(f"Failed to deserialize filter for query {self.qid}: {e}")
            return None


@dataclass(frozen=True, slots=True)
class EvalDataset(DataClassJsonMixin):
    """
    A collection of evaluation queries with metadata.
    """

    name: str
    version: str
    description: str
    queries: list[EvalQuery] = field(default_factory=list)
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def filter_by_type(self, query_type: QueryType) -> list[EvalQuery]:
        """Filter queries by type."""
        return [q for q in self.queries if q.query_type == query_type]

    def filter_by_difficulty(self, difficulty: Difficulty) -> list[EvalQuery]:
        """Filter queries by difficulty."""
        return [q for q in self.queries if q.difficulty == difficulty]

    def filter_by_tags(self, tags: set[str]) -> list[EvalQuery]:
        """Filter queries that have any of the given tags."""
        return [q for q in self.queries if set(q.tags) & tags]

    def stats(self) -> dict[str, Any]:
        """Get dataset statistics."""
        total = len(self.queries)
        if total == 0:
            return {"total": 0}

        return {
            "total": total,
            "by_type": {
                qt.value: len([q for q in self.queries if q.query_type == qt]) for qt in QueryType
            },
            "by_difficulty": {
                d.value: len([q for q in self.queries if q.difficulty == d]) for d in Difficulty
            },
            "answerable": len([q for q in self.queries if not q.is_unanswerable]),
            "unanswerable": len([q for q in self.queries if q.is_unanswerable]),
            "requires_synthesis": len([q for q in self.queries if q.requires_synthesis]),
            "with_expected_answer": len([q for q in self.queries if q.expected_answer]),
        }


# Example template for manual curation
EXAMPLE_QUERIES = [
    EvalQuery(
        qid="example_001",
        query="What are the main benefits of X?",
        relevant_chunk_ids={"doc123:fixed_chars_v1:0:0-500", "doc456:fixed_chars_v1:2:1000-1500"},
        expected_answer="The main benefits include A, B, and C...",
        query_type=QueryType.FACTUAL,
        difficulty=Difficulty.EASY,
        requires_synthesis=False,
        notes="Simple factual query requiring single chunk",
        tags=["benefits", "concept-x"],
        created_by="manual",
    ),
    EvalQuery(
        qid="example_002",
        query="How does X compare to Y?",
        relevant_chunk_ids={"doc789:fixed_chars_v1:0:0-800", "doc789:fixed_chars_v1:1:800-1600"},
        expected_answer="X differs from Y in that...",
        query_type=QueryType.COMPARISON,
        difficulty=Difficulty.MEDIUM,
        requires_synthesis=True,
        notes="Requires comparing information from multiple chunks",
        tags=["comparison", "concept-x", "concept-y"],
        created_by="manual",
    ),
    EvalQuery(
        qid="example_003",
        query="What is the capital of Atlantis?",
        relevant_chunk_ids=set(),
        expected_answer=None,
        query_type=QueryType.FACTUAL,
        difficulty=Difficulty.EASY,
        is_unanswerable=True,
        unanswerable_reason="not_in_corpus",
        notes="Negative example - topic not in vault",
        tags=["negative-example"],
        created_by="manual",
    ),
]
