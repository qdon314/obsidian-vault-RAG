from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


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


@dataclass(frozen=True, slots=True)
class EvalQuery:
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
    relevant_chunk_ids: set[str]

    # Optional: expected answer for answer quality evaluation
    expected_answer: str | None = None
    expected_answer_alternatives: list[str] = field(default_factory=list)

    # Query characteristics
    query_type: QueryType = QueryType.FACTUAL
    difficulty: Difficulty = Difficulty.EASY
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

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "qid": self.qid,
            "query": self.query,
            "relevant_chunk_ids": sorted(self.relevant_chunk_ids),
            "expected_answer": self.expected_answer,
            "expected_answer_alternatives": self.expected_answer_alternatives,
            "query_type": self.query_type.value,
            "difficulty": self.difficulty.value,
            "requires_synthesis": self.requires_synthesis,
            "notes": self.notes,
            "tags": self.tags,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "is_unanswerable": self.is_unanswerable,
            "unanswerable_reason": self.unanswerable_reason,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalQuery:
        """Create from dictionary."""
        return cls(
            qid=str(data["qid"]),
            query=str(data["query"]),
            relevant_chunk_ids=set(data.get("relevant_chunk_ids", [])),
            expected_answer=data.get("expected_answer"),
            expected_answer_alternatives=data.get("expected_answer_alternatives", []),
            query_type=QueryType(data.get("query_type", "factual")),
            difficulty=Difficulty(data.get("difficulty", "easy")),
            requires_synthesis=data.get("requires_synthesis", False),
            notes=data.get("notes"),
            tags=data.get("tags", []),
            created_at=data.get("created_at"),
            created_by=data.get("created_by"),
            is_unanswerable=data.get("is_unanswerable", False),
            unanswerable_reason=data.get("unanswerable_reason"),
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class EvalDataset:
    """
    A collection of evaluation queries with metadata.
    """
    name: str
    version: str
    description: str
    queries: list[EvalQuery]
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "queries": [q.to_dict() for q in self.queries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalDataset:
        """Create from dictionary."""
        return cls(
            name=data["name"],
            version=data["version"],
            description=data["description"],
            created_at=data.get("created_at"),
            metadata=data.get("metadata", {}),
            queries=[EvalQuery.from_dict(q) for q in data.get("queries", [])],
        )

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
                qt.value: len([q for q in self.queries if q.query_type == qt])
                for qt in QueryType
            },
            "by_difficulty": {
                d.value: len([q for q in self.queries if q.difficulty == d])
                for d in Difficulty
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
