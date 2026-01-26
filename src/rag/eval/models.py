from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dataclasses_json import DataClassJsonMixin, config

from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.schema import Difficulty, QueryType
from rag.utils.enum_parse import parse_enum

if TYPE_CHECKING:
    from rag.domain.models import Answer


def _tuple_encoder(val: tuple[str, ...]) -> list[str]:
    return list(val)


def _tuple_decoder(val: list[str] | None) -> tuple[str, ...]:
    return tuple(val) if val else ()


def _set_encoder(val: set[str]) -> list[str]:
    return sorted(val)


def _set_decoder(val: list[str] | None) -> set[str]:
    return set(val) if val else set()

def _dt_from_iso_or_now(v: Any) -> datetime:
    if isinstance(v, datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.fromisoformat(v)
        except ValueError:
            pass
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class RetrievalResult(DataClassJsonMixin):
    qid: str
    retrieved_chunk_ids: tuple[str, ...] = field(
        default_factory=tuple,
        metadata=config(encoder=_tuple_encoder, decoder=_tuple_decoder),
    )
    relevant_chunk_ids: set[str] = field(
        default_factory=set,
        metadata=config(encoder=_set_encoder, decoder=_set_decoder),
    )

    @classmethod
    def from_qid_dict(cls, qid: str, data: dict[str, Any]) -> RetrievalResult:
        """Create from dict + qid (for deserialization where qid comes from context)."""
        return cls(
            qid=qid,
            retrieved_chunk_ids=tuple(data.get("retrieved_chunk_ids", [])),
            relevant_chunk_ids=set(data.get("relevant_chunk_ids", [])),
        )

def _parse_flattened_metrics(data: dict[str, Any], prefix: str) -> dict[int, float]:
    """Extract metrics like 'recall@10' -> {10: 0.42}."""
    result = {}
    for key, value in data.items():
        if key.startswith(f"{prefix}@"):
            k = int(key.split("@")[1])
            result[k] = float(value)
    return result


@dataclass(frozen=True, slots=True)
class RetrievalSummary(DataClassJsonMixin):
    """Aggregate retrieval metrics over a set of queries."""
    num_queries: int
    avg_retrieved: float

    # Per-K metrics
    recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)

    # Global ranking metrics
    mrr: float = 0.0
    map: float = 0.0

    def to_flat_dict(self) -> dict[str, float]:
        """
        Flatten to raw dict style (e.g., 'recall@10': 0.42) for human-readable JSON.
        """
        out: dict[str, float] = {
            "num_queries": float(self.num_queries),
            "avg_retrieved": float(self.avg_retrieved),
            "mrr": float(self.mrr),
            "map": float(self.map),
        }
        for k, v in self.recall_at_k.items():
            out[f"recall@{k}"] = float(v)
        for k, v in self.precision_at_k.items():
            out[f"precision@{k}"] = float(v)
        for k, v in self.hit_rate_at_k.items():
            out[f"hit_rate@{k}"] = float(v)
        for k, v in self.ndcg_at_k.items():
            out[f"ndcg@{k}"] = float(v)
        return out

    @classmethod
    def from_flat_dict(cls, data: dict[str, float]) -> RetrievalSummary:
        """
        Create from flattened dict style (e.g., 'recall@10': 0.42).
        """
        return cls(
            num_queries=int(data.get("num_queries", 0)),
            avg_retrieved=float(data.get("avg_retrieved", 0.0)),
            recall_at_k=_parse_flattened_metrics(data, "recall"),
            precision_at_k=_parse_flattened_metrics(data, "precision"),
            hit_rate_at_k=_parse_flattened_metrics(data, "hit_rate"),
            ndcg_at_k=_parse_flattened_metrics(data, "ndcg"),
            mrr=float(data.get("mrr", 0.0)),
            map=float(data.get("map", 0.0)),
        )
   
@dataclass
class EvalResult:
    """Complete evaluation result for a single query."""
    qid: str
    query: str

    # Retrieval results
    retrieval_result: RetrievalResult

    # Answer (if generated)
    answer: Answer | None = None
    answer_metrics: AnswerQualityMetrics | None = None

    # Query metadata
    query_type: QueryType | None = None
    difficulty: Difficulty | None = None
    is_unanswerable: bool = False

    # Timing
    latency_ms: int | None = None

    trace_id: str | None = None  # link to QueryTrace if available

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalResult:
        # Import here to avoid circular dependency
        from rag.domain.models import Answer as AnswerCls

        qid = data["qid"]
        query = data.get("query", "")

        # Parse retrieval result
        retrieval_data = data.get("retrieval", {})
        retrieval = RetrievalResult.from_dict(qid, **retrieval_data)

        # Parse answer if present
        answer = None
        answer_data = data.get("answer")
        if answer_data:
            answer = AnswerCls.from_query_dict(query, answer_data)

        # Parse answer metrics if present
        answer_metrics = None
        metrics_data = data.get("answer_metrics")
        if metrics_data:
            answer_metrics = AnswerQualityMetrics.from_dict(metrics_data)

        # Parse query type and difficulty
        query_type = None
        if data.get("query_type"):
            query_type = parse_enum(QueryType, data["query_type"])

        difficulty = None
        if data.get("difficulty"):
            difficulty = parse_enum(Difficulty, data["difficulty"])

        return cls(
            qid=qid,
            query=query,
            retrieval_result=retrieval,
            answer=answer,
            answer_metrics=answer_metrics,
            query_type=query_type,
            difficulty=difficulty,
            is_unanswerable=data.get("is_unanswerable", False),
            latency_ms=data.get("latency_ms"),
            trace_id=data.get("trace_id"),
        )
    

@dataclass(frozen=True, slots=True)
class EvalRunMeta(DataClassJsonMixin):
    run_id: str = "unknown"
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
        metadata=config(decoder=_dt_from_iso_or_now, encoder=lambda dt: dt.isoformat()),
    )
    queries_path: str | None = None
    top_k: int = 10
    keep_k: int | None = None
    token_budget: int = 1500
    run_generation: bool = False
    use_llm_judge: bool = False
    judge_model: str | None = None
    generator_model: str | None = None
    embedder_model: str | None = None
    reranker_name: str | None = None
    notes: str | None = None
    gold_judge_version: str | None = None
    groundedness_judge_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalAggregates:
    overall: RetrievalSummary
    by_type: dict[str, RetrievalSummary]
    by_difficulty: dict[str, RetrievalSummary]

    # Optional: add later
    answer_quality: dict[str, float] | None = None
    latency_ms: dict[str, float] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalAggregates:
        overall = RetrievalSummary.from_dict(data.get("overall", {}))

        by_type: dict[str, RetrievalSummary] = {}
        for type_name, type_data in data.get("by_type", {}).items():
            by_type[type_name] = RetrievalSummary.from_dict(type_data)

        by_difficulty: dict[str, RetrievalSummary] = {}
        for diff_name, diff_data in data.get("by_difficulty", {}).items():
            by_difficulty[diff_name] = RetrievalSummary.from_dict(diff_data)

        return cls(
            overall=overall,
            by_type=by_type,
            by_difficulty=by_difficulty,
            answer_quality=data.get("answer_quality"),
            latency_ms=data.get("latency_ms"),
        )

@dataclass(frozen=True, slots=True)
class EvalRun:
    meta: EvalRunMeta
    results: tuple[EvalResult, ...]
    aggregates: EvalAggregates
    # optional: where things were written
    artifacts: dict[str, str] | None = None  # e.g. {"results_jsonl": "...", "metrics_json": "..."}
    
    
