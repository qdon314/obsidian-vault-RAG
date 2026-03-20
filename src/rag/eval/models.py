from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from dataclasses_json import DataClassJsonMixin, config

from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.judges import GroundednessJudgeResult
from rag.eval.reducers import OutcomeLabel
from rag.eval.schema import Difficulty, QueryType
from rag.utils.dataclass_json_utils import (
    datetime_decoder,
    datetime_encoder,
    enum_encoder,
    make_enum_decoder,
    set_decoder,
    set_encoder,
    tuple_decoder,
    tuple_encoder,
)
from rag.utils.enum_parse import parse_enum

if TYPE_CHECKING:
    from rag.domain.models import Answer


# Pre-create decoders for optional enum fields
_query_type_decoder = make_enum_decoder(QueryType)
_difficulty_decoder = make_enum_decoder(Difficulty)
_outcome_label_decoder = make_enum_decoder(OutcomeLabel)  # for future use


@dataclass(frozen=True, slots=True)
class RetrievalResult(DataClassJsonMixin):
    qid: str
    retrieved_chunk_ids: tuple[str, ...] = field(
        default_factory=tuple,
        metadata=config(encoder=tuple_encoder, decoder=tuple_decoder),
    )
    relevant_chunk_ids: set[str] = field(
        default_factory=set,
        metadata=config(encoder=set_encoder, decoder=set_decoder),
    )
    critical_chunk_ids: set[str] = field(
        default_factory=set,
        metadata=config(encoder=set_encoder, decoder=set_decoder),
    )
    supporting_chunk_ids: set[str] = field(
        default_factory=set,
        metadata=config(encoder=set_encoder, decoder=set_decoder),
    )
    context_chunk_ids: set[str] = field(
        default_factory=set,
        metadata=config(encoder=set_encoder, decoder=set_decoder),
    )

    @classmethod
    def from_qid_dict(cls, qid: str, data: dict[str, Any]) -> RetrievalResult:
        """Create from dict + qid (for deserialization where qid comes from context)."""
        return cls(
            qid=qid,
            retrieved_chunk_ids=tuple(data.get("retrieved_chunk_ids", [])),
            relevant_chunk_ids=set(data.get("relevant_chunk_ids", [])),
            critical_chunk_ids=set(data.get("critical_chunk_ids", [])),
            supporting_chunk_ids=set(data.get("supporting_chunk_ids", [])),
            context_chunk_ids=set(data.get("context_chunk_ids", [])),
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
    capped_recall_at_k: dict[int, float] = field(default_factory=dict)
    precision_at_k: dict[int, float] = field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)
    ndcg_at_k: dict[int, float] = field(default_factory=dict)
    critical_recall_at_k: dict[int, float] = field(default_factory=dict)
    weighted_recall_at_k: dict[int, float] = field(default_factory=dict)
    critical_hit_rate_at_k: dict[int, float] = field(default_factory=dict)

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
        for k, v in self.capped_recall_at_k.items():
            out[f"capped_recall@{k}"] = float(v)
        for k, v in self.precision_at_k.items():
            out[f"precision@{k}"] = float(v)
        for k, v in self.hit_rate_at_k.items():
            out[f"hit_rate@{k}"] = float(v)
        for k, v in self.ndcg_at_k.items():
            out[f"ndcg@{k}"] = float(v)
        for k, v in self.critical_recall_at_k.items():
            out[f"critical_recall@{k}"] = float(v)
        for k, v in self.weighted_recall_at_k.items():
            out[f"weighted_recall@{k}"] = float(v)
        for k, v in self.critical_hit_rate_at_k.items():
            out[f"critical_hit_rate@{k}"] = float(v)
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
            capped_recall_at_k=_parse_flattened_metrics(data, "capped_recall"),
            precision_at_k=_parse_flattened_metrics(data, "precision"),
            hit_rate_at_k=_parse_flattened_metrics(data, "hit_rate"),
            ndcg_at_k=_parse_flattened_metrics(data, "ndcg"),
            critical_recall_at_k=_parse_flattened_metrics(data, "critical_recall"),
            weighted_recall_at_k=_parse_flattened_metrics(data, "weighted_recall"),
            critical_hit_rate_at_k=_parse_flattened_metrics(data, "critical_hit_rate"),
            mrr=float(data.get("mrr", 0.0)),
            map=float(data.get("map", 0.0)),
        )


@dataclass
class EvalResult(DataClassJsonMixin):
    """Complete evaluation result for a single query."""

    qid: str
    query: str

    # Retrieval results
    retrieval_result: RetrievalResult

    # Answer (if generated)
    answer: Answer | None = None
    answer_metrics: AnswerQualityMetrics | None = None
    groundedness_result: GroundednessJudgeResult | None = None
    outcome_label: OutcomeLabel | None = field(
        default=None,
        metadata=config(encoder=enum_encoder, decoder=_outcome_label_decoder),
    )

    # Query metadata
    query_type: QueryType | None = field(
        default=None,
        metadata=config(encoder=enum_encoder, decoder=_query_type_decoder),
    )
    difficulty: Difficulty | None = field(
        default=None,
        metadata=config(encoder=enum_encoder, decoder=_difficulty_decoder),
    )
    is_unanswerable: bool = False

    # Timing
    latency_ms: int | None = None

    trace_id: str | None = None  # link to QueryTrace if available

    # Compression metrics (populated when PromptCompressor ran)
    compression: dict[str, Any] | None = None

    @classmethod
    def from_results_dict(cls, data: dict[str, Any]) -> EvalResult:
        """
        Create from results.jsonl format which has nested 'retrieval' and 'answer' keys.

        This handles the legacy format where retrieval data is nested under 'retrieval'
        and answer requires the query string for construction.
        """
        # Import here to avoid circular dependency
        from rag.domain.models import Answer as AnswerCls

        qid = data["qid"]
        query = data.get("query", "")

        # Parse retrieval result from nested format
        retrieval_data = data.get("retrieval", {})
        retrieval = RetrievalResult.from_qid_dict(qid, retrieval_data)

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

        # Parse groundedness result if present
        groundedness_result = None
        groundedness_data = data.get("groundedness_result")
        if groundedness_data:
            groundedness_result = GroundednessJudgeResult.from_dict(groundedness_data)

        # Parse query type and difficulty
        query_type = parse_enum(QueryType, data.get("query_type"))
        difficulty = parse_enum(Difficulty, data.get("difficulty"))
        # Persisted results.jsonl rows include outcome_label as a string value.
        # Parse it back to OutcomeLabel so downstream verdict gating can use it.
        outcome_label = parse_enum(OutcomeLabel, data.get("outcome_label"))

        return cls(
            qid=qid,
            query=query,
            retrieval_result=retrieval,
            answer=answer,
            answer_metrics=answer_metrics,
            groundedness_result=groundedness_result,
            outcome_label=outcome_label,
            query_type=query_type,
            difficulty=difficulty,
            is_unanswerable=data.get("is_unanswerable", False),
            latency_ms=data.get("latency_ms"),
            trace_id=data.get("trace_id"),
            compression=data.get("compression"),
        )


@dataclass(frozen=True, slots=True)
class EvalRunMeta(DataClassJsonMixin):
    run_id: str = "unknown"
    started_at: datetime = field(
        default_factory=lambda: datetime.now(UTC),
        metadata=config(decoder=datetime_decoder, encoder=datetime_encoder),
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
    run_name: str | None = None
    notes: str | None = None
    gold_judge_version: str | None = None
    groundedness_judge_version: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    index_name: str | None = None
    index_id: str | None = None
    index_build_id: str | None = None


@dataclass(frozen=True, slots=True)
class EvalAggregates(DataClassJsonMixin):
    overall: RetrievalSummary
    by_type: dict[str, RetrievalSummary] = field(default_factory=dict)
    by_difficulty: dict[str, RetrievalSummary] = field(default_factory=dict)

    # Optional: add later
    answer_quality: dict[str, float] | None = None
    latency_ms: dict[str, float] | None = None
    compression: dict[str, float] | None = None

    @classmethod
    def from_flat_dict(cls, data: dict[str, Any]) -> EvalAggregates:
        """
        Create from flattened dict format (used in metrics.json files).

        Uses RetrievalSummary.from_flat_dict for nested summaries.
        """
        overall = RetrievalSummary.from_flat_dict(data.get("overall", {}))

        by_type: dict[str, RetrievalSummary] = {}
        for type_name, type_data in data.get("by_type", {}).items():
            by_type[type_name] = RetrievalSummary.from_flat_dict(type_data)

        by_difficulty: dict[str, RetrievalSummary] = {}
        for diff_name, diff_data in data.get("by_difficulty", {}).items():
            by_difficulty[diff_name] = RetrievalSummary.from_flat_dict(diff_data)

        return cls(
            overall=overall,
            by_type=by_type,
            by_difficulty=by_difficulty,
            answer_quality=data.get("answer_quality"),
            latency_ms=data.get("latency_ms"),
            compression=data.get("compression"),
        )


@dataclass(frozen=True, slots=True)
class EvalRun:
    meta: EvalRunMeta
    results: tuple[EvalResult, ...]
    aggregates: EvalAggregates
    # optional: where things were written
    artifacts: dict[str, str] | None = None  # e.g. {"results_jsonl": "...", "metrics_json": "..."}
