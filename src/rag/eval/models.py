from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.schema import Difficulty, QueryType

if TYPE_CHECKING:
    from rag.domain.models import Answer


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    qid: str
    retrieved_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: set[str]

    @classmethod
    def from_dict(cls, qid: str, data: dict[str, Any]) -> RetrievalResult:
        return cls(
            qid=qid,
            retrieved_chunk_ids=tuple(data.get("retrieved_chunk_ids", [])),
            relevant_chunk_ids=set(data.get("relevant_chunk_ids", [])),
        )

@dataclass(frozen=True, slots=True)
class RetrievalSummary:
    """Aggregate retrieval metrics over a set of queries."""
    num_queries: int
    avg_retrieved: float

    # Per-K metrics
    recall_at_k: dict[int, float]
    precision_at_k: dict[int, float]
    hit_rate_at_k: dict[int, float]
    ndcg_at_k: dict[int, float]

    # Global ranking metrics
    mrr: float
    map: float

    def to_dict(self) -> dict[str, float]:
        """
        Flatten to raw dict style (e.g., 'recall@10': 0.42) for compatibility.
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
    def from_dict(cls, data: dict[str, float]) -> RetrievalSummary:
        """
        Create from flattened dict style.
        """
        num_queries = int(data["num_queries"])
        avg_retrieved = float(data["avg_retrieved"])
        mrr = float(data["mrr"])
        map_score = float(data["map"])

        recall_at_k = {}
        precision_at_k = {}
        hit_rate_at_k = {}
        ndcg_at_k = {}

        for key, value in data.items():
            if key.startswith("recall@"):
                k = int(key.split("@")[1])
                recall_at_k[k] = float(value)
            elif key.startswith("precision@"):
                k = int(key.split("@")[1])
                precision_at_k[k] = float(value)
            elif key.startswith("hit_rate@"):
                k = int(key.split("@")[1])
                hit_rate_at_k[k] = float(value)
            elif key.startswith("ndcg@"):
                k = int(key.split("@")[1])
                ndcg_at_k[k] = float(value)

        return cls(
            num_queries=num_queries,
            avg_retrieved=avg_retrieved,
            recall_at_k=recall_at_k,
            precision_at_k=precision_at_k,
            hit_rate_at_k=hit_rate_at_k,
            ndcg_at_k=ndcg_at_k,
            mrr=mrr,
            map=map_score,
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
        retrieval = RetrievalResult.from_dict(qid, retrieval_data)

        # Parse answer if present
        answer = None
        answer_data = data.get("answer")
        if answer_data:
            answer = AnswerCls.from_dict(query, answer_data)

        # Parse answer metrics if present
        answer_metrics = None
        metrics_data = data.get("answer_metrics")
        if metrics_data:
            answer_metrics = AnswerQualityMetrics.from_dict(metrics_data)

        # Parse query type and difficulty
        query_type = None
        if data.get("query_type"):
            with contextlib.suppress(ValueError):
                query_type = QueryType(data["query_type"])

        difficulty = None
        if data.get("difficulty"):
            with contextlib.suppress(ValueError):
                difficulty = Difficulty(data["difficulty"])

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
class EvalRunMeta:
    run_id: str
    started_at: datetime
    queries_path: str | None
    top_k: int
    keep_k: int | None
    token_budget: int
    run_generation: bool
    use_llm_judge: bool
    judge_model: str | None
    generator_model: str | None
    embedder_model: str | None
    reranker_name: str | None
    notes: str | None = None
    gold_judge_version: str | None = None
    groundedness_judge_version: str | None = None
    extra: dict[str, Any] | None = None  # commit hash, dataset hash, etc.

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalRunMeta:
        started_at = datetime.now(UTC)
        if "started_at" in data:
            with contextlib.suppress(ValueError):
                started_at = datetime.fromisoformat(data["started_at"])

        return cls(
            run_id=data.get("run_id", "unknown"),
            started_at=started_at,
            queries_path=data.get("queries_path"),
            top_k=data.get("top_k", 10),
            keep_k=data.get("keep_k"),
            token_budget=data.get("token_budget", 1500),
            run_generation=data.get("run_generation", False),
            use_llm_judge=data.get("use_llm_judge", False),
            judge_model=data.get("judge_model"),
            generator_model=data.get("generator_model"),
            embedder_model=data.get("embedder_model"),
            reranker_name=data.get("reranker_name"),
            notes=data.get("notes"),
            gold_judge_version=data.get("gold_judge_version"),
            groundedness_judge_version=data.get("groundedness_judge_version"),
            extra=data.get("extra"),
        )


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
    
    
