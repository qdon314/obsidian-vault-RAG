from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from rag.domain.models import Answer
from rag.eval.answer_metrics import AnswerQualityMetrics
from rag.eval.schema import Difficulty, QueryType


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    qid: str
    retrieved_chunk_ids: tuple[str, ...]
    relevant_chunk_ids: set[str]

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


@dataclass(frozen=True, slots=True)
class EvalAggregates:
    overall: RetrievalSummary
    by_type: dict[str, RetrievalSummary]
    by_difficulty: dict[str, RetrievalSummary]

    # Optional: add later
    answer_quality: dict[str, float] | None = None
    latency_ms: dict[str, float] | None = None

@dataclass(frozen=True, slots=True)
class EvalRun:
    meta: EvalRunMeta
    results: tuple[EvalResult, ...]
    aggregates: EvalAggregates
    # optional: where things were written
    artifacts: dict[str, str] | None = None  # e.g. {"results_jsonl": "...", "metrics_json": "..."}
    
    
