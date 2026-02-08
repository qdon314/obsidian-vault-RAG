"""Pure logic for query change diagnostics (spec 06).

No Streamlit imports -- all functions are testable standalone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.eval.models import EvalResult

_NATURAL_SORT_RE = re.compile(r"(\d+)")


def natural_sort_key(qid: str) -> tuple:
    """Sort key that orders numeric segments numerically.

    "q_2" < "q_10" < "q_100" instead of lexicographic "q_10" < "q_100" < "q_2".
    """
    parts = _NATURAL_SORT_RE.split(qid)
    return tuple(int(p) if p.isdigit() else p.lower() for p in parts)


@dataclass(frozen=True, slots=True)
class ChunkDiffRow:
    """One row of the retrieval diff table."""

    chunk_id: str
    relevant: bool
    rank_a: int | None  # 1-indexed, None if absent from run
    rank_b: int | None
    status: str  # "TP lost", "TP gained", "FP lost", "FP gained",
    #              "Moved up", "Moved down", "Unchanged"


# Sort priority: lower number = higher in table
_STATUS_SORT_ORDER = {
    "TP lost": 0,
    "TP gained": 1,
    "Moved up": 2,
    "Moved down": 3,
    "Unchanged": 4,
    "FP gained": 5,
    "FP lost": 6,
}


def compute_retrieval_diff(
    result_a: EvalResult,
    result_b: EvalResult,
    k: int = 10,
) -> list[ChunkDiffRow]:
    """Compute a unified diff of retrieved chunks between two runs.

    Args:
        result_a: Result from run A (must have same qid as result_b).
        result_b: Result from run B (must have same qid as result_a).
        k: Number of top results to compare (default 10, matching recall@10).

    Returns:
        Rows sorted by diagnostic priority: TP lost first, then TP gained,
        then rank movers, then the rest.

    Raises:
        ValueError: If result_a.qid != result_b.qid.
    """
    if result_a.qid != result_b.qid:
        raise ValueError(
            f"Cannot diff results for different queries: "
            f"{result_a.qid} vs {result_b.qid}"
        )

    ids_a = result_a.retrieval_result.retrieved_chunk_ids[:k]
    ids_b = result_b.retrieval_result.retrieved_chunk_ids[:k]

    rank_a = {cid: i + 1 for i, cid in enumerate(ids_a)}
    rank_b = {cid: i + 1 for i, cid in enumerate(ids_b)}

    # Union of relevant chunks from both results (should be identical, but be safe)
    relevant = (
        result_a.retrieval_result.relevant_chunk_ids
        | result_b.retrieval_result.relevant_chunk_ids
    )

    all_chunk_ids = list(
        dict.fromkeys(list(ids_a) + list(ids_b))
    )  # preserve order, deduplicate

    rows: list[ChunkDiffRow] = []
    for cid in all_chunk_ids:
        ra = rank_a.get(cid)
        rb = rank_b.get(cid)
        is_relevant = cid in relevant

        if ra is not None and rb is None:
            status = "TP lost" if is_relevant else "FP lost"
        elif ra is None and rb is not None:
            status = "TP gained" if is_relevant else "FP gained"
        elif ra is not None and rb is not None:
            if ra == rb:
                status = "Unchanged"
            elif rb < ra:
                status = "Moved up"
            else:
                status = "Moved down"
        else:
            continue  # shouldn't happen

        rows.append(
            ChunkDiffRow(
                chunk_id=cid,
                relevant=is_relevant,
                rank_a=ra,
                rank_b=rb,
                status=status,
            )
        )

    rows.sort(key=lambda r: _STATUS_SORT_ORDER.get(r.status, 99))
    return rows
