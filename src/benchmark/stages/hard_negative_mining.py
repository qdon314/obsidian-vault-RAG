"""Hard negative mining via the RAG retriever.

This is the only place the benchmark package crosses the RAG boundary.
See ``docs/decisions/adr-benchmark-rag-boundary-crossing.md`` for rationale.

Hard negatives are top-k retrieval results that are NOT in the query's
evidence set.  They are required for valid reranker evaluation — without
them, the eval measures easy separation (evidence vs. random) rather than
real retrieval discrimination.
"""

from __future__ import annotations

import logging
from typing import Any

from benchmark.domain.models import (
    EvidenceSet,
    HardNegativeResult,
    ValidatedQuery,
)
from rag.domain.models import Candidate
from rag.ports.retriever import Retriever

logger = logging.getLogger(__name__)


def mine_hard_negatives(
    queries: list[ValidatedQuery],
    evidence_sets: dict[str, EvidenceSet],  # keyed by unit_id
    retriever: Retriever,
    *,
    top_k: int = 20,
    min_hard_negatives: int = 2,
    retriever_config: dict[str, Any],
) -> list[HardNegativeResult]:
    """Run the retriever against each query; collect hard negatives.

    Hard negatives are top-k results whose chunk_ids are **not** present
    in the query's evidence set (any tier).  ``retriever_config`` is recorded
    on every result for staleness detection when the index changes.

    Args:
        queries: Validated query candidates to mine for.
        evidence_sets: Per-unit evidence, keyed by unit_id.
        retriever: Live RAG retriever to call.
        top_k: Number of candidates to retrieve per query.
        min_hard_negatives: Flag insufficient=True below this count.
        retriever_config: Metadata recorded on each result (model, index_version, top_k).

    Returns:
        One ``HardNegativeResult`` per input query, in the same order.
    """
    results: list[HardNegativeResult] = []

    for query in queries:
        evidence = evidence_sets.get(query.unit_id)
        evidence_chunk_ids = _collect_evidence_chunk_ids(evidence)

        try:
            candidates: list[Candidate] = retriever.retrieve(query.query, top_k=top_k)
        except Exception:
            logger.exception(
                "Retriever failed for candidate %s — using empty hard negatives",
                query.candidate_id,
            )
            candidates = []

        hard_negative_ids = tuple(
            c.chunk.chunk_id for c in candidates if c.chunk.chunk_id not in evidence_chunk_ids
        )

        insufficient = len(hard_negative_ids) < min_hard_negatives

        if insufficient:
            logger.info(
                "Candidate %s: only %d hard negatives found (min=%d)",
                query.candidate_id,
                len(hard_negative_ids),
                min_hard_negatives,
            )

        results.append(
            HardNegativeResult(
                candidate_id=query.candidate_id,
                hard_negative_chunk_ids=hard_negative_ids,
                retriever_config=retriever_config,
                insufficient=insufficient,
            )
        )

    return results


def _collect_evidence_chunk_ids(evidence: EvidenceSet | None) -> frozenset[str]:
    """Flatten all chunk_ids from all evidence tiers into a frozen set."""
    if evidence is None:
        return frozenset()
    ids: list[str] = []
    for entry in (*evidence.critical, *evidence.supporting, *evidence.contextual):
        ids.extend(entry.chunk_ids)
    return frozenset(ids)
