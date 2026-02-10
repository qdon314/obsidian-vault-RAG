from __future__ import annotations

from rag.eval.harness import _build_citation_chunk_indexes, _resolve_relevant_chunk_ids
from rag.eval.schema import EvalQuery
from tests.conftest import make_chunk


def test_resolve_relevant_chunk_ids_from_citations() -> None:
    chunks = [
        make_chunk(
            chunk_id="chunk-a",
            metadata={
                "citation": "10 CFR §50.34(a)",
                "citation_key": "10 CFR §50.34",
            },
        ),
        make_chunk(
            chunk_id="chunk-b",
            metadata={
                "citation": "10 CFR §50.36(a)",
                "citation_key": "10 CFR §50.36",
            },
        ),
    ]
    citation_to_ids, citation_key_to_ids = _build_citation_chunk_indexes(chunks)

    query = EvalQuery(
        qid="q1",
        query="test",
        relevant_chunk_ids=set(),
        relevant_citations={"10 CFR §50.34(a)"},
        relevant_doc_citations={"10 CFR §50.36"},
    )

    resolved = _resolve_relevant_chunk_ids(
        query,
        citation_to_ids=citation_to_ids,
        citation_key_to_ids=citation_key_to_ids,
    )

    assert resolved == {"chunk-a", "chunk-b"}


def test_resolve_relevant_chunk_ids_keeps_explicit_chunk_ids() -> None:
    query = EvalQuery(
        qid="q2",
        query="test",
        relevant_chunk_ids={"legacy-chunk-id"},
        relevant_citations=set(),
        relevant_doc_citations=set(),
    )

    resolved = _resolve_relevant_chunk_ids(
        query,
        citation_to_ids={},
        citation_key_to_ids={},
    )

    assert resolved == {"legacy-chunk-id"}
