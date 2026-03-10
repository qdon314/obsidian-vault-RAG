# tests/eval/app_v2/engine/test_facets.py
from eval.app_v2.engine.domain.models import QueryRecord
from eval.app_v2.engine.facets.registry import FACETS, FacetDef


def _record(query_type="factual", difficulty="easy", is_unanswerable=False):
    return QueryRecord(
        qid="q1", query="test", query_type=query_type, difficulty=difficulty,
        is_unanswerable=is_unanswerable, requires_synthesis=False, tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",),
        reranked_chunk_ids=None, packed_chunk_ids=None,
        per_query_recall_at_k={10: 1.0},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: 1.0},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None, answer_metrics=None, groundedness=None,
        latency_ms=None, trace_id=None, trace=None,
    )


def test_facets_list_is_nonempty():
    assert len(FACETS) >= 4


def test_facet_def_has_required_fields():
    for f in FACETS:
        assert isinstance(f, FacetDef)
        assert f.key
        assert f.label
        assert f.value_type in ("enum", "bool", "numeric_bucket")
        assert callable(f.extract)


def test_query_type_facet_extracts_correctly():
    facet = next(f for f in FACETS if f.key == "query_type")
    r = _record(query_type="factual")
    assert facet.extract(r) == "factual"


def test_bool_facet_extracts_correctly():
    facet = next(f for f in FACETS if f.key == "is_unanswerable")
    r = _record(is_unanswerable=True)
    assert facet.extract(r) is True
