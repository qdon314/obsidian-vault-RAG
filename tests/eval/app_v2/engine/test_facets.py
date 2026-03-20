# tests/eval/app_v2/engine/test_facets.py
from eval.app_v2.engine.domain.models import QueryRecord
from eval.app_v2.engine.facets.registry import FACETS, FacetDef, get_facet
from rag.eval.answer_metrics import AnswerQualityMetrics


def _record(
    query_type="factual",
    difficulty="easy",
    is_unanswerable=False,
    recall_10=0.8,
    ndcg_10=None,
    quality_score=None,
    hallucination_severity=None,
    latency_ms=None,
):
    am: AnswerQualityMetrics | None = None
    if quality_score is not None or hallucination_severity is not None:
        am = AnswerQualityMetrics(
            quality_score=quality_score,
            hallucination_severity=hallucination_severity,
        )
    return QueryRecord(
        qid="q1",
        query="test",
        query_type=query_type,
        difficulty=difficulty,
        is_unanswerable=is_unanswerable,
        requires_synthesis=False,
        tags=(),
        relevant_chunk_ids=frozenset(["c1"]),
        retrieved_chunk_ids=("c1",),
        reranked_chunk_ids=None,
        packed_chunk_ids=None,
        per_query_recall_at_k={10: recall_10},
        per_query_precision_at_k={10: 1.0},
        per_query_ndcg_at_k={10: ndcg_10 if ndcg_10 is not None else recall_10},
        per_query_hit_rate_at_k={10: 1.0},
        answer_text=None,
        answer_metrics=am,
        groundedness=None,
        latency_ms=latency_ms,
        trace_id=None,
        trace=None,
    )


def test_facets_list_is_nonempty():
    assert len(FACETS) >= 4


def test_facet_def_has_required_fields():
    for f in FACETS:
        assert isinstance(f, FacetDef)
        assert f.key
        assert f.label
        assert f.value_type in ("enum", "bool", "numeric_range")
        assert callable(f.extract)


def test_query_type_facet_extracts_correctly():
    facet = next(f for f in FACETS if f.key == "query_type")
    r = _record(query_type="factual")
    assert facet.extract(r) == "factual"


def test_bool_facet_extracts_correctly():
    facet = next(f for f in FACETS if f.key == "is_unanswerable")
    r = _record(is_unanswerable=True)
    assert facet.extract(r) is True


# ── New: numeric facet extraction tests ──────────────────────────────────────


def test_recall_at_10_facet_extracts():
    facet = get_facet("recall_at_10")
    assert facet is not None
    assert facet.value_type == "numeric_range"
    r = _record(recall_10=0.75)
    assert facet.extract(r) == 0.75


def test_ndcg_at_10_facet_extracts():
    facet = get_facet("ndcg_at_10")
    assert facet is not None
    # Use a distinct ndcg value so we're not accidentally passing because ndcg == recall
    r = _record(recall_10=0.6, ndcg_10=0.42)
    assert facet.extract(r) == 0.42


def test_quality_score_facet_extracts():
    facet = get_facet("quality_score")
    assert facet is not None
    r = _record(quality_score=0.82)
    assert facet.extract(r) == 0.82


def test_quality_score_facet_returns_none_when_no_answer_metrics():
    facet = get_facet("quality_score")
    assert facet is not None
    r = _record()  # no answer_metrics
    assert facet.extract(r) is None


def test_hallucination_severity_facet_extracts():
    facet = get_facet("hallucination_severity")
    assert facet is not None
    r = _record(hallucination_severity=2.5)
    assert facet.extract(r) == 2.5


def test_hallucination_severity_facet_returns_none_when_no_answer_metrics():
    facet = get_facet("hallucination_severity")
    assert facet is not None
    r = _record()
    assert facet.extract(r) is None


def test_latency_ms_facet_extracts():
    facet = get_facet("latency_ms")
    assert facet is not None
    r = _record(latency_ms=350)
    assert facet.extract(r) == 350.0


def test_latency_ms_facet_returns_none_when_missing():
    facet = get_facet("latency_ms")
    assert facet is not None
    r = _record()  # latency_ms=None
    assert facet.extract(r) is None


def test_diagnostic_code_facet_exists():
    facet = get_facet("diagnostic_code")
    assert facet is not None
    assert facet.value_type == "enum"


def test_numeric_facets_higher_is_better_flags():
    recall_f = get_facet("recall_at_10")
    halluc_f = get_facet("hallucination_severity")
    latency_f = get_facet("latency_ms")
    assert recall_f is not None and recall_f.higher_is_better is True
    assert halluc_f is not None and halluc_f.higher_is_better is False
    assert latency_f is not None and latency_f.higher_is_better is False


def test_recall_at_10_facet_returns_none_when_key_absent():
    facet = get_facet("recall_at_10")
    assert facet is not None
    # Build a record with an empty per_query_recall_at_k dict
    import dataclasses

    r = _record(recall_10=0.5)
    r_no_k = dataclasses.replace(r, per_query_recall_at_k={})
    assert facet.extract(r_no_k) is None


def test_ndcg_at_10_facet_returns_none_when_key_absent():
    facet = get_facet("ndcg_at_10")
    assert facet is not None
    import dataclasses

    r = _record(recall_10=0.5, ndcg_10=0.5)
    r_no_k = dataclasses.replace(r, per_query_ndcg_at_k={})
    assert facet.extract(r_no_k) is None
