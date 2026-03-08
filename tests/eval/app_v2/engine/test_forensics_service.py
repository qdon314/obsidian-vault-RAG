from pathlib import Path

import pytest

REAL_RUN = Path("eval/runs/run_2026_02_12T20-40")


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_get_query_returns_analyzed_query():
    from eval.app_v2.engine.domain.enums import DiagnosticCode
    from eval.app_v2.engine.loaders.bundle import build_bundle
    from eval.app_v2.engine.services.forensics import get_query, list_queries_by_code

    bundle = build_bundle(REAL_RUN)
    first_qid = bundle.queries[0].record.qid

    aq = get_query(bundle, first_qid)
    assert aq is not None
    assert aq.record.qid == first_qid

    misses = list_queries_by_code(bundle, DiagnosticCode.RETRIEVAL_MISS)
    # just verify it returns a tuple without error
    assert isinstance(misses, tuple)
