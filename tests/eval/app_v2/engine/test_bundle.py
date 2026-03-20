# tests/eval/app_v2/engine/test_bundle.py
from pathlib import Path

import pytest

REAL_RUN = Path("eval/runs/run_2026_02_12T20-40")


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_build_bundle_produces_run_bundle():
    from eval.app_v2.engine.domain.models import RunBundle
    from eval.app_v2.engine.loaders.bundle import build_bundle

    bundle = build_bundle(REAL_RUN)
    assert isinstance(bundle, RunBundle)
    assert bundle.run_id
    assert len(bundle.queries) > 0
    assert bundle.health.headline_recall_at_10 >= 0.0


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_build_bundle_queries_have_diagnostics():
    from eval.app_v2.engine.domain.models import AnalyzedQuery
    from eval.app_v2.engine.loaders.bundle import build_bundle

    bundle = build_bundle(REAL_RUN)
    assert all(isinstance(q, AnalyzedQuery) for q in bundle.queries)
    assert all(q.diagnostic is not None for q in bundle.queries)
