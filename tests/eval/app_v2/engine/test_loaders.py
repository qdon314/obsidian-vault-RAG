# tests/eval/app_v2/engine/test_loaders.py
from pathlib import Path

import pytest

REAL_RUN = Path("eval/runs/run_2026_02_12T20-40")


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_metrics_loader_loads_aggregates():
    from eval.app_v2.engine.loaders.metrics import MetricsLoader
    from rag.eval.models import EvalAggregates
    loader = MetricsLoader()
    assert loader.can_load(REAL_RUN)
    artifact = loader.load(REAL_RUN)
    assert artifact.payload is not None
    agg, meta = artifact.payload
    assert isinstance(agg, EvalAggregates)
    assert meta.top_k >= 1


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_results_loader_loads_eval_results():
    from eval.app_v2.engine.loaders.results import ResultsLoader
    from rag.eval.models import EvalResult
    loader = ResultsLoader()
    assert loader.can_load(REAL_RUN)
    artifact = loader.load(REAL_RUN)
    results = artifact.payload
    assert isinstance(results, tuple)
    assert len(results) > 0
    assert isinstance(results[0], EvalResult)


@pytest.mark.skipif(not REAL_RUN.exists(), reason="no real run dir")
def test_traces_loader_absent_dir():
    from eval.app_v2.engine.loaders.traces import TracesLoader
    loader = TracesLoader()
    no_traces_dir = Path("eval/runs/run_2026_02_20T19-49")  # known to have no traces
    # can_load returns False when file absent
    if not (no_traces_dir / "traces.jsonl").exists():
        assert not loader.can_load(no_traces_dir)


@pytest.mark.skipif(not (REAL_RUN / "traces.jsonl").exists(), reason="no traces")
def test_traces_loader_loads_dict():
    from eval.app_v2.engine.loaders.traces import TracesLoader
    loader = TracesLoader()
    artifact = loader.load(REAL_RUN)
    traces = artifact.payload  # dict[trace_id, QueryTrace]
    assert isinstance(traces, dict)
