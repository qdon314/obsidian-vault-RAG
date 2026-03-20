from datetime import UTC, datetime, timedelta

from eval.app_v2.engine.domain.enums import DiagnosticCode, Severity
from eval.app_v2.engine.domain.models import (
    RunBundle,
    RunConfig,
    RunHealthSummary,
)
from eval.app_v2.engine.services.trend import detect_config_change_events


def _make_config(**kwargs) -> RunConfig:
    defaults = {
        "retriever": "dense",
        "index_name": "idx",
        "reranker_model": None,
        "reranker_top_n": None,
        "generator_model": "gpt-4",
        "embedder_model": "ada-002",
        "top_k": 10,
        "token_budget": 4000,
    }
    defaults.update(kwargs)
    return RunConfig(**defaults)


def _make_health(recall: float = 0.8) -> RunHealthSummary:
    return RunHealthSummary(
        headline_recall_at_10=recall,
        headline_ndcg_at_10=recall,
        avg_quality_score=None,
        avg_latency_ms=None,
        severity_counts={Severity.OK: 10},
        diagnostic_counts={DiagnosticCode.GROUNDED_ANSWER: 10},
        dominant_failure_mode=None,
        dominant_failure_summary=None,
        worst_slice=None,
        verdict_status=None,
    )


def _make_bundle(
    run_id: str,
    ts: datetime,
    config: RunConfig | None = None,
    recall: float = 0.8,
) -> RunBundle:
    from rag.eval.models import EvalAggregates, RetrievalSummary

    return RunBundle(
        run_id=run_id,
        display_name=run_id,
        timestamp=ts,
        config=config or _make_config(),
        aggregates=EvalAggregates(overall=RetrievalSummary(num_queries=10, avg_retrieved=10.0)),
        queries=(),
        health=_make_health(recall),
        verdict=None,
        warnings=(),
        raw_artifacts={},
    )


_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def test_no_change_produces_empty_events():
    runs = [
        _make_bundle("r1", _T0),
        _make_bundle("r2", _T0 + timedelta(days=1)),
    ]
    assert detect_config_change_events(runs) == ()


def test_top_k_change_detected():
    runs = [
        _make_bundle("r1", _T0, _make_config(top_k=10)),
        _make_bundle("r2", _T0 + timedelta(days=1), _make_config(top_k=20)),
    ]
    events = detect_config_change_events(runs)
    assert len(events) == 1
    assert events[0].from_run_id == "r1"
    assert events[0].to_run_id == "r2"
    change_fields = {c.field_name for c in events[0].changes}
    assert "top_k" in change_fields


def test_single_run_produces_no_events():
    assert detect_config_change_events([_make_bundle("r1", _T0)]) == ()


def test_custom_tracked_fields_ignores_others():
    runs = [
        _make_bundle("r1", _T0, _make_config(top_k=10, generator_model="gpt-3")),
        _make_bundle(
            "r2", _T0 + timedelta(days=1), _make_config(top_k=20, generator_model="gpt-4")
        ),
    ]
    # Only track generator_model — top_k change should be ignored
    events = detect_config_change_events(runs, tracked_fields={"generator_model"})
    assert len(events) == 1
    change_fields = {c.field_name for c in events[0].changes}
    assert "generator_model" in change_fields
    assert "top_k" not in change_fields


def test_three_runs_two_changes():
    runs = [
        _make_bundle("r1", _T0, _make_config(top_k=10)),
        _make_bundle("r2", _T0 + timedelta(days=1), _make_config(top_k=20)),
        _make_bundle("r3", _T0 + timedelta(days=2), _make_config(top_k=30)),
    ]
    events = detect_config_change_events(runs)
    assert len(events) == 2
    assert events[0].from_run_id == "r1"
    assert events[1].from_run_id == "r2"


# ── build_trend_bundle tests ──────────────────────────────────────────────────

from eval.app_v2.engine.services.trend import build_trend_bundle  # noqa: E402


def test_build_trend_bundle_orders_by_timestamp():
    runs = [
        _make_bundle("r2", _T0 + timedelta(days=1), recall=0.9),
        _make_bundle("r1", _T0, recall=0.8),
    ]
    bundle = build_trend_bundle(runs)
    assert bundle.runs[0].run_id == "r1"
    assert bundle.runs[1].run_id == "r2"


def test_build_trend_bundle_metric_series():
    runs = [
        _make_bundle("r1", _T0, recall=0.8),
        _make_bundle("r2", _T0 + timedelta(days=1), recall=0.9),
    ]
    bundle = build_trend_bundle(runs)
    assert bundle.metric_series["recall@10"] == (0.8, 0.9)
    assert len(bundle.timestamps) == 2


def test_build_trend_bundle_includes_config_changes():
    runs = [
        _make_bundle("r1", _T0, config=_make_config(top_k=10)),
        _make_bundle("r2", _T0 + timedelta(days=1), config=_make_config(top_k=20)),
    ]
    bundle = build_trend_bundle(runs)
    assert len(bundle.config_change_events) == 1


def test_build_trend_bundle_verdict_series_none_when_absent():
    runs = [
        _make_bundle("r1", _T0),
        _make_bundle("r2", _T0 + timedelta(days=1)),
    ]
    bundle = build_trend_bundle(runs)
    assert all(v is None for v in bundle.verdict_series)


def test_build_trend_bundle_diagnostic_rate_series_sums_to_one():
    runs = [_make_bundle("r1", _T0)]
    bundle = build_trend_bundle(runs)
    from eval.app_v2.engine.domain.enums import DiagnosticCode

    total_rate = sum((bundle.diagnostic_rate_series[c][0] or 0.0) for c in DiagnosticCode)
    assert abs(total_rate - 1.0) < 1e-9
