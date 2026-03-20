# eval/app_v2/ui/pages/trends.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

from eval.app_v2.engine.domain.enums import DiagnosticCode
from eval.app_v2.engine.domain.models import TrendBundle
from eval.app_v2.engine.services.trend import build_trend_bundle
from eval.app_v2.ui.app import load_bundle


def render(runs: list[tuple[str, Path]]) -> None:
    st.header("Trends")

    if not runs:
        st.info("No runs found in eval/runs/.")
        return

    names = [name for name, _ in runs]
    default_selection = names[: min(5, len(names))]
    selected_names = st.multiselect("Select runs to compare", names, default=default_selection)

    if len(selected_names) < 2:
        st.info("Select at least 2 runs to view trends.")
        return

    selected_pairs = [(n, p) for n, p in runs if n in selected_names]
    bundles = [load_bundle(name, str(path)) for name, path in selected_pairs]
    trend = build_trend_bundle(bundles)

    # ── Metric trends ──────────────────────────────────────────────────────────
    st.subheader("Metric trends")
    run_labels = [r.display_name for r in trend.runs]
    for metric_name in ("recall@10", "ndcg@10"):
        series = trend.metric_series[metric_name]
        if any(v is not None for v in series):
            import pandas as pd

            chart_data = {
                "Run": run_labels,
                metric_name: [v if v is not None else 0.0 for v in series],
            }
            df = pd.DataFrame(chart_data).set_index("Run")
            st.line_chart(df, x_label="Run", y_label=metric_name)

    # ── Config change events ───────────────────────────────────────────────────
    if trend.config_change_events:
        st.subheader("Configuration changes")
        for evt in trend.config_change_events:
            label = f"{evt.from_run_id} → {evt.to_run_id} ({evt.timestamp.strftime('%Y-%m-%d')})"
            with st.expander(label):
                for c in evt.changes:
                    st.markdown(f"- **{c.field_name}**: `{c.before}` → `{c.after}`")

    # ── Diagnostic failure-mode rates ─────────────────────────────────────────
    st.subheader("Failure mode rates over time")
    failure_codes = [
        c
        for c in DiagnosticCode
        if c not in (DiagnosticCode.GROUNDED_ANSWER, DiagnosticCode.NO_CLEAR_FAILURE)
    ]
    active_codes = [
        c
        for c in failure_codes
        if any((v or 0.0) > 0.0 for v in trend.diagnostic_rate_series.get(c, ()))
    ]
    if active_codes:
        import pandas as pd

        rate_data: dict[str, list[str] | list[float]] = {"Run": run_labels}
        for code in active_codes:
            rate_data[code.value] = [
                v if v is not None else 0.0 for v in trend.diagnostic_rate_series[code]
            ]
        df_rates = pd.DataFrame(rate_data).set_index("Run")
        st.line_chart(df_rates)
    else:
        st.caption("No active failure modes across selected runs.")

    # ── Verdict timeline ───────────────────────────────────────────────────────
    st.subheader("Verdict timeline")
    for run, verdict in zip(trend.runs, trend.verdict_series, strict=False):
        icon = "✅" if verdict == "SHIP" else ("🚫" if verdict == "BLOCK" else "—")
        st.markdown(f"- **{run.display_name}**: {icon} {verdict or 'n/a'}")

    # ── Run summary table ──────────────────────────────────────────────────────
    st.subheader("Run summary")
    import pandas as pd

    rows = [
        {
            "Run": run.display_name,
            "recall@10": f"{run.health.headline_recall_at_10:.3f}",
            "ndcg@10": f"{run.health.headline_ndcg_at_10:.3f}",
            "Avg latency": f"{run.health.avg_latency_ms:.0f} ms"
            if run.health.avg_latency_ms
            else "—",
            "Verdict": run.health.verdict_status or "—",
            "Config changes": sum(
                1 for e in trend.config_change_events if e.to_run_id == run.run_id
            ),
        }
        for run in trend.runs
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    _render_correlation(trend)


def _render_correlation(trend: TrendBundle) -> None:
    """Scatter chart: latency vs recall@10, coloured by severity, across all selected runs."""
    import pandas as pd

    rows = []
    for run in trend.runs:
        for aq in run.queries:
            r = aq.record
            if r.latency_ms is not None:
                rows.append(
                    {
                        "latency_ms": r.latency_ms,
                        "recall@10": r.per_query_recall_at_k.get(10, 0.0),
                        "severity": aq.diagnostic.severity.value,
                        "run": run.display_name,
                    }
                )

    st.subheader("Latency vs recall@10 (per query)")
    if not rows:
        st.caption("No latency data available.")
        return

    df = pd.DataFrame(rows)
    st.scatter_chart(df, x="latency_ms", y="recall@10", color="severity", use_container_width=True)
