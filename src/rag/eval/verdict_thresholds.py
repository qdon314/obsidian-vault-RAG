"""Config surface for eval verdict release-gating thresholds."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class VerdictThresholds:
    # Absolute minimums (block if below)
    min_recall_at_10: float = 0.60
    min_ndcg_at_10: float = 0.50
    min_mrr: float = 0.40
    max_avg_hallucination_severity: float = 2.5
    min_evidence_bounded_rate: float = 0.70
    max_latency_p95_ms: float = 5000.0

    # Behavioral rates (block if exceeded)
    max_unsafe_miss_rate: float = 0.10
    max_abstain_bad_rate: float = 0.10

    # Regression limits (block if delta exceeds)
    max_recall_regression: float = 0.05
    max_quality_regression: float = 0.10
    max_latency_regression_ms: float = 1000.0

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> VerdictThresholds:
        # Partial config is allowed; unspecified keys inherit class defaults.
        if not mapping:
            return cls()
        defaults = cls()
        return cls(
            min_recall_at_10=float(mapping.get("min_recall_at_10", defaults.min_recall_at_10)),
            min_ndcg_at_10=float(mapping.get("min_ndcg_at_10", defaults.min_ndcg_at_10)),
            min_mrr=float(mapping.get("min_mrr", defaults.min_mrr)),
            max_avg_hallucination_severity=float(
                mapping.get(
                    "max_avg_hallucination_severity", defaults.max_avg_hallucination_severity
                )
            ),
            min_evidence_bounded_rate=float(
                mapping.get("min_evidence_bounded_rate", defaults.min_evidence_bounded_rate)
            ),
            max_latency_p95_ms=float(
                mapping.get("max_latency_p95_ms", defaults.max_latency_p95_ms)
            ),
            max_unsafe_miss_rate=float(
                mapping.get("max_unsafe_miss_rate", defaults.max_unsafe_miss_rate)
            ),
            max_abstain_bad_rate=float(
                mapping.get("max_abstain_bad_rate", defaults.max_abstain_bad_rate)
            ),
            max_recall_regression=float(
                mapping.get("max_recall_regression", defaults.max_recall_regression)
            ),
            max_quality_regression=float(
                mapping.get("max_quality_regression", defaults.max_quality_regression)
            ),
            max_latency_regression_ms=float(
                mapping.get("max_latency_regression_ms", defaults.max_latency_regression_ms)
            ),
        )


def load_verdict_thresholds(path: str | Path = "settings.toml") -> VerdictThresholds:
    config_path = Path(path)
    # Missing config falls back to conservative in-code defaults.
    if not config_path.exists():
        return VerdictThresholds()

    with config_path.open("rb") as f:
        raw = tomllib.load(f)

    eval_section = raw.get("eval", {})
    if not isinstance(eval_section, dict):
        return VerdictThresholds()
    verdict_section = eval_section.get("verdict", {})
    # Non-table values are treated as absent to avoid runtime failures in CI.
    if not isinstance(verdict_section, dict):
        return VerdictThresholds()

    return VerdictThresholds.from_mapping(verdict_section)
