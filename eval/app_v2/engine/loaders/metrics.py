# eval/app_v2/engine/loaders/metrics.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import LoadedArtifact
from rag.eval.models import EvalAggregates, EvalRunMeta

logger = logging.getLogger(__name__)


class MetricsLoader:
    artifact_name = "metrics.json"

    def can_load(self, run_dir: Path) -> bool:
        return (run_dir / "metrics.json").exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        try:
            data = json.loads((run_dir / "metrics.json").read_text())
            meta = EvalRunMeta.from_dict(data.get("meta", {}))  # type: ignore[attr-defined]
            aggregates = EvalAggregates.from_flat_dict(data)
            return LoadedArtifact(
                artifact_name=self.artifact_name,
                payload=(aggregates, meta),
                warnings=tuple(warnings),
            )
        except Exception as exc:
            warnings.append(
                BundleWarning(
                    code=BundleWarningCode.PARTIAL_RESULTS_PARSE,
                    message=f"Failed to parse metrics.json: {exc}",
                    artifact_name=self.artifact_name,
                )
            )
            return LoadedArtifact(
                artifact_name=self.artifact_name, payload=None, warnings=tuple(warnings)
            )
