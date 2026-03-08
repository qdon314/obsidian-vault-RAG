# eval/app_v2/engine/loaders/results.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import LoadedArtifact
from rag.eval.models import EvalResult

logger = logging.getLogger(__name__)


class ResultsLoader:
    artifact_name = "results.jsonl"

    def can_load(self, run_dir: Path) -> bool:
        return (run_dir / "results.jsonl").exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        results: list[EvalResult] = []
        path = run_dir / "results.jsonl"
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(EvalResult.from_results_dict(json.loads(line)))
            except Exception as exc:
                warnings.append(BundleWarning(
                    code=BundleWarningCode.PARTIAL_RESULTS_PARSE,
                    message=f"Row {i} parse error: {exc}",
                    artifact_name=self.artifact_name,
                ))
        return LoadedArtifact(
            artifact_name=self.artifact_name,
            payload=tuple(results),
            warnings=tuple(warnings),
        )
