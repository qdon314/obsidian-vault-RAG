# eval/app_v2/engine/loaders/verdict.py
from __future__ import annotations

import json
import logging
from pathlib import Path

from eval.app_v2.engine.domain.models import VerdictSummary
from eval.app_v2.engine.domain.warnings import BundleWarning, BundleWarningCode
from eval.app_v2.engine.loaders.base import LoadedArtifact
from rag.eval.verdict import verdict_from_dict

logger = logging.getLogger(__name__)

_DEFAULT_VERDICT_PATH = Path("eval/verdicts/verdict.json")


class VerdictLoader:
    artifact_name = "verdict.json"

    def __init__(self, verdict_path: Path = _DEFAULT_VERDICT_PATH) -> None:
        self._verdict_path = verdict_path

    def can_load(self, run_dir: Path) -> bool:
        # Verdict is run-agnostic; check the conventional path
        return self._verdict_path.exists()

    def load(self, run_dir: Path) -> LoadedArtifact:
        warnings: list[BundleWarning] = []
        if not self._verdict_path.exists():
            warnings.append(BundleWarning(
                code=BundleWarningCode.MISSING_VERDICT,
                message=f"No verdict file at {self._verdict_path}",
            ))
            return LoadedArtifact(artifact_name=self.artifact_name, payload=None, warnings=tuple(warnings))
        try:
            data = json.loads(self._verdict_path.read_text())
            verdict = verdict_from_dict(data)
            decision_str = verdict.decision.value.upper()  # "SHIP" or "BLOCK"
            failed = tuple(c.name for c in verdict.checks if not c.passed)
            summary = VerdictSummary(
                decision=decision_str,  # type: ignore[arg-type]
                failed_check_names=failed,
                raw=verdict,
            )
            return LoadedArtifact(artifact_name=self.artifact_name, payload=summary, warnings=())
        except Exception as exc:
            warnings.append(BundleWarning(
                code=BundleWarningCode.MISSING_VERDICT,
                message=f"Verdict parse error: {exc}",
            ))
            return LoadedArtifact(artifact_name=self.artifact_name, payload=None, warnings=tuple(warnings))
