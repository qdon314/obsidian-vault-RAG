# eval/app_v2/engine/loaders/base.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from eval.app_v2.engine.domain.warnings import BundleWarning


@dataclass(frozen=True, slots=True)
class LoadedArtifact:
    artifact_name: str
    payload: Any
    warnings: tuple[BundleWarning, ...]


@runtime_checkable
class ArtifactLoader(Protocol):
    artifact_name: str

    def can_load(self, run_dir: Path) -> bool: ...
    def load(self, run_dir: Path) -> LoadedArtifact: ...
