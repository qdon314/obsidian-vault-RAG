"""Local filesystem artifact store (passthrough, no remote sync)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LocalArtifactStore:
    """No-op artifact store for local development.

    pull() returns the local directory as-is.
    push() is a no-op (files are already local).
    """

    def pull(self, remote_key: str, local_dir: Path) -> Path:
        return local_dir

    def push(self, local_dir: Path, remote_key: str) -> None:
        pass
