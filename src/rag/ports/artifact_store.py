"""Port for reading/writing index artifacts to a storage backend."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ArtifactStore(Protocol):
    """Abstracts artifact storage (local filesystem, S3, etc.).

    The contract: callers get a local directory path to work with.
    Implementations handle syncing to/from the backing store.
    """

    def pull(self, remote_key: str, local_dir: Path) -> Path:
        """Download artifacts from remote storage to a local directory.

        Returns the local directory path (same as local_dir).
        """
        ...

    def push(self, local_dir: Path, remote_key: str) -> None:
        """Upload artifacts from a local directory to remote storage."""
        ...
