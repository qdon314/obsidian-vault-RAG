"""Index build manifest for artifact provenance tracking."""

from __future__ import annotations

import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _get_git_sha() -> str:
    """Get current git HEAD SHA, or 'unknown' if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _is_git_dirty() -> bool:
    """Check if working tree has uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode != 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@dataclass(frozen=True, slots=True)
class IndexManifest:
    """Self-describing manifest for an index build.

    Every index directory should contain a manifest.json written
    from this object, enabling provenance tracking and reproducibility.
    """

    index_name: str
    created_at: str
    git_sha: str
    corpus: str
    doc_count: int
    chunk_count: int
    chunking: dict[str, Any] = field(default_factory=dict)
    embedding: dict[str, Any] = field(default_factory=dict)
    ingest_report: dict[str, Any] = field(default_factory=dict)
    store: dict[str, Any] = field(default_factory=dict)
    build_id: str = ""
    status: str = "complete"
    git_dirty: bool = False
    build_duration_s: float = 0.0

    @staticmethod
    def create(
        *,
        index_name: str,
        corpus: str,
        doc_count: int,
        chunk_count: int,
        chunking: dict[str, Any] | None = None,
        embedding: dict[str, Any] | None = None,
        ingest_report: dict[str, Any] | None = None,
        store: dict[str, Any] | None = None,
        status: str = "complete",
        build_duration_s: float = 0.0,
    ) -> IndexManifest:
        """Create a manifest with auto-populated timestamp, git SHA, build_id."""
        return IndexManifest(
            index_name=index_name,
            created_at=datetime.now(UTC).isoformat(),
            git_sha=_get_git_sha(),
            corpus=corpus,
            doc_count=doc_count,
            chunk_count=chunk_count,
            chunking=chunking or {},
            embedding=embedding or {},
            ingest_report=ingest_report or {},
            store=store or {},
            build_id=uuid.uuid4().hex,
            status=status,
            git_dirty=_is_git_dirty(),
            build_duration_s=build_duration_s,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> IndexManifest:
        return IndexManifest(
            **{k: v for k, v in data.items() if k in IndexManifest.__dataclass_fields__}
        )

    def save(self, directory: Path) -> Path:
        """Write manifest.json to directory. Returns the file path."""
        path = directory / "manifest.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @staticmethod
    def load(directory: Path) -> IndexManifest:
        """Load manifest.json from directory."""
        path = directory / "manifest.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return IndexManifest.from_dict(data)
