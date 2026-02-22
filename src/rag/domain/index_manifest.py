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
    index_id: str
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
        index_id = f"{index_name}_{getattr(chunking, 'backend', 'unknown')}_{getattr(embedding, 'backend', 'unknown')}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        return IndexManifest(
            index_name=index_name,
            index_id=index_id,
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

    @staticmethod
    def load_uri(uri: str) -> IndexManifest:
        """Load a manifest from a local path or ``s3://bucket/key`` URI."""
        if uri.startswith("s3://"):
            import boto3  # type: ignore[import-untyped]

            parts = uri[5:].split("/", 1)
            bucket, key = parts[0], parts[1]
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            data = json.loads(obj["Body"].read().decode("utf-8"))
            return IndexManifest.from_dict(data)

        path = Path(uri)
        if path.is_dir():
            return IndexManifest.load(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        return IndexManifest.from_dict(data)

    @staticmethod
    def latest_from_s3(
        bucket: str,
        corpus_prefix: str,
        manifests_subprefix: str = "manifests",
    ) -> IndexManifest | None:
        """Return the most recent manifest from S3, or None if unavailable.

        Searches ``s3://{bucket}/{corpus_prefix}/{manifests_subprefix}/``
        for index-id subdirectories.  Lexicographic sort on the directory
        name gives chronological order because ``index_id`` embeds an
        ISO-8601 timestamp suffix (``%Y%m%dT%H%M%SZ``).

        Returns *None* when S3 is unreachable, the bucket is
        unconfigured, or no manifests exist.  All failures are logged as
        warnings so that eval runs degrade gracefully.
        """
        import logging

        import boto3  # type: ignore[import-untyped]

        logger = logging.getLogger(__name__)

        # Build listing prefix (same strip/filter/join as run_orchestrator.py)
        parts = [p for p in (corpus_prefix.strip("/"), manifests_subprefix.strip("/")) if p]
        search_prefix = "/".join(parts) + "/" if parts else "manifests/"

        try:
            s3 = boto3.client("s3")
            paginator = s3.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=bucket,
                Prefix=search_prefix,
                Delimiter="/",
            )

            index_ids: list[str] = []
            for page in pages:
                for entry in page.get("CommonPrefixes", []):
                    full_prefix = entry["Prefix"]
                    index_id = full_prefix.rstrip("/").rsplit("/", 1)[-1]
                    index_ids.append(index_id)

            if not index_ids:
                logger.warning(
                    "No manifests found at s3://%s/%s — skipping auto-discovery",
                    bucket,
                    search_prefix,
                )
                return None

            latest_index_id = sorted(index_ids)[-1]
            key_parts = [
                p
                for p in (
                    corpus_prefix.strip("/"),
                    manifests_subprefix.strip("/"),
                    latest_index_id,
                    "manifest.json",
                )
                if p
            ]
            manifest_key = "/".join(key_parts)
            manifest_uri = f"s3://{bucket}/{manifest_key}"

            logger.info("Auto-discovered latest manifest: %s", manifest_uri)
            return IndexManifest.load_uri(manifest_uri)

        except Exception:
            logger.warning(
                "Failed to auto-discover manifest from s3://%s/%s",
                bucket,
                search_prefix,
                exc_info=True,
            )
            return None
