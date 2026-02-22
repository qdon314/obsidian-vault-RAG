"""S3-backed eval run loader.

Discovers and lazily loads evaluation runs stored in S3 by
run_remote_eval.py. Downloads only metrics.json for browsing;
full run data (results.jsonl, traces.jsonl) is fetched on demand
and cached to disk.

S3 layout (written by scripts/run_remote_eval.py):
    s3://{bucket}/{prefix}/{run_label}/
        metrics.json
        results.jsonl
        traces.jsonl   (optional)
        config.json    (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from eval.app.results.adapters.filesystem_loader import (
    FilesystemRunLoader,
)
from eval.app.results.domain.models import LoadedRun, RunSummary

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class S3RunLoader:
    """Loads evaluation runs from S3.

    Implements the RunLoader protocol via structural subtyping.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    prefix:
        S3 key prefix for eval runs (e.g. "eval/runs").
    cache_dir:
        Local directory for caching downloaded run data.
    client:
        A boto3 S3 client. Created lazily if not provided.
    """

    bucket: str
    prefix: str
    cache_dir: Path
    client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            import boto3  # type: ignore[import-untyped]

            object.__setattr__(self, "client", boto3.client("s3"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── RunLoader protocol ──────────────────────────────────────

    def discover_runs(self) -> list[RunSummary]:
        """List all runs in S3 by enumerating prefixes."""
        try:
            paginator = self.client.get_paginator("list_objects_v2")
            pages = paginator.paginate(
                Bucket=self.bucket,
                Prefix=self.prefix.rstrip("/") + "/",
                Delimiter="/",
            )
        except Exception:
            logger.warning("Failed to list S3 runs at s3://%s/%s", self.bucket, self.prefix, exc_info=True)
            return []

        summaries: list[RunSummary] = []

        for page in pages:
            for entry in page.get("CommonPrefixes", []):
                full_prefix = entry["Prefix"]
                run_id = full_prefix.rstrip("/").rsplit("/", 1)[-1]

                try:
                    summary = self._load_summary_from_s3(run_id)
                    summaries.append(summary)
                except Exception:
                    logger.debug("Skipping S3 run %s: could not load metrics.json", run_id, exc_info=True)
                    continue

        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        return summaries

    def load_run(self, run_id: str) -> LoadedRun:
        """Download full run data (or use cache) and parse."""
        local_run_dir = self.cache_dir / run_id

        if not self._is_cached(local_run_dir):
            self._download_run(run_id, local_run_dir)

        # Delegate parsing to FilesystemRunLoader
        fs_loader = FilesystemRunLoader(runs_dir=self.cache_dir)
        loaded = fs_loader._load_run_from_dir(local_run_dir)

        # Override the summary source to "s3"
        patched_summary = replace(loaded.summary, source="s3")

        return LoadedRun(
            summary=patched_summary,
            meta=loaded.meta,
            aggregates=loaded.aggregates,
            results=loaded.results,
            traces=loaded.traces,
            raw_metrics=loaded.raw_metrics,
        )

    def load_summary(self, run_id: str) -> RunSummary:
        """Load only the summary for a single run."""
        return self._load_summary_from_s3(run_id)

    # ── Internal helpers ────────────────────────────────────────

    def _load_summary_from_s3(self, run_id: str) -> RunSummary:
        """Download metrics.json and parse into RunSummary."""
        local_run_dir = self.cache_dir / run_id
        metrics_path = local_run_dir / "metrics.json"

        if not metrics_path.exists():
            local_run_dir.mkdir(parents=True, exist_ok=True)
            s3_key = f"{self.prefix.rstrip('/')}/{run_id}/metrics.json"
            self.client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=str(metrics_path),
            )

        fs_loader = FilesystemRunLoader(runs_dir=self.cache_dir)
        summary = fs_loader._load_summary_from_dir(local_run_dir)

        # Return a copy with source="s3"
        return replace(summary, source="s3")

    def _is_cached(self, local_run_dir: Path) -> bool:
        """Check if a run is fully cached (has metrics.json + results.jsonl)."""
        return (
            (local_run_dir / "metrics.json").exists()
            and (local_run_dir / "results.jsonl").exists()
        )

    def _download_run(self, run_id: str, local_run_dir: Path) -> None:
        """Download all files for a run from S3 to local cache."""
        local_run_dir.mkdir(parents=True, exist_ok=True)
        s3_prefix = f"{self.prefix.rstrip('/')}/{run_id}/"

        paginator = self.client.get_paginator("list_objects_v2")
        pages = paginator.paginate(
            Bucket=self.bucket,
            Prefix=s3_prefix,
        )

        for page in pages:
            for obj in page.get("Contents", []):
                s3_key = obj["Key"]
                relative = s3_key[len(s3_prefix):]
                if not relative:
                    continue

                local_path = local_run_dir / relative
                local_path.parent.mkdir(parents=True, exist_ok=True)
                self.client.download_file(
                    Bucket=self.bucket,
                    Key=s3_key,
                    Filename=str(local_path),
                )
                logger.debug("Cached s3://%s/%s -> %s", self.bucket, s3_key, local_path)
