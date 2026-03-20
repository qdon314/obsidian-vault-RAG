# eval/app_v2/engine/loaders/s3_runs.py
"""S3 run discovery and local-cache sync for the Results Analyzer v2.

Usage pattern (download-then-load):
    client = boto3.client("s3")
    runs = list_s3_runs(client, bucket="my-bucket", prefix="eval/runs")
    local_dir = sync_run_from_s3(client, bucket, s3_prefix=runs[0][2], cache_dir=Path("/tmp/eval-cache"))
    bundle = build_bundle(local_dir)
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# S3 runs use ISO format: 2026-02-14T21-05-06  (no run_ prefix, dashes throughout)
# Local runs use:        run_2026_02_14T22-17  (run_ prefix, underscores in date)
_S3_RUN_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})$")
_S3_TS_FORMAT = "%Y-%m-%dT%H-%M-%S"


def list_s3_runs(
    client: Any,
    bucket: str,
    prefix: str,
) -> list[tuple[datetime, str, str]]:
    """List run prefixes in S3.

    Returns a list of (timestamp, display_name, s3_prefix) sorted newest-first.
    Uses the S3 common-prefix delimiter so only top-level run directories are returned.
    """
    clean_prefix = prefix.rstrip("/") + "/"
    paginator = client.get_paginator("list_objects_v2")
    seen: dict[str, tuple[datetime, str, str]] = {}

    for page in paginator.paginate(Bucket=bucket, Prefix=clean_prefix, Delimiter="/"):
        for cp in page.get("CommonPrefixes", []):
            run_name = cp["Prefix"].rstrip("/").split("/")[-1]
            m = _S3_RUN_PATTERN.match(run_name)
            if not m:
                continue
            try:
                ts = datetime.strptime(m.group(1), _S3_TS_FORMAT).replace(tzinfo=UTC)
            except ValueError:
                continue
            s3_prefix = cp["Prefix"].rstrip("/")
            seen[run_name] = (ts, run_name, s3_prefix)

    return sorted(seen.values(), key=lambda x: x[0], reverse=True)


def sync_run_from_s3(
    client: Any,
    bucket: str,
    s3_prefix: str,
    cache_dir: Path,
) -> Path:
    """Download a run from S3 to a local cache directory.

    If metrics.json already exists in the cache, the download is skipped
    (metrics.json is the last file written by the eval pipeline, so its
    presence means the run is fully synced).

    Returns the local run directory path.
    """
    from rag.adapters.artifacts.s3_store import S3ArtifactStore

    run_name = s3_prefix.rstrip("/").split("/")[-1]
    local_dir = cache_dir / run_name

    if (local_dir / "metrics.json").exists():
        return local_dir

    local_dir.mkdir(parents=True, exist_ok=True)
    store = S3ArtifactStore(bucket=bucket, client=client)
    store.pull(s3_prefix, local_dir)
    return local_dir
