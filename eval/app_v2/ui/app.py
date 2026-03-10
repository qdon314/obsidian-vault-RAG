from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import streamlit as st

from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.engine.loaders.bundle import build_bundle
from eval.app_v2.engine.loaders.s3_runs import list_s3_runs, sync_run_from_s3

logger = logging.getLogger(__name__)

DEFAULT_RUNS_DIR = Path("eval/runs")
_RUN_DIR_PATTERN = re.compile(r"run_(\d{4}_\d{2}_\d{2}T\d{2}-\d{2})")


def discover_runs(runs_dir: Path) -> list[tuple[str, Path]]:
    """Return [(display_name, run_dir)] sorted newest-first."""
    entries: list[tuple[datetime, str, Path]] = []
    for d in runs_dir.iterdir():
        if not d.is_dir() or not (d / "metrics.json").exists():
            continue
        m = _RUN_DIR_PATTERN.match(d.name)
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y_%m_%dT%H-%M").replace(tzinfo=UTC)
                entries.append((ts, d.name, d))
            except ValueError:
                pass
    entries.sort(reverse=True)
    return [(name, path) for _, name, path in entries]


@st.cache_data(show_spinner="Building run bundle...")
def load_bundle(run_id: str, run_dir_str: str) -> RunBundle:
    return build_bundle(Path(run_dir_str))


def discover_runs_s3(
    client: Any,
    bucket: str,
    prefix: str,
    cache_dir: Path,
) -> list[tuple[str, Path]]:
    """List runs in S3 and return (display_name, local_cache_path) pairs.

    The local_cache_path is where the run *will be* cached once loaded —
    it may not exist yet. The actual download happens in load_bundle_from_s3.
    """
    entries = list_s3_runs(client, bucket, prefix)
    return [(name, cache_dir / name) for _ts, name, _s3_prefix in entries]


@st.cache_data(show_spinner="Syncing run from S3...")
def load_bundle_from_s3(
    run_id: str,
    bucket: str,
    s3_prefix: str,
    cache_dir_str: str,
) -> RunBundle:
    """Download run from S3 to local cache, then build and return a RunBundle."""
    import boto3  # type: ignore[import-untyped]
    client = boto3.client("s3")
    local_dir = sync_run_from_s3(client, bucket, s3_prefix, Path(cache_dir_str))
    return build_bundle(local_dir)


def run_selector_widget(
    runs: list[tuple[str, Path]],
    *,
    key: str = "run_a",
    label: str = "Select run",
) -> tuple[str, Path] | None:
    if not runs:
        st.warning("No runs found in eval/runs/")
        return None
    names = [name for name, _ in runs]
    idx = st.selectbox(label, range(len(names)), format_func=lambda i: names[i], key=key)
    return runs[idx]
