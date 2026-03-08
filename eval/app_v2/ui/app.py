from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from pathlib import Path

import streamlit as st

from eval.app_v2.engine.domain.models import RunBundle
from eval.app_v2.engine.loaders.bundle import build_bundle

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


def run_selector_widget(runs: list[tuple[str, Path]]) -> tuple[str, Path] | None:
    if not runs:
        st.warning("No runs found in eval/runs/")
        return None
    names = [name for name, _ in runs]
    idx = st.selectbox("Select run", range(len(names)), format_func=lambda i: names[i])
    return runs[idx]
