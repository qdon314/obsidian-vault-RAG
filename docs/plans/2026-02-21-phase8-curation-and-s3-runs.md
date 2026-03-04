# Phase 8: Query Curation & S3 Eval Run Loading — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add S3-backed eval run loading to the Streamlit results analyzer (unified run list with lazy download), and build a Streamlit query curation UI for approving/editing/rejecting generated eval queries.

**Architecture:** A new `S3RunLoader` adapter implements the existing `RunLoader` protocol. It discovers runs via S3 `list_objects_v2`, downloads only `metrics.json` for browsing, and lazily downloads full run data on selection (cached to disk). The `InMemoryRunRepository` composes both filesystem and S3 loaders. Separately, a new Streamlit page provides a review UI for draft queries with approve/edit/reject actions and JSONL export.

**Tech Stack:** Python, boto3, Streamlit, existing `eval/app/results` hexagonal architecture, `EvalQuery` from `rag.eval.schema`.

---

## Task 1: Add `source` Field to `RunSummary`

**Files:**
- Modify: `eval/app/results/domain/models.py:19-44`
- Test: `tests/eval/app/test_models.py` (create if needed)

**Step 1: Write the failing test**

Create `tests/eval/app/test_run_summary_source.py`:

```python
"""Test that RunSummary supports a source field."""
from datetime import UTC, datetime
from pathlib import Path

from eval.app.results.domain.models import RunSummary


def test_run_summary_defaults_to_local_source() -> None:
    summary = RunSummary(
        run_id="run_2026_02_20T19-49",
        timestamp=datetime(2026, 2, 20, 19, 49, tzinfo=UTC),
        display_name="run_2026_02_20T19-49",
        run_dir=Path("/tmp/runs/run_2026_02_20T19-49"),
        num_queries=50,
        run_generation=True,
        generator_model="gpt-4o",
        embedder_model="text-embedding-3-small",
        reranker_name=None,
        overall_recall_at_10=0.85,
        overall_ndcg_at_10=0.78,
        avg_quality_score=0.72,
        avg_latency_ms=350.0,
    )
    assert summary.source == "local"


def test_run_summary_accepts_s3_source() -> None:
    summary = RunSummary(
        run_id="run_2026_02_20T19-49",
        timestamp=datetime(2026, 2, 20, 19, 49, tzinfo=UTC),
        display_name="run_2026_02_20T19-49",
        run_dir=Path("/tmp/cache/run_2026_02_20T19-49"),
        num_queries=50,
        run_generation=True,
        generator_model=None,
        embedder_model=None,
        reranker_name=None,
        overall_recall_at_10=None,
        overall_ndcg_at_10=None,
        avg_quality_score=None,
        avg_latency_ms=None,
        source="s3",
    )
    assert summary.source == "s3"
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/app/test_run_summary_source.py -v`
Expected: FAIL — `RunSummary.__init__() got an unexpected keyword argument 'source'`

**Step 3: Add `source` field to `RunSummary`**

In `eval/app/results/domain/models.py`, add after line 43 (`avg_latency_ms: float | None`):

```python
    # Source indicator for UI badging
    source: str = "local"  # "local" | "s3"
```

**Note:** `RunSummary` is `frozen=True` but does NOT use `slots=True`, so adding a field with a default is safe — no need for the `field(init=False)` + `object.__setattr__` dance.

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/eval/app/test_run_summary_source.py -v`
Expected: PASS

**Step 5: Run existing tests to check for regressions**

Run: `./scripts/py -m pytest tests/eval/ -v`
Expected: All existing tests pass (the default `"local"` preserves backward compatibility).

**Step 6: Commit**

```
feat(eval): add source field to RunSummary for local/s3 badging
```

---

## Task 2: S3RunLoader Adapter

**Files:**
- Create: `eval/app/results/adapters/s3_loader.py`
- Test: `tests/eval/app/test_s3_loader.py`

**Step 1: Write the failing test**

Create `tests/eval/app/test_s3_loader.py`:

```python
"""Tests for S3RunLoader — eval run discovery and loading from S3."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from eval.app.results.adapters.s3_loader import S3RunLoader


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".s3-cache"
    d.mkdir()
    return d


@pytest.fixture
def mock_s3_client() -> MagicMock:
    return MagicMock()


def _make_metrics_json(*, run_name: str = "test-run", num_queries: int = 50) -> str:
    """Build a minimal metrics.json blob."""
    return json.dumps({
        "meta": {
            "run_id": "abc-123",
            "started_at": "2026-02-20T19:49:00+00:00",
            "run_name": run_name,
            "run_generation": True,
            "generator_model": "gpt-4o",
            "embedder_model": "text-embedding-3-small",
            "reranker_name": None,
        },
        "overall": {
            "num_queries": num_queries,
            "recall@10": 0.85,
            "ndcg@10": 0.78,
        },
        "answer_quality": {"avg_quality_score": 0.72},
        "latency_ms": {"avg": 350.0},
    })


class TestDiscoverRuns:
    def test_lists_s3_prefixes_and_builds_summaries(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        # Mock list_objects_v2 to return two "run" prefixes
        mock_s3_client.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": "eval/runs/run_2026_02_20T19-49/"},
                {"Prefix": "eval/runs/run_2026_02_20T21-05/"},
            ],
        }

        # Mock download_file to write a metrics.json into the cache
        def fake_download(Bucket, Key, Filename):
            Path(Filename).parent.mkdir(parents=True, exist_ok=True)
            Path(Filename).write_text(_make_metrics_json())

        mock_s3_client.download_file.side_effect = fake_download

        loader = S3RunLoader(
            bucket="test-bucket",
            prefix="eval/runs",
            cache_dir=cache_dir,
            client=mock_s3_client,
        )

        summaries = loader.discover_runs()

        assert len(summaries) == 2
        assert all(s.source == "s3" for s in summaries)
        # Sorted by timestamp descending
        assert summaries[0].run_id == "run_2026_02_20T21-05"

    def test_skips_prefixes_without_metrics(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        mock_s3_client.list_objects_v2.return_value = {
            "CommonPrefixes": [
                {"Prefix": "eval/runs/run_2026_02_20T19-49/"},
            ],
        }
        # download_file raises — simulating missing metrics.json
        mock_s3_client.download_file.side_effect = Exception("NoSuchKey")

        loader = S3RunLoader(
            bucket="test-bucket",
            prefix="eval/runs",
            cache_dir=cache_dir,
            client=mock_s3_client,
        )

        summaries = loader.discover_runs()
        assert summaries == []

    def test_empty_bucket_returns_empty_list(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        mock_s3_client.list_objects_v2.return_value = {}

        loader = S3RunLoader(
            bucket="test-bucket",
            prefix="eval/runs",
            cache_dir=cache_dir,
            client=mock_s3_client,
        )

        assert loader.discover_runs() == []


class TestLoadRun:
    def test_downloads_full_run_and_parses(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        # Simulate S3 listing the run's files
        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "eval/runs/run_2026_02_20T19-49/metrics.json"},
                {"Key": "eval/runs/run_2026_02_20T19-49/results.jsonl"},
            ],
        }

        def fake_download(Bucket, Key, Filename):
            Path(Filename).parent.mkdir(parents=True, exist_ok=True)
            if Key.endswith("metrics.json"):
                Path(Filename).write_text(_make_metrics_json())
            elif Key.endswith("results.jsonl"):
                # Minimal empty results
                Path(Filename).write_text("")

        mock_s3_client.download_file.side_effect = fake_download

        loader = S3RunLoader(
            bucket="test-bucket",
            prefix="eval/runs",
            cache_dir=cache_dir,
            client=mock_s3_client,
        )

        loaded = loader.load_run("run_2026_02_20T19-49")
        assert loaded.summary.source == "s3"
        assert loaded.summary.run_id == "run_2026_02_20T19-49"

    def test_uses_disk_cache_on_second_load(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        # Pre-populate the cache directory
        run_dir = cache_dir / "run_2026_02_20T19-49"
        run_dir.mkdir(parents=True)
        (run_dir / "metrics.json").write_text(_make_metrics_json())
        (run_dir / "results.jsonl").write_text("")

        loader = S3RunLoader(
            bucket="test-bucket",
            prefix="eval/runs",
            cache_dir=cache_dir,
            client=mock_s3_client,
        )

        loaded = loader.load_run("run_2026_02_20T19-49")
        assert loaded.summary.run_id == "run_2026_02_20T19-49"

        # S3 was never called — served from cache
        mock_s3_client.list_objects_v2.assert_not_called()
        mock_s3_client.download_file.assert_not_called()
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/app/test_s3_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.app.results.adapters.s3_loader'`

**Step 3: Implement `S3RunLoader`**

Create `eval/app/results/adapters/s3_loader.py`:

```python
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eval.app.results.adapters.filesystem_loader import (
    FilesystemRunLoader,
    _parse_timestamp_from_dirname,
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
            response = self.client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=self.prefix.rstrip("/") + "/",
                Delimiter="/",
            )
        except Exception:
            logger.warning("Failed to list S3 runs at s3://%s/%s", self.bucket, self.prefix, exc_info=True)
            return []

        prefixes = response.get("CommonPrefixes", [])
        summaries: list[RunSummary] = []

        for entry in prefixes:
            full_prefix = entry["Prefix"]  # e.g. "eval/runs/run_2026_02_20T19-49/"
            run_id = full_prefix.rstrip("/").rsplit("/", 1)[-1]

            if not run_id.startswith("run_"):
                continue

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
        patched_summary = RunSummary(
            run_id=loaded.summary.run_id,
            timestamp=loaded.summary.timestamp,
            display_name=loaded.summary.display_name,
            run_dir=loaded.summary.run_dir,
            num_queries=loaded.summary.num_queries,
            run_generation=loaded.summary.run_generation,
            generator_model=loaded.summary.generator_model,
            embedder_model=loaded.summary.embedder_model,
            reranker_name=loaded.summary.reranker_name,
            overall_recall_at_10=loaded.summary.overall_recall_at_10,
            overall_ndcg_at_10=loaded.summary.overall_ndcg_at_10,
            avg_quality_score=loaded.summary.avg_quality_score,
            avg_latency_ms=loaded.summary.avg_latency_ms,
            source="s3",
        )

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
        return RunSummary(
            run_id=summary.run_id,
            timestamp=summary.timestamp,
            display_name=summary.display_name,
            run_dir=summary.run_dir,
            num_queries=summary.num_queries,
            run_generation=summary.run_generation,
            generator_model=summary.generator_model,
            embedder_model=summary.embedder_model,
            reranker_name=summary.reranker_name,
            overall_recall_at_10=summary.overall_recall_at_10,
            overall_ndcg_at_10=summary.overall_ndcg_at_10,
            avg_quality_score=summary.avg_quality_score,
            avg_latency_ms=summary.avg_latency_ms,
            source="s3",
        )

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

        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=s3_prefix,
        )

        for obj in response.get("Contents", []):
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
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/eval/app/test_s3_loader.py -v`
Expected: All tests PASS

**Step 5: Run type checker**

Run: `./scripts/py -m mypy eval/app/results/adapters/s3_loader.py --config-file pyproject.toml`
Expected: No errors (or only boto3 `import-untyped` notes, already suppressed).

**Step 6: Commit**

```
feat(eval): add S3RunLoader for discovering and loading remote eval runs
```

---

## Task 3: Wire S3 Loader into Repository

**Files:**
- Modify: `eval/app/results/adapters/repository.py:21-30`
- Test: `tests/eval/app/test_repository_s3.py`

**Step 1: Write the failing test**

Create `tests/eval/app/test_repository_s3.py`:

```python
"""Tests for InMemoryRunRepository with S3 loader integration."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from eval.app.results.adapters.filesystem_loader import FilesystemRunLoader
from eval.app.results.adapters.repository import InMemoryRunRepository
from eval.app.results.domain.models import RunSummary


def _make_summary(run_id: str, source: str = "local") -> RunSummary:
    ts_str = run_id.replace("run_", "").replace("T", " ").replace("-", ":")
    return RunSummary(
        run_id=run_id,
        timestamp=datetime(2026, 2, 20, 19, 49, tzinfo=UTC),
        display_name=run_id,
        run_dir=Path(f"/tmp/{run_id}"),
        num_queries=50,
        run_generation=True,
        generator_model=None,
        embedder_model=None,
        reranker_name=None,
        overall_recall_at_10=0.85,
        overall_ndcg_at_10=None,
        avg_quality_score=None,
        avg_latency_ms=None,
        source=source,
    )


def test_list_runs_merges_local_and_s3(tmp_path: Path) -> None:
    """Repository merges runs from both filesystem and S3 loaders."""
    fs_loader = FilesystemRunLoader(runs_dir=tmp_path / "runs")
    (tmp_path / "runs").mkdir()

    s3_loader = MagicMock()
    s3_loader.discover_runs.return_value = [
        _make_summary("run_2026_02_20T21-05", source="s3"),
    ]

    repo = InMemoryRunRepository(loader=fs_loader, s3_loader=s3_loader)
    runs = repo.list_runs()

    # Should include the S3 run
    assert any(r.run_id == "run_2026_02_20T21-05" for r in runs)
    assert any(r.source == "s3" for r in runs)


def test_local_run_wins_over_s3_duplicate(tmp_path: Path) -> None:
    """When the same run_id exists locally and in S3, local wins."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    # Create a local run directory with metrics.json
    run_dir = runs_dir / "run_2026_02_20T19-49"
    run_dir.mkdir()
    import json
    (run_dir / "metrics.json").write_text(json.dumps({
        "meta": {"run_id": "abc", "started_at": "2026-02-20T19:49:00+00:00"},
        "overall": {"num_queries": 50},
    }))

    fs_loader = FilesystemRunLoader(runs_dir=runs_dir)

    s3_loader = MagicMock()
    s3_loader.discover_runs.return_value = [
        _make_summary("run_2026_02_20T19-49", source="s3"),
    ]

    repo = InMemoryRunRepository(loader=fs_loader, s3_loader=s3_loader)
    runs = repo.list_runs()

    # Should have exactly one entry for this run_id, and it should be local
    matching = [r for r in runs if r.run_id == "run_2026_02_20T19-49"]
    assert len(matching) == 1
    assert matching[0].source == "local"


def test_list_runs_works_without_s3_loader(tmp_path: Path) -> None:
    """Repository works fine when s3_loader is None."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    fs_loader = FilesystemRunLoader(runs_dir=runs_dir)
    repo = InMemoryRunRepository(loader=fs_loader, s3_loader=None)

    runs = repo.list_runs()
    assert runs == []


def test_get_run_falls_through_to_s3(tmp_path: Path) -> None:
    """get_run tries S3 loader when local loader raises FileNotFoundError."""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()

    fs_loader = FilesystemRunLoader(runs_dir=runs_dir)

    mock_loaded_run = MagicMock()
    mock_loaded_run.summary.source = "s3"

    s3_loader = MagicMock()
    s3_loader.load_run.return_value = mock_loaded_run

    repo = InMemoryRunRepository(loader=fs_loader, s3_loader=s3_loader)
    result = repo.get_run("run_2026_02_20T21-05")

    assert result is mock_loaded_run
    s3_loader.load_run.assert_called_once_with("run_2026_02_20T21-05")
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/app/test_repository_s3.py -v`
Expected: FAIL — `InMemoryRunRepository.__init__() got an unexpected keyword argument 's3_loader'`

**Step 3: Add S3 loader support to `InMemoryRunRepository`**

In `eval/app/results/adapters/repository.py`:

Add import at the top (after line 14):

```python
from eval.app.results.adapters.s3_loader import S3RunLoader
```

Add field to `InMemoryRunRepository` (after `additional_paths` on line 30):

```python
    s3_loader: S3RunLoader | None = None
```

Update `_discover_all_runs()` (after the `additional_paths` loop, before the final sort on line 109). Add this block before `summaries.sort(...)`:

```python
        # Add runs from S3 loader
        if self.s3_loader is not None:
            try:
                s3_summaries = self.s3_loader.discover_runs()
                for summary in s3_summaries:
                    if summary.run_id not in seen_ids:
                        summaries.append(summary)
                        seen_ids.add(summary.run_id)
            except Exception as e:
                logger.warning(f"Failed to discover S3 runs: {e}")
```

Update `get_run()` (after the `additional_paths` loop on line 58, before the `raise`). Add:

```python
        # Try S3 loader
        if self.s3_loader is not None:
            try:
                return self.s3_loader.load_run(run_id)
            except (FileNotFoundError, Exception):
                pass
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/eval/app/test_repository_s3.py -v`
Expected: All tests PASS

**Step 5: Run all eval tests for regressions**

Run: `./scripts/py -m pytest tests/eval/ -v`
Expected: All pass. The new `s3_loader=None` default preserves backward compatibility.

**Step 6: Commit**

```
feat(eval): add S3 loader support to InMemoryRunRepository
```

---

## Task 4: Update Run Selector UI for S3 Badges

**Files:**
- Modify: `eval/app/results/ui/run_selector.py:38-54`

**Step 1: Update `format_run` to show source badge**

In `eval/app/results/ui/run_selector.py`, modify the `format_run` function (line 38-54).

Replace:

```python
    def format_run(run: RunSummary) -> str:
        """Format run for display."""
        parts = [run.display_name]
```

With:

```python
    def format_run(run: RunSummary) -> str:
        """Format run for display."""
        name = run.display_name
        if hasattr(run, "source") and run.source == "s3":
            name = f"{run.display_name} [S3]"
        parts = [name]
```

**Step 2: Verify manually**

No automated test needed — this is a display-only change. Verify by running `make results` (or `./scripts/py -m streamlit run eval/app/results_analyzer.py`) once S3 is wired in Task 5.

**Step 3: Commit**

```
feat(eval): show [S3] badge in run selector for remote runs
```

---

## Task 5: Wire S3 Loader into Results Analyzer

**Files:**
- Modify: `eval/app/results_analyzer.py:28-67`

**Step 1: Update imports**

In `eval/app/results_analyzer.py`, add after line 29 (`from eval.app.results.adapters.repository import InMemoryRunRepository`):

```python
from eval.app.results.adapters.s3_loader import S3RunLoader
```

Also add at the top with stdlib imports:

```python
import os
```

**Step 2: Update `get_repository()` to conditionally wire S3 loader**

Replace the `get_repository()` function (lines 59-67):

```python
@st.cache_resource
def get_repository() -> InMemoryRunRepository:
    """Initialize the run repository.

    If RAG_EVAL_S3_BUCKET is set (or settings.toml has an S3 bucket),
    an S3RunLoader is wired in alongside the filesystem loader. Runs
    from both sources appear in a unified list.
    """
    runs_dir = DEFAULT_RUNS_DIR
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    loader = FilesystemRunLoader(runs_dir=runs_dir)

    # Conditionally wire S3 loader
    s3_loader: S3RunLoader | None = None
    s3_bucket = os.environ.get("RAG_EVAL_S3_BUCKET", "")
    s3_prefix = os.environ.get("RAG_EVAL_S3_PREFIX", "eval")

    if s3_bucket:
        try:
            s3_cache_dir = runs_dir / ".s3-cache"
            s3_loader = S3RunLoader(
                bucket=s3_bucket,
                prefix=f"{s3_prefix}/runs",
                cache_dir=s3_cache_dir,
            )
            logger.info("S3 run loader enabled: s3://%s/%s/runs", s3_bucket, s3_prefix)
        except Exception:
            logger.warning("Failed to initialize S3 run loader", exc_info=True)

    return InMemoryRunRepository(loader=loader, s3_loader=s3_loader)
```

**Step 3: Update the sidebar info caption**

In `main()`, after line 160 (`st.caption(f"Runs directory: {DEFAULT_RUNS_DIR}")`), add:

```python
        s3_bucket = os.environ.get("RAG_EVAL_S3_BUCKET", "")
        if s3_bucket:
            s3_prefix = os.environ.get("RAG_EVAL_S3_PREFIX", "eval")
            st.caption(f"S3: s3://{s3_bucket}/{s3_prefix}/runs/")
```

**Step 4: Add `.s3-cache` to `.gitignore`**

Check if `eval/runs/.s3-cache` would be gitignored already. If `eval/runs/` is ignored, this is already covered. If not, add to `.gitignore`:

```
eval/runs/.s3-cache/
```

**Step 5: Verify locally (no S3)**

Run: `./scripts/py -m streamlit run eval/app/results_analyzer.py`
Expected: App starts normally. No S3 errors (since `RAG_EVAL_S3_BUCKET` is not set). Local runs appear as before.

**Step 6: Verify with S3 (if credentials available)**

Run: `RAG_EVAL_S3_BUCKET=obsidian-rag-corpus ./scripts/py -m streamlit run eval/app/results_analyzer.py`
Expected: S3 runs appear in the selector alongside local runs, with `[S3]` badge. Selecting an S3 run triggers download with spinner, then displays normally.

**Step 7: Commit**

```
feat(eval): wire S3RunLoader into Streamlit results analyzer

S3 loading is conditional on RAG_EVAL_S3_BUCKET env var.
When unset, the app behaves exactly as before (local-only).
```

---

## Task 6: Query Curator Streamlit Page — Foundation

**Files:**
- Create: `eval/app/query_curator.py`
- Test: Manual (Streamlit page)

**Step 1: Create the query curator page**

Create `eval/app/query_curator.py`:

```python
#!/usr/bin/env python3
"""
Query Curator — Streamlit App

Interactive review UI for approving, editing, and rejecting
generated evaluation queries before they enter the production
eval dataset.

Usage:
    streamlit run eval/app/query_curator.py

    Or via Makefile:
    make curate-case-queries
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import streamlit as st

from rag.eval.schema import Difficulty, EvalQuery, QueryType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parents[2]
DEFAULT_DRAFT_PATH = PROJECT_ROOT / "eval" / "datasets" / "case_generated_queries_DRAFT.jsonl"
DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "eval" / "datasets" / "case_generated_queries.jsonl"


def load_queries(path: Path) -> list[dict]:
    """Load queries from JSONL as raw dicts (for editability)."""
    if not path.exists():
        return []
    queries = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                queries.append(json.loads(line))
    return queries


def save_queries(queries: list[dict], path: Path) -> None:
    """Save queries to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q, default=str) + "\n")


def main() -> None:
    st.set_page_config(
        page_title="Query Curator",
        page_icon="",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Query Curator")
    st.caption("Review, edit, and approve generated eval queries")

    # ── Sidebar: File selection & actions ────────────────────
    with st.sidebar:
        st.subheader("Input")
        draft_path = st.text_input(
            "Draft JSONL path",
            value=str(DEFAULT_DRAFT_PATH),
            key="draft_path",
        )
        draft_path = Path(draft_path)

        if st.button("Load queries", key="load_btn"):
            queries = load_queries(draft_path)
            if queries:
                st.session_state["queries"] = queries
                st.session_state["decisions"] = {
                    q["qid"]: "pending" for q in queries
                }
                st.session_state["edits"] = {}
                st.success(f"Loaded {len(queries)} queries")
            else:
                st.error(f"No queries found at {draft_path}")

        st.divider()

        # Stats
        if "queries" in st.session_state and "decisions" in st.session_state:
            decisions = st.session_state["decisions"]
            total = len(decisions)
            approved = sum(1 for v in decisions.values() if v == "approved")
            rejected = sum(1 for v in decisions.values() if v == "rejected")
            pending = sum(1 for v in decisions.values() if v == "pending")
            st.metric("Total", total)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Approved", approved)
            with col2:
                st.metric("Rejected", rejected)
            with col3:
                st.metric("Pending", pending)

            st.divider()

            # Filters
            st.subheader("Filter")
            status_filter = st.selectbox(
                "Status",
                ["all", "pending", "approved", "rejected"],
                key="status_filter",
            )
            strategy_tags = sorted(
                {
                    tag
                    for q in st.session_state["queries"]
                    for tag in q.get("tags", [])
                }
            )
            tag_filter = st.multiselect("Tags", strategy_tags, key="tag_filter")

            st.divider()

            # Batch actions
            st.subheader("Batch Actions")
            if st.button("Approve all pending"):
                for qid, status in st.session_state["decisions"].items():
                    if status == "pending":
                        st.session_state["decisions"][qid] = "approved"
                st.rerun()

            selected_tag = st.selectbox(
                "Approve all with tag:", [""] + strategy_tags, key="batch_tag"
            )
            if selected_tag and st.button(f"Approve all '{selected_tag}'"):
                for q in st.session_state["queries"]:
                    if selected_tag in q.get("tags", []):
                        st.session_state["decisions"][q["qid"]] = "approved"
                st.rerun()

            st.divider()

            # Export
            st.subheader("Export")
            output_path = st.text_input(
                "Output JSONL path",
                value=str(DEFAULT_OUTPUT_PATH),
                key="output_path",
            )
            if st.button("Export approved queries", key="export_btn"):
                approved_queries = _collect_approved()
                if approved_queries:
                    save_queries(approved_queries, Path(output_path))
                    st.success(f"Exported {len(approved_queries)} queries to {output_path}")
                else:
                    st.warning("No approved queries to export")

    # ── Main content: Query review ──────────────────────────
    if "queries" not in st.session_state:
        st.info(f"Click **Load queries** in the sidebar to start reviewing.\n\nExpected file: `{DEFAULT_DRAFT_PATH}`")
        return

    queries = st.session_state["queries"]
    decisions = st.session_state["decisions"]

    # Apply filters
    filtered = queries
    status_filter = st.session_state.get("status_filter", "all")
    if status_filter != "all":
        filtered = [q for q in filtered if decisions.get(q["qid"]) == status_filter]
    tag_filter = st.session_state.get("tag_filter", [])
    if tag_filter:
        filtered = [
            q for q in filtered if set(tag_filter) & set(q.get("tags", []))
        ]

    st.write(f"Showing {len(filtered)} of {len(queries)} queries")

    # Render each query
    for i, q in enumerate(filtered):
        _render_query_card(q, i)


def _render_query_card(q: dict, idx: int) -> None:
    """Render a single query review card."""
    qid = q["qid"]
    decisions = st.session_state["decisions"]
    current_status = decisions.get(qid, "pending")

    # Status color indicator
    status_prefix = {"approved": "+", "rejected": "-", "pending": "?"}.get(
        current_status, "?"
    )

    strategy = ", ".join(q.get("tags", []))
    difficulty = q.get("difficulty", "?")
    query_type = q.get("query_type", "?")

    label = (
        f"[{status_prefix}] {qid} | {query_type} · {difficulty} | {strategy}"
    )

    with st.expander(label, expanded=(current_status == "pending")):
        # ── Read-only metadata ──
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"Source: {q.get('source_case', 'n/a')}")
        with col2:
            st.caption(f"Doc type: {q.get('case_document_type', 'n/a')}")
        with col3:
            st.caption(f"Unanswerable: {'Yes' if q.get('is_unanswerable') else 'No'}")
        with col4:
            if q.get("unanswerable_reason"):
                st.caption(f"Reason: {q['unanswerable_reason']}")

        # ── Editable query text ──
        edit_key = f"edit_query_{qid}"
        edited_query = st.text_area(
            "Query",
            value=st.session_state.get(f"edits_{qid}_query", q["query"]),
            height=80,
            key=edit_key,
        )
        if edited_query != q["query"]:
            st.session_state[f"edits_{qid}_query"] = edited_query

        # ── Editable difficulty & type ──
        col_d, col_t = st.columns(2)
        with col_d:
            diff_options = [d.value for d in Difficulty]
            current_diff = q.get("difficulty", "easy")
            new_diff = st.selectbox(
                "Difficulty",
                diff_options,
                index=diff_options.index(current_diff) if current_diff in diff_options else 0,
                key=f"diff_{qid}",
            )
        with col_t:
            type_options = [t.value for t in QueryType]
            current_type = q.get("query_type", "factual")
            new_type = st.selectbox(
                "Query type",
                type_options,
                index=type_options.index(current_type) if current_type in type_options else 0,
                key=f"type_{qid}",
            )

        # ── Citations ──
        citations = q.get("relevant_citations", [])
        if isinstance(citations, set):
            citations = sorted(citations)
        st.text_input(
            "Relevant citations (comma-separated)",
            value=", ".join(citations),
            key=f"cit_{qid}",
        )

        # ── Decision buttons ──
        col_a, col_r, col_p = st.columns(3)
        with col_a:
            if st.button("Approve", key=f"approve_{qid}", type="primary"):
                decisions[qid] = "approved"
                st.rerun()
        with col_r:
            if st.button("Reject", key=f"reject_{qid}"):
                decisions[qid] = "rejected"
                st.rerun()
        with col_p:
            if st.button("Reset to pending", key=f"pending_{qid}"):
                decisions[qid] = "pending"
                st.rerun()


def _collect_approved() -> list[dict]:
    """Collect approved queries, applying any edits."""
    queries = st.session_state.get("queries", [])
    decisions = st.session_state.get("decisions", {})

    approved = []
    for q in queries:
        if decisions.get(q["qid"]) != "approved":
            continue

        # Apply edits
        out = dict(q)
        edited_query = st.session_state.get(f"edits_{q['qid']}_query")
        if edited_query:
            out["query"] = edited_query

        edited_diff = st.session_state.get(f"diff_{q['qid']}")
        if edited_diff:
            out["difficulty"] = edited_diff

        edited_type = st.session_state.get(f"type_{q['qid']}")
        if edited_type:
            out["query_type"] = edited_type

        edited_cit = st.session_state.get(f"cit_{q['qid']}")
        if edited_cit is not None:
            out["relevant_citations"] = [
                c.strip() for c in edited_cit.split(",") if c.strip()
            ]

        approved.append(out)

    return approved


if __name__ == "__main__":
    main()
```

**Step 2: Verify the page loads**

Run: `./scripts/py -m streamlit run eval/app/query_curator.py`
Expected: Page renders with "Click Load queries" prompt. Loading the existing `case_generated_queries.jsonl` (which has data) shows query cards.

**Step 3: Commit**

```
feat(eval): add Streamlit query curator page for reviewing draft queries
```

---

## Task 7: Dataset Merge Script

**Files:**
- Create: `scripts/merge_eval_datasets.py`
- Test: `tests/eval/test_merge_datasets.py`

**Step 1: Write the failing test**

Create `tests/eval/test_merge_datasets.py`:

```python
"""Tests for eval dataset merging."""
from __future__ import annotations

import json
from pathlib import Path


def _write_jsonl(path: Path, queries: list[dict]) -> None:
    with path.open("w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_merge_combines_and_deduplicates(tmp_path: Path) -> None:
    from scripts.merge_eval_datasets import merge_datasets

    file_a = tmp_path / "a.jsonl"
    file_b = tmp_path / "b.jsonl"
    output = tmp_path / "merged.jsonl"

    _write_jsonl(file_a, [
        {"qid": "q-001", "query": "What is X?", "dataset_source": "manual"},
        {"qid": "q-002", "query": "What is Y?", "dataset_source": "manual"},
    ])
    _write_jsonl(file_b, [
        {"qid": "q-003", "query": "What is Z?", "dataset_source": "case_generated"},
        {"qid": "q-004", "query": "What is X?", "dataset_source": "case_generated"},  # duplicate query text
    ])

    merge_datasets([file_a, file_b], output)

    merged = _read_jsonl(output)
    # q-001 and q-004 have the same query text — case_generated wins
    assert len(merged) == 3
    texts = {q["query"] for q in merged}
    assert texts == {"What is X?", "What is Y?", "What is Z?"}
    # The "What is X?" entry should be from case_generated (q-004)
    x_query = next(q for q in merged if q["query"] == "What is X?")
    assert x_query["qid"] == "q-004"


def test_merge_adds_dataset_source_if_missing(tmp_path: Path) -> None:
    from scripts.merge_eval_datasets import merge_datasets

    file_a = tmp_path / "a.jsonl"
    output = tmp_path / "merged.jsonl"

    _write_jsonl(file_a, [
        {"qid": "q-001", "query": "What is X?"},
    ])

    merge_datasets([file_a], output, default_source="manual")

    merged = _read_jsonl(output)
    assert merged[0]["dataset_source"] == "manual"
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/eval/test_merge_datasets.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.merge_eval_datasets'`

**Step 3: Implement the merge script**

Create `scripts/merge_eval_datasets.py`:

```python
"""Merge multiple eval query JSONL files into a single deduplicated dataset.

Deduplication is by query text. When duplicates exist, the entry from the
later file wins (case_generated takes precedence over manual).

Usage:
    ./scripts/py scripts/merge_eval_datasets.py \
        --inputs eval/datasets/regulatory_adversarial.jsonl \
                 eval/datasets/case_generated_queries.jsonl \
        --output eval/datasets/all_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def merge_datasets(
    input_files: list[Path],
    output_file: Path,
    *,
    default_source: str | None = None,
) -> None:
    """Merge JSONL files, deduplicating by query text.

    Later files win on duplicate query text.
    """
    # Collect all queries, keyed by query text (later wins)
    by_text: dict[str, dict] = {}

    for path in input_files:
        if not path.exists():
            logger.warning("Skipping missing file: %s", path)
            continue

        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                q = json.loads(line)

                # Add default source if missing
                if "dataset_source" not in q and default_source:
                    q["dataset_source"] = default_source

                by_text[q["query"]] = q

    # Write merged output
    queries = list(by_text.values())
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        for q in queries:
            f.write(json.dumps(q, default=str) + "\n")

    logger.info("Merged %d queries from %d files -> %s", len(queries), len(input_files), output_file)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    ap = argparse.ArgumentParser(description="Merge eval query datasets")
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="Input JSONL files")
    ap.add_argument("--output", type=Path, required=True, help="Output JSONL file")
    ap.add_argument("--default-source", type=str, default=None, help="Default dataset_source tag")
    args = ap.parse_args()

    merge_datasets(args.inputs, args.output, default_source=args.default_source)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `./scripts/py -m pytest tests/eval/test_merge_datasets.py -v`
Expected: All tests PASS

**Step 5: Commit**

```
feat(eval): add merge_eval_datasets script for combining query datasets
```

---

## Task 8: Makefile Targets

**Files:**
- Modify: `Makefile`

**Step 1: Add targets for curation and merge**

Add these targets to the `Makefile` (in the eval section, after the existing eval targets):

```makefile
curate-case-queries:  ## Open query curator Streamlit UI
	$(PY) -m streamlit run eval/app/query_curator.py

merge-eval-datasets:  ## Merge eval query datasets into all_queries.jsonl
	$(PY) scripts/merge_eval_datasets.py \
		--inputs eval/datasets/regulatory_adversarial.jsonl \
		         eval/datasets/case_generated_queries.jsonl \
		--output eval/datasets/all_queries.jsonl
```

**Step 2: Verify targets**

Run: `make curate-case-queries` — should launch the Streamlit curator.
Run: `make merge-eval-datasets` — should merge datasets (will error if files don't exist, which is fine for now).

**Step 3: Commit**

```
feat(make): add curate-case-queries and merge-eval-datasets targets
```

---

## Task 9: Update `.gitignore` for S3 Cache

**Files:**
- Modify: `.gitignore`

**Step 1: Check current gitignore**

Read `.gitignore` and check if `eval/runs/` or `.s3-cache` is already covered.

**Step 2: Add S3 cache exclusion if needed**

If not already covered, add:

```
# S3 eval run cache (downloaded by results analyzer)
eval/runs/.s3-cache/
```

**Step 3: Commit**

```
chore: gitignore S3 eval run cache directory
```

---

## Task 10: Update Phase 8 in Implementation Plan

**Files:**
- Modify: `plans/nrc_case_ingestion_implementation_plan.md:406-456`

**Step 1: Replace Phase 8 content**

Replace the Phase 8 section (lines 406-456) with the expanded version covering both S3 run loading and query curation. Reference the design doc at `docs/plans/2026-02-21-phase8-curation-and-s3-runs-design.md` for architectural rationale.

**Step 2: Commit**

```
docs: update Phase 8 in implementation plan with S3 runs and curation
```

---

## Summary

| Task | Deliverable | Test |
|------|-------------|------|
| 1 | `RunSummary.source` field | Unit test |
| 2 | `S3RunLoader` adapter | Unit tests (mocked boto3) |
| 3 | Repository S3 integration | Unit tests |
| 4 | Run selector `[S3]` badge | Manual (Streamlit) |
| 5 | Results analyzer wiring | Manual (Streamlit ± S3) |
| 6 | Query curator Streamlit page | Manual (Streamlit) |
| 7 | Dataset merge script | Unit tests |
| 8 | Makefile targets | Manual |
| 9 | `.gitignore` update | N/A |
| 10 | Implementation plan update | N/A |

**Dependencies:** Tasks 1 → 2 → 3 → 4/5 (sequential). Tasks 6, 7, 8 are independent of each other and of Tasks 1-5.
