"""Tests for InMemoryRunRepository with S3 loader integration."""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

from eval.app.results.adapters.filesystem_loader import FilesystemRunLoader
from eval.app.results.adapters.repository import InMemoryRunRepository
from eval.app.results.domain.models import RunSummary


def _make_summary(run_id: str, source: str = "local") -> RunSummary:
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
