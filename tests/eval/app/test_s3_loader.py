"""Tests for S3RunLoader — eval run discovery and loading from S3."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from eval.app.results.adapters.s3_loader import S3RunLoader


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    d = tmp_path / ".s3-cache"
    d.mkdir()
    return d


def _mock_paginator(pages: list[dict]) -> MagicMock:
    """Build a mock paginator whose .paginate() iterates over *pages*."""
    paginator = MagicMock()
    paginator.paginate.return_value = iter(pages)
    return paginator


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
        # Mock paginator to return two "run" prefixes
        mock_s3_client.get_paginator.return_value = _mock_paginator([{
            "CommonPrefixes": [
                {"Prefix": "eval/runs/run_2026_02_20T19-49/"},
                {"Prefix": "eval/runs/run_2026_02_20T21-05/"},
            ],
        }])

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
        mock_s3_client.get_paginator.return_value = _mock_paginator([{
            "CommonPrefixes": [
                {"Prefix": "eval/runs/run_2026_02_20T19-49/"},
            ],
        }])
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

    def test_discovers_iso_formatted_run_names(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        """Run dirs needn't start with 'run_' — ISO timestamps work too."""
        mock_s3_client.get_paginator.return_value = _mock_paginator([{
            "CommonPrefixes": [
                {"Prefix": "runs/2026-02-14T21-05-06/"},
            ],
        }])

        def fake_download(Bucket, Key, Filename):
            Path(Filename).parent.mkdir(parents=True, exist_ok=True)
            Path(Filename).write_text(_make_metrics_json())

        mock_s3_client.download_file.side_effect = fake_download

        loader = S3RunLoader(
            bucket="test-bucket",
            prefix="runs",
            cache_dir=cache_dir,
            client=mock_s3_client,
        )

        summaries = loader.discover_runs()
        assert len(summaries) == 1
        assert summaries[0].run_id == "2026-02-14T21-05-06"
        assert summaries[0].source == "s3"

    def test_empty_bucket_returns_empty_list(
        self, mock_s3_client: MagicMock, cache_dir: Path
    ) -> None:
        mock_s3_client.get_paginator.return_value = _mock_paginator([{}])

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
        mock_s3_client.get_paginator.return_value = _mock_paginator([{
            "Contents": [
                {"Key": "eval/runs/run_2026_02_20T19-49/metrics.json"},
                {"Key": "eval/runs/run_2026_02_20T19-49/results.jsonl"},
            ],
        }])

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
        mock_s3_client.get_paginator.assert_not_called()
        mock_s3_client.download_file.assert_not_called()
