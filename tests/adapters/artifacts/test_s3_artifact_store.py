"""Tests for S3ArtifactStore.

These tests mock boto3 to avoid real AWS calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from rag.adapters.artifacts.s3_store import S3ArtifactStore


@pytest.fixture
def mock_s3_client() -> MagicMock:
    return MagicMock()


@pytest.fixture
def store(mock_s3_client: MagicMock) -> S3ArtifactStore:
    return S3ArtifactStore(bucket="test-bucket", client=mock_s3_client)


class TestS3ArtifactStorePush:
    def test_push_uploads_all_files(
        self, store: S3ArtifactStore, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        local_dir = tmp_path / "index"
        local_dir.mkdir()
        (local_dir / "chunks.jsonl").write_text('{"test": true}')
        (local_dir / "manifest.json").write_text('{"name": "test"}')

        store.push(local_dir, "indexes/my_index")

        uploaded_keys = {
            c.kwargs["Key"] if "Key" in c.kwargs else c.args[1]
            for c in mock_s3_client.upload_file.call_args_list
        }
        assert "indexes/my_index/chunks.jsonl" in uploaded_keys
        assert "indexes/my_index/manifest.json" in uploaded_keys

    def test_push_without_manifest_raises(
        self, store: S3ArtifactStore, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        local_dir = tmp_path / "no_manifest"
        local_dir.mkdir()
        (local_dir / "chunks.jsonl").write_text("{}")

        with pytest.raises(FileNotFoundError, match="manifest.json not found"):
            store.push(local_dir, "indexes/bad")

        mock_s3_client.upload_file.assert_not_called()


class TestS3ArtifactStorePull:
    def test_pull_downloads_to_local_dir(
        self, store: S3ArtifactStore, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        local_dir = tmp_path / "index"
        local_dir.mkdir()

        mock_s3_client.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "indexes/test/chunks.jsonl"},
                {"Key": "indexes/test/manifest.json"},
            ]
        }

        result = store.pull("indexes/test", local_dir)
        assert result == local_dir
        assert mock_s3_client.download_file.call_count == 2

    def test_pull_empty_prefix_creates_no_files(
        self, store: S3ArtifactStore, mock_s3_client: MagicMock, tmp_path: Path
    ) -> None:
        local_dir = tmp_path / "index"
        local_dir.mkdir()

        mock_s3_client.list_objects_v2.return_value = {}

        result = store.pull("indexes/empty", local_dir)
        assert result == local_dir
        mock_s3_client.download_file.assert_not_called()


class TestS3ArtifactStoreProtocol:
    def test_satisfies_protocol(self, mock_s3_client: MagicMock) -> None:
        from rag.ports.artifact_store import ArtifactStore

        store = S3ArtifactStore(bucket="test", client=mock_s3_client)
        assert isinstance(store, ArtifactStore)
