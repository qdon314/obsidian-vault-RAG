"""Tests for IndexManifest domain object."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import MagicMock, patch

from rag.domain.errors import IndexIncompatibleError, RagAppError
from rag.domain.index_manifest import IndexManifest


class TestIndexManifest:
    def test_create_populates_timestamp_and_git_sha(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data/vault",
            doc_count=10,
            chunk_count=50,
        )
        assert m.index_name == "test"
        assert m.created_at  # non-empty
        assert m.git_sha  # non-empty (either real SHA or "unknown")
        assert m.doc_count == 10
        assert m.chunk_count == 50

    def test_to_dict_roundtrip(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
            chunking={"backend": "fixed", "chunk_size": 800},
            embedding={"backend": "openai", "model": "text-embedding-3-large"},
        )
        d = m.to_dict()
        m2 = IndexManifest.from_dict(d)
        assert m == m2

    def test_save_and_load(self, tmp_path: Path) -> None:
        m = IndexManifest.create(
            index_name="roundtrip",
            corpus="/test",
            doc_count=3,
            chunk_count=15,
        )
        m.save(tmp_path)
        loaded = IndexManifest.load(tmp_path)
        assert loaded == m

    def test_save_creates_manifest_json(self, tmp_path: Path) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/test",
            doc_count=1,
            chunk_count=5,
        )
        path = m.save(tmp_path)
        assert path == tmp_path / "manifest.json"
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["index_name"] == "test"
        assert "git_sha" in data
        assert data["index_id"].startswith("test_")

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = {
            "index_name": "test",
            "index_id": "test_unknown_unknown_20260101T000000Z",
            "created_at": "2026-01-01T00:00:00",
            "git_sha": "abc123",
            "corpus": "/data",
            "doc_count": 1,
            "chunk_count": 5,
            "extra_field": "should be ignored",
        }
        m = IndexManifest.from_dict(d)
        assert m.index_name == "test"

    def test_create_populates_build_id(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
        )
        assert m.build_id  # non-empty UUID string
        assert len(m.build_id) == 32  # hex UUID without dashes

    def test_create_populates_status_default(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
        )
        assert m.status == "complete"

    def test_create_accepts_status_override(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=0,
            chunk_count=0,
            status="failed",
        )
        assert m.status == "failed"

    def test_create_populates_git_dirty(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
        )
        assert isinstance(m.git_dirty, bool)

    def test_create_accepts_build_duration(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
            build_duration_s=12.5,
        )
        assert m.build_duration_s == 12.5

    def test_roundtrip_preserves_new_fields(self) -> None:
        m = IndexManifest.create(
            index_name="test",
            corpus="/data",
            doc_count=5,
            chunk_count=20,
            build_duration_s=10.0,
        )
        d = m.to_dict()
        m2 = IndexManifest.from_dict(d)
        assert m2.build_id == m.build_id
        assert m2.status == m.status
        assert m2.git_dirty == m.git_dirty
        assert m2.build_duration_s == m.build_duration_s

    def test_index_incompatible_error_is_rag_app_error(self) -> None:
        err = IndexIncompatibleError("model mismatch")
        assert isinstance(err, RagAppError)
        assert str(err) == "model mismatch"


class TestLatestFromS3:
    """Unit tests for IndexManifest.latest_from_s3 using mocked boto3."""

    _MANIFEST_DATA: ClassVar[dict[str, Any]] = {
        "index_name": "regulatory",
        "index_id": "regulatory_unknown_unknown_20260220T210549Z",
        "created_at": "2026-02-20T21:05:49+00:00",
        "git_sha": "abc123",
        "corpus": "regulatory",
        "doc_count": 100,
        "chunk_count": 500,
    }

    def _mock_s3(self, common_prefixes: list[dict[str, str]]) -> MagicMock:
        """Build a mock boto3 S3 client with paginated list + get_object."""
        mock_s3 = MagicMock()
        mock_paginator = MagicMock()
        mock_s3.get_paginator.return_value = mock_paginator
        mock_paginator.paginate.return_value = [
            {"CommonPrefixes": common_prefixes},
        ]
        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(self._MANIFEST_DATA).encode()),
        }
        return mock_s3

    def test_returns_latest_by_lexicographic_sort(self) -> None:
        mock_s3 = self._mock_s3(
            [
                {"Prefix": "regulatory/manifests/regulatory_unknown_unknown_20260101T000000Z/"},
                {"Prefix": "regulatory/manifests/regulatory_unknown_unknown_20260220T210549Z/"},
            ]
        )

        with patch("boto3.client", return_value=mock_s3):
            result = IndexManifest.latest_from_s3(
                bucket="obsidian-rag-corpus",
                corpus_prefix="regulatory",
            )

        assert result is not None
        assert result.index_id == "regulatory_unknown_unknown_20260220T210549Z"
        mock_s3.get_object.assert_called_once_with(
            Bucket="obsidian-rag-corpus",
            Key="regulatory/manifests/regulatory_unknown_unknown_20260220T210549Z/manifest.json",
        )

    def test_returns_none_when_no_manifests(self) -> None:
        mock_s3 = self._mock_s3([])

        with patch("boto3.client", return_value=mock_s3):
            result = IndexManifest.latest_from_s3(
                bucket="obsidian-rag-corpus",
                corpus_prefix="regulatory",
            )

        assert result is None

    def test_returns_none_on_s3_exception(self) -> None:
        with patch("boto3.client", side_effect=Exception("no credentials")):
            result = IndexManifest.latest_from_s3(
                bucket="obsidian-rag-corpus",
                corpus_prefix="regulatory",
            )

        assert result is None

    def test_empty_corpus_prefix_builds_correct_listing_prefix(self) -> None:
        mock_s3 = self._mock_s3(
            [
                {"Prefix": "manifests/regulatory_unknown_unknown_20260220T210549Z/"},
            ]
        )

        with patch("boto3.client", return_value=mock_s3):
            result = IndexManifest.latest_from_s3(
                bucket="obsidian-rag-corpus",
                corpus_prefix="",
            )

        assert result is not None
        mock_paginator = mock_s3.get_paginator.return_value
        mock_paginator.paginate.assert_called_once_with(
            Bucket="obsidian-rag-corpus",
            Prefix="manifests/",
            Delimiter="/",
        )
        mock_s3.get_object.assert_called_once_with(
            Bucket="obsidian-rag-corpus",
            Key="manifests/regulatory_unknown_unknown_20260220T210549Z/manifest.json",
        )
