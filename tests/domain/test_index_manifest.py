"""Tests for IndexManifest domain object."""

from __future__ import annotations

import json
from pathlib import Path

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

    def test_from_dict_ignores_extra_keys(self) -> None:
        d = {
            "index_name": "test",
            "created_at": "2026-01-01T00:00:00",
            "git_sha": "abc123",
            "corpus": "/data",
            "doc_count": 1,
            "chunk_count": 5,
            "extra_field": "should be ignored",
        }
        m = IndexManifest.from_dict(d)
        assert m.index_name == "test"
