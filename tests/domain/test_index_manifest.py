"""Tests for IndexManifest domain object."""

from __future__ import annotations

import json
from pathlib import Path

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
