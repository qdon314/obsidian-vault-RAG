"""Tests for LocalArtifactStore (passthrough, no remote sync)."""

from __future__ import annotations

from pathlib import Path

from rag.adapters.artifacts.local_store import LocalArtifactStore


class TestLocalArtifactStore:
    def test_pull_returns_local_dir(self, tmp_path: Path) -> None:
        store = LocalArtifactStore()
        local_dir = tmp_path / "index"
        local_dir.mkdir()
        result = store.pull("some/key", local_dir)
        assert result == local_dir

    def test_push_is_noop(self, tmp_path: Path) -> None:
        store = LocalArtifactStore()
        local_dir = tmp_path / "index"
        local_dir.mkdir()
        (local_dir / "chunks.jsonl").write_text("{}")
        # Should not raise
        store.push(local_dir, "some/key")

    def test_satisfies_protocol(self) -> None:
        from rag.ports.artifact_store import ArtifactStore

        store = LocalArtifactStore()
        assert isinstance(store, ArtifactStore)
