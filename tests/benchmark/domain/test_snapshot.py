"""Tests for corpus snapshot utilities."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.domain.snapshot import compute_snapshot_id, verify_snapshot


@dataclass(frozen=True)
class _FakeDoc:
    """Minimal stand-in for rag.domain.models.Document."""

    doc_id: str
    text: str


class TestComputeSnapshotId:
    def test_deterministic(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="hello"), _FakeDoc(doc_id="d2", text="world")]
        assert compute_snapshot_id(docs) == compute_snapshot_id(docs)

    def test_order_independent(self) -> None:
        """Sorting ensures the same corpus in different order gives the same ID."""
        docs_a = [_FakeDoc(doc_id="d1", text="a"), _FakeDoc(doc_id="d2", text="b")]
        docs_b = [_FakeDoc(doc_id="d2", text="b"), _FakeDoc(doc_id="d1", text="a")]
        assert compute_snapshot_id(docs_a) == compute_snapshot_id(docs_b)

    def test_content_sensitive(self) -> None:
        docs_a = [_FakeDoc(doc_id="d1", text="version_1")]
        docs_b = [_FakeDoc(doc_id="d1", text="version_2")]
        assert compute_snapshot_id(docs_a) != compute_snapshot_id(docs_b)

    def test_empty_corpus(self) -> None:
        result = compute_snapshot_id([])
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 hex

    def test_returns_hex_string(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="x")]
        result = compute_snapshot_id(docs)
        assert len(result) == 64
        int(result, 16)  # should not raise


class TestVerifySnapshot:
    def test_match(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="hello")]
        snap_id = compute_snapshot_id(docs)
        assert verify_snapshot(docs, snap_id) is True

    def test_mismatch(self) -> None:
        docs = [_FakeDoc(doc_id="d1", text="hello")]
        assert verify_snapshot(docs, "0" * 64) is False
