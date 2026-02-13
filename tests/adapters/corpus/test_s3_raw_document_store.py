"""Tests for S3RawDocumentStore — uses unittest.mock for boto3."""
from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
from tests.conftest import make_document


@pytest.fixture
def mock_s3() -> MagicMock:
    return MagicMock()


@pytest.fixture
def store(mock_s3: MagicMock) -> S3RawDocumentStore:
    return S3RawDocumentStore._for_test(
        bucket="test-bucket",
        prefix="corpus/test_corpus/raw",
        s3_client=mock_s3,
    )


class TestStoreDocument:
    def test_returns_s3_key(self, store: S3RawDocumentStore, mock_s3: MagicMock) -> None:
        doc = make_document(doc_id="abc123")
        key = store.store_document(doc, corpus_id="test_corpus", content_sha256="dead" * 16)
        assert key.startswith("corpus/test_corpus/raw/")
        assert key.endswith(".json")
        mock_s3.put_object.assert_called_once()

    def test_s3_body_is_valid_json(self, store: S3RawDocumentStore, mock_s3: MagicMock) -> None:
        doc = make_document(doc_id="abc123", text="hello world")
        store.store_document(doc, corpus_id="test_corpus", content_sha256="dead" * 16)

        call_kwargs = mock_s3.put_object.call_args.kwargs
        body = json.loads(call_kwargs["Body"])
        assert body["doc_id"] == "abc123"
        assert body["text"] == "hello world"
        assert body["content_sha256"] == "dead" * 16

    def test_key_uses_hash_prefix_for_distribution(
        self, store: S3RawDocumentStore, mock_s3: MagicMock
    ) -> None:
        doc = make_document(doc_id="abc123")
        key = store.store_document(doc, corpus_id="c", content_sha256="f" * 64)
        # Key should contain a hash prefix subdirectory
        parts = key.split("/")
        # prefix / hash_prefix / filename
        assert len(parts) >= 3


class TestGetDocument:
    def test_round_trip(self, store: S3RawDocumentStore, mock_s3: MagicMock) -> None:
        doc = make_document(doc_id="abc123", text="hello", source="filesystem", uri="/x.md")
        payload = json.dumps({
            "doc_id": doc.doc_id,
            "text": doc.text,
            "source": doc.source,
            "uri": doc.uri,
            "metadata": dict(doc.metadata),
            "corpus_id": "c",
            "content_sha256": "f" * 64,
        }).encode("utf-8")

        mock_s3.get_object.return_value = {
            "Body": MagicMock(read=MagicMock(return_value=payload))
        }

        result = store.get_document("corpus/c/raw/ab/abc.json")
        assert result.doc_id == "abc123"
        assert result.text == "hello"
