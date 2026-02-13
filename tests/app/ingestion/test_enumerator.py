"""Tests for the ingestion enumerator."""
from __future__ import annotations

import pytest

from rag.app.ingestion.enumerator import Enumerator
from rag.domain.ingestion import JobStatus, TaskStatus
from tests.conftest import make_document
from tests.fakes.fake_ingest_job_store import FakeIngestJobStore


class FakeRawDocumentStore:
    """Fake for RawDocumentStore protocol."""

    def __init__(self) -> None:
        self.stored: dict[str, object] = {}

    def store_document(self, doc, *, corpus_id: str, content_sha256: str) -> str:
        key = f"corpus/{corpus_id}/raw/{doc.doc_id}.json"
        self.stored[key] = doc
        return key

    def get_document(self, key: str):
        return self.stored.get(key)


class FakeTaskQueue:
    """Fake for TaskQueue protocol."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    def send(self, message: dict) -> None:
        self.sent.append(message)

    def send_batch(self, messages) -> None:
        self.sent.extend(messages)

    def receive(self, **kw):
        return []

    def ack(self, handle: str) -> None:
        pass

    def nack(self, handle: str, **kw) -> None:
        pass


@pytest.fixture
def job_store() -> FakeIngestJobStore:
    s = FakeIngestJobStore()
    s.ensure_schema()
    return s


@pytest.fixture
def raw_store() -> FakeRawDocumentStore:
    return FakeRawDocumentStore()


@pytest.fixture
def queue() -> FakeTaskQueue:
    return FakeTaskQueue()


class TestEnumerator:
    def test_enumerate_creates_job_and_tasks(
        self,
        job_store: FakeIngestJobStore,
        raw_store: FakeRawDocumentStore,
        queue: FakeTaskQueue,
    ) -> None:
        docs = [
            make_document(doc_id=f"doc-{i}", text=f"content {i}")
            for i in range(3)
        ]
        enum = Enumerator(
            job_store=job_store,
            raw_document_store=raw_store,
            task_queue=queue,
        )
        job = enum.enumerate(
            docs=docs,
            corpus_id="test_corpus",
            index_id="test_index",
            chunking_strategy="fixed",
            embedder_model="dummy",
            qdrant_collection="test",
        )

        # Job should be created and running
        assert job.status == JobStatus.RUNNING
        got = job_store.get_job(job.job_id)
        assert got is not None
        assert got.status == JobStatus.RUNNING

        # All docs should be stored in S3
        assert len(raw_store.stored) == 3

        # All docs should be enqueued
        assert len(queue.sent) == 3

        # Tasks should be created in DB
        counts = job_store.get_task_counts(job.job_id)
        assert counts[TaskStatus.PENDING] == 3

    def test_enumerate_empty_docs(
        self,
        job_store: FakeIngestJobStore,
        raw_store: FakeRawDocumentStore,
        queue: FakeTaskQueue,
    ) -> None:
        enum = Enumerator(
            job_store=job_store,
            raw_document_store=raw_store,
            task_queue=queue,
        )
        job = enum.enumerate(
            docs=[],
            corpus_id="c",
            index_id="i",
            chunking_strategy="fixed",
            embedder_model="dummy",
            qdrant_collection="q",
        )
        assert job.status == JobStatus.COMPLETED
        assert len(queue.sent) == 0
