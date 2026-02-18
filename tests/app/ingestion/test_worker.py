"""Tests for the ingestion worker."""

from __future__ import annotations

import uuid

import pytest

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.app.ingestion.worker import Worker
from rag.domain.ingestion import IngestJob, IngestTask, JobStatus, TaskStatus
from tests.app.ingestion.test_enumerator import FakeRawDocumentStore
from tests.conftest import FakeChunkStore, make_document
from tests.fakes.fake_ingest_job_store import FakeIngestJobStore


@pytest.fixture
def job_store() -> FakeIngestJobStore:
    s = FakeIngestJobStore()
    s.ensure_schema()
    return s


@pytest.fixture
def raw_store() -> FakeRawDocumentStore:
    store = FakeRawDocumentStore()
    # Pre-populate with a raw document
    doc = make_document(doc_id="doc-1", text="Hello world. " * 50)
    store.store_document(doc, corpus_id="c", content_sha256="abc")
    return store


@pytest.fixture
def worker(
    job_store: FakeIngestJobStore,
    raw_store: FakeRawDocumentStore,
) -> Worker:
    return Worker(
        job_store=job_store,
        raw_document_store=raw_store,
        chunker=FixedChunker(chunk_size=100, overlap=20),
        embedder=DummyEmbedder(dim=128),
        vector_store=InMemoryVectorStore(),
        chunk_store=FakeChunkStore(),
        worker_id="test-worker-1",
    )


class TestWorkerProcessTask:
    def test_processes_document_successfully(
        self,
        worker: Worker,
        job_store: FakeIngestJobStore,
        raw_store: FakeRawDocumentStore,
    ) -> None:
        # Setup: create job and task
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="c",
            index_id="i",
            chunking_strategy="fixed",
            embedder_model="dummy",
            qdrant_collection="q",
            status=JobStatus.RUNNING,
        )
        job_store.create_job(job)

        task = IngestTask(
            job_id=job.job_id,
            task_id=uuid.uuid4(),
            doc_id="doc-1",
            status=TaskStatus.PENDING,
        )
        job_store.create_tasks([task])

        # Get the S3 key for our pre-populated doc
        s3_key = "corpus/c/raw/doc-1.json"

        # Process the task
        result = worker.process_task(
            task=task,
            s3_raw_key=s3_key,
            corpus_id="c",
        )

        assert result is True

    def test_marks_task_failed_on_error(
        self,
        worker: Worker,
        job_store: FakeIngestJobStore,
    ) -> None:
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="c",
            index_id="i",
            chunking_strategy="fixed",
            embedder_model="dummy",
            qdrant_collection="q",
            status=JobStatus.RUNNING,
        )
        job_store.create_job(job)

        task = IngestTask(
            job_id=job.job_id,
            task_id=uuid.uuid4(),
            doc_id="doc-missing",
            status=TaskStatus.RUNNING,
        )
        job_store.create_tasks([task])

        # Process with a missing doc key — should fail gracefully
        result = worker.process_task(
            task=task,
            s3_raw_key="corpus/c/raw/nonexistent.json",
            corpus_id="c",
        )
        assert result is False
        stored = job_store._tasks[task.task_id]
        assert stored.status == TaskStatus.RETRYABLE
        assert "not found" in (stored.last_error or "").lower()
