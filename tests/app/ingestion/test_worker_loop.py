"""Tests for the SQS worker loop."""

from __future__ import annotations

import uuid

import pytest

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.app.ingestion.worker import Worker
from rag.app.ingestion.worker_loop import process_one_message
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
    doc = make_document(doc_id="doc-1", text="Hello world. " * 50)
    store.store_document(doc, corpus_id="c", content_sha256="abc")
    return store


class TestProcessOneMessage:
    def test_success_acks_message(
        self,
        job_store: FakeIngestJobStore,
        raw_store: FakeRawDocumentStore,
    ) -> None:
        from rag.domain.ingestion import DocumentRecord

        worker = Worker(
            job_store=job_store,
            raw_document_store=raw_store,
            chunker=FixedChunker(chunk_size=100, overlap=20),
            embedder=DummyEmbedder(dim=128),
            vector_store=InMemoryVectorStore(),
            chunk_store=FakeChunkStore(),
            worker_id="test-worker",
        )

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

        # Need to also store the document record in job_store
        doc_record = DocumentRecord(
            corpus_id="c",
            doc_id="doc-1",
            source="filesystem",
            uri="/test.md",
            content_sha256="abc",
            s3_raw_key="corpus/c/raw/doc-1.json",
        )
        job_store.upsert_document(doc_record)

        message = {
            "body": {
                "job_id": str(job.job_id),
                "corpus_id": "c",
                "doc_id": "doc-1",
            },
            "receipt_handle": "handle-1",
        }

        acked = []
        nacked = []

        result = process_one_message(
            message=message,
            worker=worker,
            job_store=job_store,
            on_ack=lambda h: acked.append(h),
            on_nack=lambda h: nacked.append(h),
        )

        assert result is True
        assert acked == ["handle-1"]
        assert nacked == []

    def test_does_not_lease_different_doc_task(
        self,
        job_store: FakeIngestJobStore,
        raw_store: FakeRawDocumentStore,
    ) -> None:
        from rag.domain.ingestion import DocumentRecord

        worker = Worker(
            job_store=job_store,
            raw_document_store=raw_store,
            chunker=FixedChunker(chunk_size=100, overlap=20),
            embedder=DummyEmbedder(dim=128),
            vector_store=InMemoryVectorStore(),
            chunk_store=FakeChunkStore(),
            worker_id="test-worker",
        )

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

        # Only doc-2 has a task; incoming message is for doc-1.
        task = IngestTask(
            job_id=job.job_id,
            task_id=uuid.uuid4(),
            doc_id="doc-2",
            status=TaskStatus.PENDING,
        )
        job_store.create_tasks([task])

        job_store.upsert_document(
            DocumentRecord(
                corpus_id="c",
                doc_id="doc-1",
                source="filesystem",
                uri="/test.md",
                content_sha256="abc",
                s3_raw_key="corpus/c/raw/doc-1.json",
            )
        )

        message = {
            "body": {
                "job_id": str(job.job_id),
                "corpus_id": "c",
                "doc_id": "doc-1",
            },
            "receipt_handle": "handle-1",
        }

        acked = []
        nacked = []
        result = process_one_message(
            message=message,
            worker=worker,
            job_store=job_store,
            on_ack=lambda h: acked.append(h),
            on_nack=lambda h: nacked.append(h),
        )

        assert result is True
        assert acked == ["handle-1"]
        assert nacked == []
        assert job_store._tasks[task.task_id].status == TaskStatus.PENDING
