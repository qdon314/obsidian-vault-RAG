"""Integration smoke test for distributed ingestion.

Uses fake adapters to avoid AWS dependencies.
"""

import uuid

import pytest

from rag.app.ingestion.enumerator import Enumerator
from rag.app.ingestion.worker import Worker
from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)
from rag.domain.models import Chunk, Document
from tests.fakes.fake_ingest_job_store import FakeIngestJobStore


class FakeRawDocumentStore:
    """Fake S3 store for testing."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self._objects = objects or {}
        self._stored: dict[str, Document] = {}

    def store_document(self, doc: Document, *, corpus_id: str, content_sha256: str) -> str:
        """Store a document and return its key."""
        key = f"{corpus_id}/{doc.doc_id}.json"
        self._stored[key] = doc
        return key

    def get_document(self, key: str) -> Document:
        """Retrieve a document by key."""
        if key not in self._stored:
            raise FileNotFoundError(f"Document not found: {key}")
        return self._stored[key]


class FakeTaskQueue:
    """Fake SQS queue for testing."""

    def __init__(self):
        self._messages: list[dict] = []

    def send(self, message: dict[str, str]) -> None:
        """Enqueue a single message."""
        self._messages.append(
            {
                "body": message,
                "receipt_handle": f"receipt-{len(self._messages)}",
            }
        )

    def send_batch(self, messages: list[dict[str, str]]) -> None:
        """Enqueue a batch of messages."""
        for msg in messages:
            self.send(msg)

    def receive(self, *, max_messages: int = 1, wait_seconds: int = 20) -> list[dict]:
        """Receive messages from the queue."""
        result = self._messages[:max_messages]
        self._messages = self._messages[max_messages:]
        return result

    def ack(self, receipt_handle: str) -> None:
        """Delete a message after processing."""
        self._messages = [m for m in self._messages if m["receipt_handle"] != receipt_handle]

    def nack(self, receipt_handle: str, *, visibility_timeout: int = 0) -> None:
        """Return a message to the queue."""
        pass


class FakeVectorStore:
    """Fake vector store for testing."""

    def __init__(self):
        self.upserted: list[tuple] = []

    def upsert(self, chunks: list, vectors: list) -> None:
        """Store chunks and vectors."""
        self.upserted.extend(zip(chunks, vectors, strict=True))


class FakeChunkStore:
    """Fake chunk store for testing."""

    def __init__(self):
        self.stored: list[list[Chunk]] = []

    def store_chunks(self, chunks: list[Chunk]) -> None:
        """Store chunks."""
        self.stored.append(chunks)


class FakeChunker:
    """Fake chunker for testing."""

    def chunk(self, document: Document) -> list[Chunk]:
        """Return single chunk per document."""
        return [
            Chunk(
                chunk_id=f"{document.doc_id}:fixed:0:0-{len(document.text)}",
                doc_id=document.doc_id,
                text=document.text[:100],
                chunk_index=0,
                start_char=0,
                end_char=min(100, len(document.text)),
                metadata={},
            )
        ]


class FakeEmbedder:
    """Fake embedder for testing."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Return random vectors."""
        import random

        return [[random.random() for _ in range(10)] for _ in texts]


@pytest.fixture
def setup():
    """Create fake infrastructure for testing."""
    job_store = FakeIngestJobStore()
    raw_store = FakeRawDocumentStore()
    task_queue = FakeTaskQueue()
    vector_store = FakeVectorStore()
    chunk_store = FakeChunkStore()
    chunker = FakeChunker()
    embedder = FakeEmbedder()

    return {
        "job_store": job_store,
        "raw_store": raw_store,
        "task_queue": task_queue,
        "vector_store": vector_store,
        "chunk_store": chunk_store,
        "chunker": chunker,
        "embedder": embedder,
    }


class TestDistributedIngestion:
    """End-to-end smoke test for distributed ingestion."""

    def test_enumerator_creates_job_and_tasks(self, setup):
        """Enumerator should create job and enqueue tasks."""
        docs = [
            Document(
                doc_id="doc1",
                text="This is document 1",
                source="test",
                uri="test://doc1",
                metadata={},
            ),
            Document(
                doc_id="doc2",
                text="This is document 2",
                source="test",
                uri="test://doc2",
                metadata={},
            ),
        ]

        enumerator = Enumerator(
            job_store=setup["job_store"],
            raw_document_store=setup["raw_store"],
            task_queue=setup["task_queue"],
        )

        job = enumerator.enumerate(
            docs=docs,
            corpus_id="test-corpus",
            index_id="test-index",
            chunking_strategy="fixed",
            embedder_model="test-model",
            qdrant_collection="test-collection",
        )

        # Verify job created
        assert job is not None
        assert job.corpus_id == "test-corpus"
        assert job.status == JobStatus.RUNNING

        # Verify tasks created
        counts = setup["job_store"].get_task_counts(job.job_id)
        assert sum(counts.values()) == 2  # 2 documents

        # Verify messages enqueued
        assert len(setup["task_queue"]._messages) == 2

    def test_worker_processes_task(self, setup):
        """Worker should process a task end-to-end."""
        # Create a document
        doc = Document(
            doc_id="doc1",
            text="This is document 1 content",
            source="test",
            uri="test://doc1",
            metadata={},
        )

        # Store the document
        s3_key = setup["raw_store"].store_document(
            doc, corpus_id="test-corpus", content_sha256="abc123"
        )

        # Create a job and task
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="test-corpus",
            index_id="test-index",
            chunking_strategy="fixed",
            embedder_model="test-model",
            qdrant_collection="test-collection",
            status=JobStatus.RUNNING,
        )
        setup["job_store"].create_job(job)

        task = IngestTask(
            job_id=job.job_id,
            task_id=uuid.uuid4(),
            doc_id=doc.doc_id,
            status=TaskStatus.PENDING,
        )
        setup["job_store"].create_tasks([task])

        # Create document record
        doc_record = DocumentRecord(
            corpus_id="test-corpus",
            doc_id=doc.doc_id,
            source="test",
            uri="test://doc1",
            content_sha256="abc123",
            s3_raw_key=s3_key,
        )
        setup["job_store"].upsert_document(doc_record)

        worker = Worker(
            job_store=setup["job_store"],
            raw_document_store=setup["raw_store"],
            chunker=setup["chunker"],
            embedder=setup["embedder"],
            vector_store=setup["vector_store"],
            chunk_store=setup["chunk_store"],
            worker_id="test-worker",
        )

        # Process the task
        success = worker.process_task(
            task=task,
            s3_raw_key=s3_key,
            corpus_id="test-corpus",
        )

        assert success is True
        assert len(setup["vector_store"].upserted) == 1
        assert len(setup["chunk_store"].stored) == 1

    def test_full_pipeline(self, setup):
        """Full pipeline: enumerator -> queue -> worker."""
        # Create documents
        docs = [
            Document(
                doc_id="doc1",
                text="This is document 1",
                source="test",
                uri="test://doc1",
                metadata={},
            ),
            Document(
                doc_id="doc2",
                text="This is document 2",
                source="test",
                uri="test://doc2",
                metadata={},
            ),
        ]

        # Enumerator creates job and tasks
        enumerator = Enumerator(
            job_store=setup["job_store"],
            raw_document_store=setup["raw_store"],
            task_queue=setup["task_queue"],
        )

        job = enumerator.enumerate(
            docs=docs,
            corpus_id="test-corpus",
            index_id="test-index",
            chunking_strategy="fixed",
            embedder_model="test-model",
            qdrant_collection="test-collection",
        )

        worker = Worker(
            job_store=setup["job_store"],
            raw_document_store=setup["raw_store"],
            chunker=setup["chunker"],
            embedder=setup["embedder"],
            vector_store=setup["vector_store"],
            chunk_store=setup["chunk_store"],
            worker_id="test-worker",
        )

        # Process all tasks from queue
        processed = 0
        while True:
            messages = setup["task_queue"].receive(max_messages=1)
            if not messages:
                break

            for msg in messages:
                body = msg["body"]
                doc_id = body["doc_id"]

                # Get the document record
                doc_record = setup["job_store"].get_document("test-corpus", doc_id)
                assert doc_record is not None

                # Acquire the task
                task = setup["job_store"].acquire_task(
                    job.job_id, lease_owner="test-worker", lease_duration_s=300
                )
                if task is None:
                    continue

                # Process the task
                success = worker.process_task(
                    task=task,
                    s3_raw_key=doc_record.s3_raw_key,
                    corpus_id="test-corpus",
                )

                if success:
                    setup["task_queue"].ack(msg["receipt_handle"])
                    processed += 1

        # Verify all processed
        assert processed == 2
        assert len(setup["vector_store"].upserted) == 2

    def test_lease_prevents_duplicate_processing(self, setup):
        """Task lease should prevent duplicate processing."""
        # Create a job and task
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="test-corpus",
            index_id="test-index",
            chunking_strategy="fixed",
            embedder_model="test-model",
            qdrant_collection="test-collection",
            status=JobStatus.RUNNING,
        )
        setup["job_store"].create_job(job)

        task = IngestTask(
            job_id=job.job_id,
            task_id=uuid.uuid4(),
            doc_id="doc1",
            status=TaskStatus.PENDING,
        )
        setup["job_store"].create_tasks([task])

        # First worker acquires task
        leased1 = setup["job_store"].acquire_task(
            job.job_id, lease_owner="worker-1", lease_duration_s=300
        )
        assert leased1 is not None
        assert leased1.lease_owner == "worker-1"
        assert leased1.status == TaskStatus.RUNNING

        # Second worker should not be able to acquire same task
        leased2 = setup["job_store"].acquire_task(
            job.job_id, lease_owner="worker-2", lease_duration_s=300
        )
        # Task is already leased, so should get None
        assert leased2 is None
