# Phase 3: Distributed Ingestion — Corpus-of-Record to S3 + DB-Backed Jobs

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Move the corpus-of-record to S3, add Postgres-backed ingestion job tracking, and SQS-driven parallel worker processing — so ingestion scales horizontally beyond a single machine.

**Architecture:** Extends the existing hexagonal architecture with three new ports (`IngestJobStore`, `RawDocumentStore`, `TaskQueue`) and their adapters (Postgres, S3, SQS). An enumerator writes raw docs to S3, records them in Postgres, and enqueues SQS messages. Workers pull messages, chunk/embed, and write to the existing S3ChunkStore + Qdrant pipeline. Job/task state in Postgres provides resumability and idempotency.

**Tech Stack:** boto3 (S3, SQS), psycopg2 (Postgres), existing rag adapters, pytest + moto for testing

**Prerequisites (already implemented in Phases 0-2):**
- `ChunkStore` port + `S3ChunkStore` adapter (S3 JSONL shards + Postgres `chunk_index` table)
- Thin Qdrant payloads + `HydratingRetriever`
- Dual-write in `pipeline.py` (`chunk_store` parameter)
- `IndexManifest` with `index_id`
- ECS + Fargate infrastructure with S3 artifacts bucket

---

## Task 1: Ingestion Domain Models

**Files:**
- Create: `src/rag/domain/ingestion.py`
- Test: `tests/domain/test_ingestion_models.py`

These frozen dataclasses model the control-plane state for distributed ingestion: jobs, tasks, and document records. They are pure domain objects with no infrastructure dependencies.

**Step 1: Write the failing test**

```python
# tests/domain/test_ingestion_models.py
"""Tests for ingestion domain models."""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)


class TestJobStatus:
    def test_terminal_states(self) -> None:
        assert JobStatus.COMPLETED.is_terminal()
        assert JobStatus.FAILED.is_terminal()
        assert JobStatus.CANCELLED.is_terminal()
        assert not JobStatus.CREATED.is_terminal()
        assert not JobStatus.RUNNING.is_terminal()


class TestTaskStatus:
    def test_terminal_states(self) -> None:
        assert TaskStatus.SUCCEEDED.is_terminal()
        assert TaskStatus.FAILED.is_terminal()
        assert not TaskStatus.PENDING.is_terminal()
        assert not TaskStatus.RUNNING.is_terminal()
        assert not TaskStatus.RETRYABLE.is_terminal()


class TestDocumentRecord:
    def test_create_minimal(self) -> None:
        rec = DocumentRecord(
            corpus_id="test_corpus",
            doc_id="abc123",
            source="filesystem",
            uri="/docs/test.md",
            content_sha256="deadbeef" * 8,
            s3_raw_key="corpus/test_corpus/raw/ab/abc123.json",
        )
        assert rec.corpus_id == "test_corpus"
        assert rec.doc_id == "abc123"
        assert rec.metadata == {}

    def test_frozen(self) -> None:
        rec = DocumentRecord(
            corpus_id="c",
            doc_id="d",
            source="filesystem",
            uri="/x",
            content_sha256="a" * 64,
            s3_raw_key="k",
        )
        try:
            rec.corpus_id = "other"  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestIngestJob:
    def test_create(self) -> None:
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="regulations_v1",
            index_id="regulations_v1__obsidian_structural_v1__text-embedding-3-large__2026-02-12",
            chunking_strategy="obsidian_structural",
            embedder_model="text-embedding-3-large",
            qdrant_collection="regulations",
            status=JobStatus.CREATED,
        )
        assert job.status == JobStatus.CREATED
        assert not job.status.is_terminal()

    def test_frozen(self) -> None:
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="c",
            index_id="i",
            chunking_strategy="fixed",
            embedder_model="m",
            qdrant_collection="q",
            status=JobStatus.CREATED,
        )
        try:
            job.status = JobStatus.RUNNING  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass


class TestIngestTask:
    def test_create_pending(self) -> None:
        task = IngestTask(
            job_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            doc_id="doc_abc",
            status=TaskStatus.PENDING,
        )
        assert task.attempt == 0
        assert task.lease_owner is None
        assert task.lease_expires_at is None
        assert task.last_error is None

    def test_frozen(self) -> None:
        task = IngestTask(
            job_id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            doc_id="d",
            status=TaskStatus.PENDING,
        )
        try:
            task.status = TaskStatus.RUNNING  # type: ignore[misc]
            assert False, "Should be frozen"
        except AttributeError:
            pass
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/domain/test_ingestion_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'rag.domain.ingestion'`

**Step 3: Write minimal implementation**

```python
# src/rag/domain/ingestion.py
"""Domain models for distributed ingestion orchestration.

These models represent the control-plane state for parallel, resumable
ingestion jobs.  They are pure domain objects — no infrastructure
dependencies.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum


class JobStatus(Enum):
    """Lifecycle states for an ingestion job."""

    CREATED = "CREATED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    def is_terminal(self) -> bool:
        return self in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED)


class TaskStatus(Enum):
    """Lifecycle states for a single-document ingestion task."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRYABLE = "RETRYABLE"

    def is_terminal(self) -> bool:
        return self in (TaskStatus.SUCCEEDED, TaskStatus.FAILED)


@dataclass(frozen=True, slots=True)
class DocumentRecord:
    """A raw document registered in the corpus-of-record.

    Tracks the S3 location, content hash (for idempotency), and source
    provenance of each document.
    """

    corpus_id: str
    doc_id: str
    source: str  # e.g. "filesystem", "web", "github"
    uri: str
    content_sha256: str
    s3_raw_key: str
    metadata: dict[str, object] = field(default_factory=dict)
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))


@dataclass(frozen=True, slots=True)
class IngestJob:
    """Top-level ingestion job that tracks a full corpus indexing run.

    One job produces many tasks (one per document).
    """

    job_id: uuid.UUID
    corpus_id: str
    index_id: str
    chunking_strategy: str
    embedder_model: str
    qdrant_collection: str
    status: JobStatus
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    stats: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IngestTask:
    """A single-document task within an ingestion job.

    Workers acquire a lease, process the document, and mark the task
    succeeded or retryable.
    """

    job_id: uuid.UUID
    task_id: uuid.UUID
    doc_id: str
    status: TaskStatus
    attempt: int = 0
    lease_owner: str | None = None
    lease_expires_at: datetime | None = None
    last_error: str | None = None
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/domain/test_ingestion_models.py -v`
Expected: All PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check src/rag/domain/ingestion.py tests/domain/test_ingestion_models.py`

**Step 6: Suggested commit**

```
feat(domain): add ingestion orchestration models

Files: src/rag/domain/ingestion.py, tests/domain/test_ingestion_models.py
```

---

## Task 2: IngestJobStore Port

**Files:**
- Create: `src/rag/ports/ingest_job_store.py`
- Modify: `src/rag/ports/__init__.py`
- Test: `tests/domain/test_ingestion_models.py` (extend with port protocol check)

Defines the port (Protocol) for persisting and querying ingestion jobs, tasks, and document records. No concrete implementation yet — just the interface.

**Step 1: Write the failing test**

Append to `tests/domain/test_ingestion_models.py`:

```python
from typing import runtime_checkable

from rag.ports.ingest_job_store import IngestJobStore


class TestIngestJobStoreProtocol:
    def test_is_runtime_checkable(self) -> None:
        assert runtime_checkable(IngestJobStore)
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/domain/test_ingestion_models.py::TestIngestJobStoreProtocol -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/rag/ports/ingest_job_store.py
"""Port for ingestion job and task persistence."""
from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)


@runtime_checkable
class IngestJobStore(Protocol):
    """Persistence layer for ingestion jobs, tasks, and document records.

    Implementations back this with Postgres (production) or in-memory
    dicts (testing).
    """

    # ── Schema bootstrap ──────────────────────────────────────────
    def ensure_schema(self) -> None:
        """Create tables/indexes if they don't exist."""
        ...

    # ── Jobs ──────────────────────────────────────────────────────
    def create_job(self, job: IngestJob) -> None:
        """Insert a new ingestion job."""
        ...

    def get_job(self, job_id: uuid.UUID) -> IngestJob | None:
        """Fetch a job by ID."""
        ...

    def update_job_status(
        self, job_id: uuid.UUID, status: JobStatus, *, stats: dict[str, object] | None = None
    ) -> None:
        """Transition a job to a new status."""
        ...

    # ── Tasks ─────────────────────────────────────────────────────
    def create_tasks(self, tasks: Sequence[IngestTask]) -> None:
        """Bulk insert pending tasks for a job."""
        ...

    def acquire_task(
        self, job_id: uuid.UUID, *, lease_owner: str, lease_duration_s: int = 300
    ) -> IngestTask | None:
        """Atomically claim the next PENDING or RETRYABLE task.

        Sets status=RUNNING, lease_owner, lease_expires_at.
        Returns None when no claimable tasks remain.
        """
        ...

    def complete_task(self, task_id: uuid.UUID) -> None:
        """Mark a task SUCCEEDED."""
        ...

    def fail_task(self, task_id: uuid.UUID, *, error: str, retryable: bool = True) -> None:
        """Mark a task FAILED or RETRYABLE."""
        ...

    def get_task_counts(self, job_id: uuid.UUID) -> dict[TaskStatus, int]:
        """Return task counts grouped by status for a job."""
        ...

    # ── Document records ──────────────────────────────────────────
    def upsert_document(self, doc: DocumentRecord) -> None:
        """Insert or update a document record (idempotent by corpus_id + doc_id)."""
        ...

    def get_document(self, corpus_id: str, doc_id: str) -> DocumentRecord | None:
        """Fetch a document record."""
        ...
```

Update `src/rag/ports/__init__.py` — add `IngestJobStore` to imports and `__all__`.

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/domain/test_ingestion_models.py::TestIngestJobStoreProtocol -v`
Expected: PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check src/rag/ports/ingest_job_store.py`

**Step 6: Suggested commit**

```
feat(ports): add IngestJobStore protocol for distributed ingestion

Files: src/rag/ports/ingest_job_store.py, src/rag/ports/__init__.py, tests/domain/test_ingestion_models.py
```

---

## Task 3: In-Memory IngestJobStore (Test Double)

**Files:**
- Create: `tests/fakes/fake_ingest_job_store.py`
- Test: `tests/fakes/test_fake_ingest_job_store.py`

Before building the Postgres adapter, build a fake that passes a comprehensive contract test suite. This same test suite will later be reused for the Postgres adapter.

**Step 1: Write the contract test module**

```python
# tests/fakes/test_fake_ingest_job_store.py
"""Contract tests for IngestJobStore implementations.

These tests verify behavior, not implementation. They can be
parameterized to run against FakeIngestJobStore and (later)
PostgresIngestJobStore.
"""
from __future__ import annotations

import uuid

import pytest

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)
from tests.fakes.fake_ingest_job_store import FakeIngestJobStore


@pytest.fixture
def store() -> FakeIngestJobStore:
    s = FakeIngestJobStore()
    s.ensure_schema()
    return s


def _make_job(
    corpus_id: str = "test_corpus",
    status: JobStatus = JobStatus.CREATED,
) -> IngestJob:
    return IngestJob(
        job_id=uuid.uuid4(),
        corpus_id=corpus_id,
        index_id=f"{corpus_id}__fixed__dummy__2026-02-12",
        chunking_strategy="fixed",
        embedder_model="dummy",
        qdrant_collection="test",
        status=status,
    )


def _make_task(job_id: uuid.UUID, doc_id: str = "doc_1") -> IngestTask:
    return IngestTask(
        job_id=job_id,
        task_id=uuid.uuid4(),
        doc_id=doc_id,
        status=TaskStatus.PENDING,
    )


class TestJobLifecycle:
    def test_create_and_get(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        got = store.get_job(job.job_id)
        assert got is not None
        assert got.job_id == job.job_id
        assert got.status == JobStatus.CREATED

    def test_get_missing_returns_none(self, store: FakeIngestJobStore) -> None:
        assert store.get_job(uuid.uuid4()) is None

    def test_update_status(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        store.update_job_status(job.job_id, JobStatus.RUNNING)
        got = store.get_job(job.job_id)
        assert got is not None
        assert got.status == JobStatus.RUNNING

    def test_update_status_with_stats(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        store.update_job_status(job.job_id, JobStatus.COMPLETED, stats={"docs": 42})
        got = store.get_job(job.job_id)
        assert got is not None
        assert got.stats == {"docs": 42}


class TestTaskLifecycle:
    def test_create_and_acquire(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        tasks = [_make_task(job.job_id, f"doc_{i}") for i in range(3)]
        store.create_tasks(tasks)

        acquired = store.acquire_task(job.job_id, lease_owner="worker-1")
        assert acquired is not None
        assert acquired.status == TaskStatus.RUNNING
        assert acquired.lease_owner == "worker-1"

    def test_acquire_returns_none_when_empty(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        assert store.acquire_task(job.job_id, lease_owner="w") is None

    def test_complete_task(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        task = _make_task(job.job_id)
        store.create_tasks([task])

        acquired = store.acquire_task(job.job_id, lease_owner="w")
        assert acquired is not None
        store.complete_task(acquired.task_id)

        counts = store.get_task_counts(job.job_id)
        assert counts.get(TaskStatus.SUCCEEDED, 0) == 1

    def test_fail_task_retryable(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        task = _make_task(job.job_id)
        store.create_tasks([task])

        acquired = store.acquire_task(job.job_id, lease_owner="w")
        assert acquired is not None
        store.fail_task(acquired.task_id, error="timeout", retryable=True)

        counts = store.get_task_counts(job.job_id)
        assert counts.get(TaskStatus.RETRYABLE, 0) == 1

    def test_fail_task_permanent(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        task = _make_task(job.job_id)
        store.create_tasks([task])

        acquired = store.acquire_task(job.job_id, lease_owner="w")
        assert acquired is not None
        store.fail_task(acquired.task_id, error="corrupt data", retryable=False)

        counts = store.get_task_counts(job.job_id)
        assert counts.get(TaskStatus.FAILED, 0) == 1

    def test_retryable_task_can_be_reacquired(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        task = _make_task(job.job_id)
        store.create_tasks([task])

        # First attempt: acquire then fail as retryable
        acquired = store.acquire_task(job.job_id, lease_owner="w1")
        assert acquired is not None
        store.fail_task(acquired.task_id, error="timeout", retryable=True)

        # Second attempt: should be acquirable again
        reacquired = store.acquire_task(job.job_id, lease_owner="w2")
        assert reacquired is not None
        assert reacquired.task_id == acquired.task_id
        assert reacquired.lease_owner == "w2"

    def test_task_counts(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        tasks = [_make_task(job.job_id, f"doc_{i}") for i in range(5)]
        store.create_tasks(tasks)

        counts = store.get_task_counts(job.job_id)
        assert counts[TaskStatus.PENDING] == 5


class TestDocumentRecords:
    def test_upsert_and_get(self, store: FakeIngestJobStore) -> None:
        doc = DocumentRecord(
            corpus_id="c",
            doc_id="d",
            source="filesystem",
            uri="/test.md",
            content_sha256="a" * 64,
            s3_raw_key="corpus/c/raw/ab/d.json",
        )
        store.upsert_document(doc)
        got = store.get_document("c", "d")
        assert got is not None
        assert got.s3_raw_key == doc.s3_raw_key

    def test_upsert_is_idempotent(self, store: FakeIngestJobStore) -> None:
        doc = DocumentRecord(
            corpus_id="c",
            doc_id="d",
            source="filesystem",
            uri="/test.md",
            content_sha256="a" * 64,
            s3_raw_key="corpus/c/raw/ab/d_v1.json",
        )
        store.upsert_document(doc)

        doc2 = DocumentRecord(
            corpus_id="c",
            doc_id="d",
            source="filesystem",
            uri="/test.md",
            content_sha256="b" * 64,
            s3_raw_key="corpus/c/raw/ab/d_v2.json",
        )
        store.upsert_document(doc2)

        got = store.get_document("c", "d")
        assert got is not None
        assert got.s3_raw_key == "corpus/c/raw/ab/d_v2.json"

    def test_get_missing_returns_none(self, store: FakeIngestJobStore) -> None:
        assert store.get_document("x", "y") is None
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/fakes/test_fake_ingest_job_store.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'tests.fakes.fake_ingest_job_store'`

**Step 3: Write minimal implementation**

Create `tests/__init__.py` and `tests/fakes/__init__.py` if they don't exist.

```python
# tests/fakes/fake_ingest_job_store.py
"""In-memory IngestJobStore for testing."""
from __future__ import annotations

import uuid
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)


class FakeIngestJobStore:
    """In-memory implementation of the IngestJobStore protocol."""

    def __init__(self) -> None:
        self._jobs: dict[uuid.UUID, IngestJob] = {}
        self._tasks: dict[uuid.UUID, IngestTask] = {}
        self._documents: dict[tuple[str, str], DocumentRecord] = {}

    def ensure_schema(self) -> None:
        pass  # no-op for in-memory

    # ── Jobs ──────────────────────────────────────────────────────

    def create_job(self, job: IngestJob) -> None:
        self._jobs[job.job_id] = job

    def get_job(self, job_id: uuid.UUID) -> IngestJob | None:
        return self._jobs.get(job_id)

    def update_job_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        *,
        stats: dict[str, object] | None = None,
    ) -> None:
        job = self._jobs[job_id]
        updates: dict[str, object] = {"status": status, "updated_at": datetime.now(UTC)}
        if stats is not None:
            updates["stats"] = stats
        self._jobs[job_id] = replace(job, **updates)  # type: ignore[arg-type]

    # ── Tasks ─────────────────────────────────────────────────────

    def create_tasks(self, tasks: Sequence[IngestTask]) -> None:
        for t in tasks:
            self._tasks[t.task_id] = t

    def acquire_task(
        self,
        job_id: uuid.UUID,
        *,
        lease_owner: str,
        lease_duration_s: int = 300,
    ) -> IngestTask | None:
        for tid, t in self._tasks.items():
            if t.job_id != job_id:
                continue
            if t.status not in (TaskStatus.PENDING, TaskStatus.RETRYABLE):
                continue
            updated = replace(
                t,
                status=TaskStatus.RUNNING,
                lease_owner=lease_owner,
                lease_expires_at=datetime.now(UTC) + timedelta(seconds=lease_duration_s),
                attempt=t.attempt + 1,
                updated_at=datetime.now(UTC),
            )
            self._tasks[tid] = updated
            return updated
        return None

    def complete_task(self, task_id: uuid.UUID) -> None:
        t = self._tasks[task_id]
        self._tasks[task_id] = replace(
            t, status=TaskStatus.SUCCEEDED, updated_at=datetime.now(UTC)
        )

    def fail_task(
        self,
        task_id: uuid.UUID,
        *,
        error: str,
        retryable: bool = True,
    ) -> None:
        t = self._tasks[task_id]
        new_status = TaskStatus.RETRYABLE if retryable else TaskStatus.FAILED
        self._tasks[task_id] = replace(
            t,
            status=new_status,
            last_error=error,
            updated_at=datetime.now(UTC),
        )

    def get_task_counts(self, job_id: uuid.UUID) -> dict[TaskStatus, int]:
        counts: dict[TaskStatus, int] = defaultdict(int)
        for t in self._tasks.values():
            if t.job_id == job_id:
                counts[t.status] += 1
        return dict(counts)

    # ── Document records ──────────────────────────────────────────

    def upsert_document(self, doc: DocumentRecord) -> None:
        self._documents[(doc.corpus_id, doc.doc_id)] = doc

    def get_document(self, corpus_id: str, doc_id: str) -> DocumentRecord | None:
        return self._documents.get((corpus_id, doc_id))
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/fakes/test_fake_ingest_job_store.py -v`
Expected: All PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check tests/fakes/fake_ingest_job_store.py tests/fakes/test_fake_ingest_job_store.py`

**Step 6: Suggested commit**

```
test: add FakeIngestJobStore with contract tests

Files: tests/fakes/fake_ingest_job_store.py, tests/fakes/test_fake_ingest_job_store.py
```

---

## Task 4: RawDocumentStore Port + S3 Adapter

**Files:**
- Create: `src/rag/ports/raw_document_store.py`
- Create: `src/rag/adapters/corpus/s3_raw_document_store.py`
- Modify: `src/rag/ports/__init__.py`
- Test: `tests/adapters/corpus/test_s3_raw_document_store.py`

This stores normalized raw documents in S3 as JSON objects. Layout follows the plan: `s3://{bucket}/corpus/{corpus_id}/raw/{hash_prefix}/{doc_id_hash}.json`.

**Step 1: Write the port**

```python
# src/rag/ports/raw_document_store.py
"""Port for storing and retrieving raw documents in the corpus-of-record."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from rag.domain.models import Document


@runtime_checkable
class RawDocumentStore(Protocol):
    """Stores and retrieves raw (pre-chunking) documents.

    Implementations persist normalized document JSON to durable storage
    (S3) so the corpus-of-record is remote and not tied to local disk.
    """

    def store_document(
        self,
        doc: Document,
        *,
        corpus_id: str,
        content_sha256: str,
    ) -> str:
        """Persist a raw document. Returns the storage key (e.g. S3 key)."""
        ...

    def get_document(self, key: str) -> Document:
        """Retrieve a raw document by storage key."""
        ...
```

Add `RawDocumentStore` to `src/rag/ports/__init__.py` imports and `__all__`.

**Step 2: Write the test (mocked S3)**

```python
# tests/adapters/corpus/test_s3_raw_document_store.py
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
```

**Step 3: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/corpus/test_s3_raw_document_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write minimal implementation**

```python
# src/rag/adapters/corpus/__init__.py
```

```python
# src/rag/adapters/corpus/s3_raw_document_store.py
"""S3-backed raw document store for the corpus-of-record.

Stores normalized documents as JSON objects in S3 with hash-prefix
sharding for even key distribution.

S3 layout::

    s3://{bucket}/{prefix}/{hash[:4]}/{hash}.json

Each JSON object contains the full document text, metadata, and
content hash for idempotency checks.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from typing import Any

from rag.domain.models import Document


def _doc_s3_key(prefix: str, doc_id: str) -> str:
    """Deterministic S3 key for a raw document."""
    h = sha256(doc_id.encode("utf-8")).hexdigest()
    parts = [p for p in [prefix, h[:4], f"{h}.json"] if p]
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class S3RawDocumentStore:
    """Stores raw documents in S3 as JSON."""

    bucket: str
    prefix: str  # e.g. "corpus/{corpus_id}/raw"
    _s3: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import boto3  # type: ignore[import-untyped]
        object.__setattr__(self, "_s3", boto3.client("s3"))

    @classmethod
    def _for_test(cls, *, bucket: str, prefix: str, s3_client: Any) -> S3RawDocumentStore:
        """Create an instance with an injected S3 client (for testing)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "bucket", bucket)
        object.__setattr__(obj, "prefix", prefix)
        object.__setattr__(obj, "_s3", s3_client)
        return obj

    def store_document(
        self,
        doc: Document,
        *,
        corpus_id: str,
        content_sha256: str,
    ) -> str:
        """Persist a raw document to S3. Returns the S3 key."""
        s3_key = _doc_s3_key(self.prefix, doc.doc_id)

        body = json.dumps(
            {
                "corpus_id": corpus_id,
                "doc_id": doc.doc_id,
                "source": doc.source,
                "uri": doc.uri,
                "content_sha256": content_sha256,
                "metadata": dict(doc.metadata),
                "text": doc.text,
            },
            ensure_ascii=False,
        )

        self._s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return s3_key

    def get_document(self, key: str) -> Document:
        """Retrieve a raw document from S3 by key."""
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return Document(
            doc_id=data["doc_id"],
            text=data["text"],
            source=data["source"],
            uri=data["uri"],
            metadata=data.get("metadata", {}),
        )
```

**Step 5: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/corpus/test_s3_raw_document_store.py -v`
Expected: All PASS

**Step 6: Lint**

Run: `./scripts/py -m ruff check src/rag/adapters/corpus/ tests/adapters/corpus/`

**Step 7: Suggested commit**

```
feat: add RawDocumentStore port and S3 adapter

Files: src/rag/ports/raw_document_store.py, src/rag/adapters/corpus/s3_raw_document_store.py, tests/adapters/corpus/test_s3_raw_document_store.py, src/rag/ports/__init__.py
```

---

## Task 5: TaskQueue Port + SQS Adapter

**Files:**
- Create: `src/rag/ports/task_queue.py`
- Create: `src/rag/adapters/queue/sqs_task_queue.py`
- Modify: `src/rag/ports/__init__.py`
- Test: `tests/adapters/queue/test_sqs_task_queue.py`

Defines a simple port for enqueuing and dequeuing doc-level ingestion tasks, with an SQS implementation.

**Step 1: Write the port**

```python
# src/rag/ports/task_queue.py
"""Port for distributing ingestion tasks to workers."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskQueue(Protocol):
    """Distributes ingestion task messages to workers.

    Each message is a small JSON dict: ``{"job_id": "...", "corpus_id": "...", "doc_id": "..."}``.
    """

    def send(self, message: dict[str, str]) -> None:
        """Enqueue a single task message."""
        ...

    def send_batch(self, messages: Sequence[dict[str, str]]) -> None:
        """Enqueue a batch of task messages (max 10 per SQS batch)."""
        ...

    def receive(self, *, max_messages: int = 1, wait_seconds: int = 20) -> list[dict[str, Any]]:
        """Long-poll for messages.

        Returns a list of dicts, each containing:
        - ``"body"``: the parsed message body (dict)
        - ``"receipt_handle"``: opaque handle for ack/nack
        """
        ...

    def ack(self, receipt_handle: str) -> None:
        """Delete a message after successful processing."""
        ...

    def nack(self, receipt_handle: str, *, visibility_timeout: int = 0) -> None:
        """Return a message to the queue for retry."""
        ...
```

Add `TaskQueue` to `src/rag/ports/__init__.py`.

**Step 2: Write the test (mocked SQS)**

```python
# tests/adapters/queue/test_sqs_task_queue.py
"""Tests for SQSTaskQueue — uses unittest.mock for boto3."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, call

import pytest

from rag.adapters.queue.sqs_task_queue import SQSTaskQueue


@pytest.fixture
def mock_sqs() -> MagicMock:
    return MagicMock()


@pytest.fixture
def queue(mock_sqs: MagicMock) -> SQSTaskQueue:
    return SQSTaskQueue._for_test(queue_url="https://sqs.us-east-1.amazonaws.com/123/test-queue", sqs_client=mock_sqs)


class TestSend:
    def test_send_single(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        queue.send({"job_id": "j1", "corpus_id": "c1", "doc_id": "d1"})
        mock_sqs.send_message.assert_called_once()
        kwargs = mock_sqs.send_message.call_args.kwargs
        body = json.loads(kwargs["MessageBody"])
        assert body["doc_id"] == "d1"

    def test_send_batch(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        mock_sqs.send_message_batch.return_value = {"Failed": []}
        msgs = [{"job_id": "j", "corpus_id": "c", "doc_id": f"d{i}"} for i in range(3)]
        queue.send_batch(msgs)
        mock_sqs.send_message_batch.assert_called_once()
        entries = mock_sqs.send_message_batch.call_args.kwargs["Entries"]
        assert len(entries) == 3


class TestReceive:
    def test_receive_returns_parsed_messages(
        self, queue: SQSTaskQueue, mock_sqs: MagicMock
    ) -> None:
        mock_sqs.receive_message.return_value = {
            "Messages": [
                {
                    "Body": json.dumps({"job_id": "j1", "doc_id": "d1"}),
                    "ReceiptHandle": "handle-1",
                }
            ]
        }
        msgs = queue.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0]["body"]["doc_id"] == "d1"
        assert msgs[0]["receipt_handle"] == "handle-1"

    def test_receive_empty_queue(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        mock_sqs.receive_message.return_value = {}
        msgs = queue.receive()
        assert msgs == []


class TestAckNack:
    def test_ack_deletes_message(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        queue.ack("handle-1")
        mock_sqs.delete_message.assert_called_once_with(
            QueueUrl=queue.queue_url, ReceiptHandle="handle-1"
        )

    def test_nack_changes_visibility(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        queue.nack("handle-1", visibility_timeout=60)
        mock_sqs.change_message_visibility.assert_called_once()
```

**Step 3: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/queue/test_sqs_task_queue.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write minimal implementation**

```python
# src/rag/adapters/queue/__init__.py
```

```python
# src/rag/adapters/queue/sqs_task_queue.py
"""SQS-backed task queue for distributing ingestion work."""
from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SQSTaskQueue:
    """Distributes ingestion task messages via AWS SQS."""

    queue_url: str
    _sqs: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import boto3  # type: ignore[import-untyped]
        object.__setattr__(self, "_sqs", boto3.client("sqs"))

    @classmethod
    def _for_test(cls, *, queue_url: str, sqs_client: Any) -> SQSTaskQueue:
        obj = object.__new__(cls)
        object.__setattr__(obj, "queue_url", queue_url)
        object.__setattr__(obj, "_sqs", sqs_client)
        return obj

    def send(self, message: dict[str, str]) -> None:
        self._sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(message),
        )

    def send_batch(self, messages: Sequence[dict[str, str]]) -> None:
        entries = [
            {"Id": str(uuid.uuid4()), "MessageBody": json.dumps(m)}
            for m in messages
        ]
        # SQS batch limit is 10
        for i in range(0, len(entries), 10):
            batch = entries[i : i + 10]
            resp = self._sqs.send_message_batch(
                QueueUrl=self.queue_url, Entries=batch
            )
            failed = resp.get("Failed", [])
            if failed:
                raise RuntimeError(f"Failed to enqueue {len(failed)} messages: {failed}")

    def receive(self, *, max_messages: int = 1, wait_seconds: int = 20) -> list[dict[str, Any]]:
        resp = self._sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
        )
        raw_messages = resp.get("Messages", [])
        return [
            {
                "body": json.loads(m["Body"]),
                "receipt_handle": m["ReceiptHandle"],
            }
            for m in raw_messages
        ]

    def ack(self, receipt_handle: str) -> None:
        self._sqs.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
        )

    def nack(self, receipt_handle: str, *, visibility_timeout: int = 0) -> None:
        self._sqs.change_message_visibility(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout,
        )
```

**Step 5: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/queue/test_sqs_task_queue.py -v`
Expected: All PASS

**Step 6: Lint**

Run: `./scripts/py -m ruff check src/rag/adapters/queue/ tests/adapters/queue/`

**Step 7: Suggested commit**

```
feat: add TaskQueue port and SQS adapter

Files: src/rag/ports/task_queue.py, src/rag/adapters/queue/sqs_task_queue.py, tests/adapters/queue/test_sqs_task_queue.py, src/rag/ports/__init__.py
```

---

## Task 6: Postgres IngestJobStore Adapter

**Files:**
- Create: `src/rag/adapters/persistence/postgres_ingest_job_store.py`
- Test: `tests/adapters/persistence/test_postgres_ingest_job_store.py`

Uses the same `psycopg2` + connection pool pattern as `S3ChunkStore`. Tests mock the DB connection.

**Step 1: Write the test**

```python
# tests/adapters/persistence/test_postgres_ingest_job_store.py
"""Tests for PostgresIngestJobStore.

Uses a real SQLite-in-memory database via psycopg2 mocking for unit tests.
For true integration tests, use a Postgres testcontainer (marked slow).
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock, patch

import pytest

from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)


class TestEnsureSchema:
    def test_creates_tables(self) -> None:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        store = PostgresIngestJobStore._for_test(pool=mock_pool)
        store.ensure_schema()

        # Should execute CREATE TABLE statements
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        assert cursor.execute.call_count >= 4  # at least 4 tables
        mock_conn.commit.assert_called()


class TestCreateJob:
    def test_inserts_job(self) -> None:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn

        store = PostgresIngestJobStore._for_test(pool=mock_pool)
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id="c",
            index_id="i",
            chunking_strategy="fixed",
            embedder_model="dummy",
            qdrant_collection="q",
            status=JobStatus.CREATED,
        )
        store.create_job(job)

        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "INSERT INTO ingest_jobs" in sql
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/adapters/persistence/test_postgres_ingest_job_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/rag/adapters/persistence/__init__.py
```

```python
# src/rag/adapters/persistence/postgres_ingest_job_store.py
"""Postgres-backed storage for ingestion jobs, tasks, and document records.

Uses psycopg2 connection pooling (same pattern as S3ChunkStore).
"""
from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    job_id       TEXT PRIMARY KEY,
    corpus_id    TEXT NOT NULL,
    index_id     TEXT NOT NULL,
    chunking_strategy TEXT NOT NULL,
    embedder_model    TEXT NOT NULL,
    qdrant_collection TEXT NOT NULL,
    status       TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    stats        JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS ingest_tasks (
    task_id          TEXT PRIMARY KEY,
    job_id           TEXT NOT NULL REFERENCES ingest_jobs(job_id),
    doc_id           TEXT NOT NULL,
    status           TEXT NOT NULL,
    attempt          INT NOT NULL DEFAULT 0,
    lease_owner      TEXT,
    lease_expires_at TIMESTAMPTZ,
    last_error       TEXT,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (job_id, doc_id)
);
CREATE INDEX IF NOT EXISTS idx_ingest_tasks_status ON ingest_tasks(job_id, status);

CREATE TABLE IF NOT EXISTS corpus_documents (
    corpus_id       TEXT NOT NULL,
    doc_id          TEXT NOT NULL,
    source          TEXT NOT NULL,
    uri             TEXT NOT NULL,
    content_sha256  TEXT NOT NULL,
    s3_raw_key      TEXT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (corpus_id, doc_id)
);
"""


@dataclass(frozen=True, slots=True)
class PostgresIngestJobStore:
    """Postgres implementation of the IngestJobStore protocol."""

    postgres_dsn: str
    _pool: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import psycopg2.pool  # type: ignore[import-untyped]
        object.__setattr__(
            self,
            "_pool",
            psycopg2.pool.SimpleConnectionPool(1, 10, self.postgres_dsn),
        )

    @classmethod
    def _for_test(cls, *, pool: Any) -> PostgresIngestJobStore:
        obj = object.__new__(cls)
        object.__setattr__(obj, "postgres_dsn", "test://")
        object.__setattr__(obj, "_pool", pool)
        return obj

    def _conn(self) -> Any:
        return self._pool.getconn()

    def _put(self, conn: Any) -> None:
        self._pool.putconn(conn)

    # ── Schema ────────────────────────────────────────────────────

    def ensure_schema(self) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                for stmt in _SCHEMA_SQL.strip().split(";"):
                    stmt = stmt.strip()
                    if stmt:
                        cur.execute(stmt)
            conn.commit()
        finally:
            self._put(conn)

    # ── Jobs ──────────────────────────────────────────────────────

    def create_job(self, job: IngestJob) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ingest_jobs "
                    "(job_id, corpus_id, index_id, chunking_strategy, "
                    " embedder_model, qdrant_collection, status, "
                    " created_at, updated_at, stats) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                    (
                        str(job.job_id),
                        job.corpus_id,
                        job.index_id,
                        job.chunking_strategy,
                        job.embedder_model,
                        job.qdrant_collection,
                        job.status.value,
                        job.created_at,
                        job.updated_at,
                        json.dumps(job.stats),
                    ),
                )
            conn.commit()
        finally:
            self._put(conn)

    def get_job(self, job_id: uuid.UUID) -> IngestJob | None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT job_id, corpus_id, index_id, chunking_strategy, "
                    "       embedder_model, qdrant_collection, status, "
                    "       created_at, updated_at, stats "
                    "FROM ingest_jobs WHERE job_id = %s",
                    (str(job_id),),
                )
                row = cur.fetchone()
        finally:
            self._put(conn)

        if row is None:
            return None

        return IngestJob(
            job_id=uuid.UUID(row[0]),
            corpus_id=row[1],
            index_id=row[2],
            chunking_strategy=row[3],
            embedder_model=row[4],
            qdrant_collection=row[5],
            status=JobStatus(row[6]),
            created_at=row[7],
            updated_at=row[8],
            stats=row[9] if isinstance(row[9], dict) else json.loads(row[9] or "{}"),
        )

    def update_job_status(
        self,
        job_id: uuid.UUID,
        status: JobStatus,
        *,
        stats: dict[str, object] | None = None,
    ) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                if stats is not None:
                    cur.execute(
                        "UPDATE ingest_jobs SET status = %s, stats = %s, "
                        "updated_at = %s WHERE job_id = %s",
                        (status.value, json.dumps(stats), datetime.now(UTC), str(job_id)),
                    )
                else:
                    cur.execute(
                        "UPDATE ingest_jobs SET status = %s, updated_at = %s "
                        "WHERE job_id = %s",
                        (status.value, datetime.now(UTC), str(job_id)),
                    )
            conn.commit()
        finally:
            self._put(conn)

    # ── Tasks ─────────────────────────────────────────────────────

    def create_tasks(self, tasks: Sequence[IngestTask]) -> None:
        if not tasks:
            return
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                for t in tasks:
                    cur.execute(
                        "INSERT INTO ingest_tasks "
                        "(task_id, job_id, doc_id, status, attempt, updated_at) "
                        "VALUES (%s, %s, %s, %s, %s, %s) "
                        "ON CONFLICT (job_id, doc_id) DO NOTHING",
                        (
                            str(t.task_id),
                            str(t.job_id),
                            t.doc_id,
                            t.status.value,
                            t.attempt,
                            t.updated_at,
                        ),
                    )
            conn.commit()
        finally:
            self._put(conn)

    def acquire_task(
        self,
        job_id: uuid.UUID,
        *,
        lease_owner: str,
        lease_duration_s: int = 300,
    ) -> IngestTask | None:
        expires = datetime.now(UTC) + timedelta(seconds=lease_duration_s)
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_tasks SET "
                    "  status = %s, lease_owner = %s, "
                    "  lease_expires_at = %s, "
                    "  attempt = attempt + 1, updated_at = %s "
                    "WHERE task_id = ("
                    "  SELECT task_id FROM ingest_tasks "
                    "  WHERE job_id = %s AND status IN (%s, %s) "
                    "  ORDER BY updated_at ASC LIMIT 1 "
                    "  FOR UPDATE SKIP LOCKED"
                    ") "
                    "RETURNING task_id, job_id, doc_id, status, attempt, "
                    "          lease_owner, lease_expires_at, last_error, updated_at",
                    (
                        TaskStatus.RUNNING.value,
                        lease_owner,
                        expires,
                        datetime.now(UTC),
                        str(job_id),
                        TaskStatus.PENDING.value,
                        TaskStatus.RETRYABLE.value,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
        finally:
            self._put(conn)

        if row is None:
            return None

        return IngestTask(
            task_id=uuid.UUID(row[0]),
            job_id=uuid.UUID(row[1]),
            doc_id=row[2],
            status=TaskStatus(row[3]),
            attempt=row[4],
            lease_owner=row[5],
            lease_expires_at=row[6],
            last_error=row[7],
            updated_at=row[8],
        )

    def complete_task(self, task_id: uuid.UUID) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_tasks SET status = %s, updated_at = %s "
                    "WHERE task_id = %s",
                    (TaskStatus.SUCCEEDED.value, datetime.now(UTC), str(task_id)),
                )
            conn.commit()
        finally:
            self._put(conn)

    def fail_task(
        self,
        task_id: uuid.UUID,
        *,
        error: str,
        retryable: bool = True,
    ) -> None:
        status = TaskStatus.RETRYABLE if retryable else TaskStatus.FAILED
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE ingest_tasks SET status = %s, last_error = %s, "
                    "updated_at = %s WHERE task_id = %s",
                    (status.value, error, datetime.now(UTC), str(task_id)),
                )
            conn.commit()
        finally:
            self._put(conn)

    def get_task_counts(self, job_id: uuid.UUID) -> dict[TaskStatus, int]:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, COUNT(*) FROM ingest_tasks "
                    "WHERE job_id = %s GROUP BY status",
                    (str(job_id),),
                )
                rows = cur.fetchall()
        finally:
            self._put(conn)

        return {TaskStatus(row[0]): row[1] for row in rows}

    # ── Document records ──────────────────────────────────────────

    def upsert_document(self, doc: DocumentRecord) -> None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO corpus_documents "
                    "(corpus_id, doc_id, source, uri, content_sha256, "
                    " s3_raw_key, metadata, updated_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (corpus_id, doc_id) DO UPDATE SET "
                    "  source = EXCLUDED.source, "
                    "  uri = EXCLUDED.uri, "
                    "  content_sha256 = EXCLUDED.content_sha256, "
                    "  s3_raw_key = EXCLUDED.s3_raw_key, "
                    "  metadata = EXCLUDED.metadata, "
                    "  updated_at = EXCLUDED.updated_at",
                    (
                        doc.corpus_id,
                        doc.doc_id,
                        doc.source,
                        doc.uri,
                        doc.content_sha256,
                        doc.s3_raw_key,
                        json.dumps(doc.metadata),
                        doc.updated_at,
                    ),
                )
            conn.commit()
        finally:
            self._put(conn)

    def get_document(self, corpus_id: str, doc_id: str) -> DocumentRecord | None:
        conn = self._conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT corpus_id, doc_id, source, uri, content_sha256, "
                    "       s3_raw_key, metadata, updated_at "
                    "FROM corpus_documents WHERE corpus_id = %s AND doc_id = %s",
                    (corpus_id, doc_id),
                )
                row = cur.fetchone()
        finally:
            self._put(conn)

        if row is None:
            return None

        return DocumentRecord(
            corpus_id=row[0],
            doc_id=row[1],
            source=row[2],
            uri=row[3],
            content_sha256=row[4],
            s3_raw_key=row[5],
            metadata=row[6] if isinstance(row[6], dict) else json.loads(row[6] or "{}"),
            updated_at=row[7],
        )
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/adapters/persistence/test_postgres_ingest_job_store.py -v`
Expected: All PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check src/rag/adapters/persistence/`

**Step 6: Suggested commit**

```
feat: add PostgresIngestJobStore adapter

Files: src/rag/adapters/persistence/postgres_ingest_job_store.py, tests/adapters/persistence/test_postgres_ingest_job_store.py
```

---

## Task 7: Settings Extensions for Distributed Ingestion

**Files:**
- Modify: `src/rag/settings.py`
- Modify: `settings.toml`
- Test: `tests/test_settings_ingestion.py`

Add configuration sections for corpus identity, ingestion DB, and SQS queue.

**Step 1: Write the failing test**

```python
# tests/test_settings_ingestion.py
"""Tests for distributed ingestion settings."""
from __future__ import annotations

from rag.settings import DistributedIngestion, load_settings


class TestDistributedIngestionDefaults:
    def test_defaults_are_disabled(self) -> None:
        cfg = DistributedIngestion()
        assert cfg.enabled is False
        assert cfg.postgres_dsn is None
        assert cfg.sqs_queue_url is None
        assert cfg.corpus_s3_bucket is None


class TestLoadSettingsIncludesDistributed:
    def test_distributed_section_present(self, tmp_path) -> None:
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text("""
[paths]
vault_dir = "/tmp/vault"

[distributed_ingestion]
enabled = true
postgres_dsn = "postgresql://user:pass@host:5432/rag"
sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
corpus_s3_bucket = "rag-prod-artifacts"
corpus_s3_prefix = "corpus"
""")
        cfg = load_settings(toml_path, require_openai=False)
        assert cfg.distributed_ingestion.enabled is True
        assert cfg.distributed_ingestion.postgres_dsn == "postgresql://user:pass@host:5432/rag"
        assert cfg.distributed_ingestion.sqs_queue_url == "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
        assert cfg.distributed_ingestion.corpus_s3_bucket == "rag-prod-artifacts"
        assert cfg.distributed_ingestion.corpus_s3_prefix == "corpus"
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/test_settings_ingestion.py -v`
Expected: FAIL — `ImportError: cannot import name 'DistributedIngestion' from 'rag.settings'`

**Step 3: Write minimal implementation**

Add to `src/rag/settings.py`:

1. New dataclass after `ChunkStorage`:

```python
@dataclass(frozen=True, slots=True)
class DistributedIngestion:
    """Configuration for distributed ingestion (Phase 3).

    When ``enabled`` is True, ingestion uses remote Postgres for job
    tracking, SQS for task distribution, and S3 for the corpus-of-record.
    """

    enabled: bool = False
    postgres_dsn: str | None = None
    sqs_queue_url: str | None = None
    corpus_s3_bucket: str | None = None
    corpus_s3_prefix: str = ""
    worker_lease_duration_s: int = 300
    max_task_retries: int = 3
```

2. Add `distributed_ingestion: DistributedIngestion` field to `Settings`.

3. In `load_settings()`, parse the `[distributed_ingestion]` section:

```python
    dist_tbl = get_tbl("distributed_ingestion")
    distributed_ingestion = DistributedIngestion(
        enabled=bool(dist_tbl.get("enabled", False)),
        postgres_dsn=dist_tbl.get("postgres_dsn"),
        sqs_queue_url=dist_tbl.get("sqs_queue_url"),
        corpus_s3_bucket=dist_tbl.get("corpus_s3_bucket"),
        corpus_s3_prefix=str(dist_tbl.get("corpus_s3_prefix", "")),
        worker_lease_duration_s=int(dist_tbl.get("worker_lease_duration_s", 300)),
        max_task_retries=int(dist_tbl.get("max_task_retries", 3)),
    )
```

4. Add to `settings.toml`:

```toml
[distributed_ingestion]
enabled = false
# postgres_dsn = "postgresql://user:pass@host:5432/rag"
# sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
# corpus_s3_bucket = "rag-prod-artifacts"
# corpus_s3_prefix = "corpus"
# worker_lease_duration_s = 300
# max_task_retries = 3
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/test_settings_ingestion.py -v`
Expected: All PASS

**Step 5: Run full test suite to verify no regressions**

Run: `./scripts/py -m pytest --tb=short -q`

**Step 6: Lint**

Run: `./scripts/py -m ruff check src/rag/settings.py`

**Step 7: Suggested commit**

```
feat(settings): add distributed ingestion configuration

Files: src/rag/settings.py, settings.toml, tests/test_settings_ingestion.py
```

---

## Task 8: Enumerator Service

**Files:**
- Create: `src/rag/app/ingestion/enumerator.py`
- Test: `tests/app/ingestion/test_enumerator.py`

The enumerator walks a corpus, writes raw docs to S3, records them in the DB, and enqueues SQS messages. This is the "control-plane" entry point for a distributed ingestion run.

**Step 1: Write the failing test**

```python
# tests/app/ingestion/test_enumerator.py
"""Tests for the ingestion enumerator."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

from rag.app.ingestion.enumerator import Enumerator
from rag.domain.ingestion import IngestJob, JobStatus, TaskStatus
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
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/app/ingestion/test_enumerator.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/rag/app/ingestion/__init__.py
```

```python
# src/rag/app/ingestion/enumerator.py
"""Enumerator: the control-plane entry point for distributed ingestion.

Walks a set of documents, writes raw docs to S3, records them in the
job store, and enqueues SQS messages for workers.
"""
from __future__ import annotations

import logging
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)
from rag.domain.models import Document

logger = logging.getLogger(__name__)


@dataclass
class Enumerator:
    """Enumerates documents and dispatches ingestion tasks."""

    job_store: object  # IngestJobStore protocol
    raw_document_store: object  # RawDocumentStore protocol
    task_queue: object  # TaskQueue protocol

    def enumerate(
        self,
        *,
        docs: Sequence[Document],
        corpus_id: str,
        index_id: str,
        chunking_strategy: str,
        embedder_model: str,
        qdrant_collection: str,
    ) -> IngestJob:
        """Create a job, store raw docs, create tasks, enqueue messages.

        Returns the created IngestJob.
        """
        job = IngestJob(
            job_id=uuid.uuid4(),
            corpus_id=corpus_id,
            index_id=index_id,
            chunking_strategy=chunking_strategy,
            embedder_model=embedder_model,
            qdrant_collection=qdrant_collection,
            status=JobStatus.CREATED,
        )
        self.job_store.create_job(job)  # type: ignore[union-attr]

        if not docs:
            from dataclasses import replace

            job = replace(job, status=JobStatus.COMPLETED)
            self.job_store.update_job_status(job.job_id, JobStatus.COMPLETED, stats={"docs": 0})  # type: ignore[union-attr]
            return job

        logger.info("Enumerating %d docs for job %s", len(docs), job.job_id)

        # 1) Store raw docs in S3 and record in DB
        tasks: list[IngestTask] = []
        messages: list[dict[str, str]] = []

        for doc in docs:
            content_hash = sha256(doc.text.encode("utf-8")).hexdigest()

            s3_key = self.raw_document_store.store_document(  # type: ignore[union-attr]
                doc, corpus_id=corpus_id, content_sha256=content_hash
            )

            doc_record = DocumentRecord(
                corpus_id=corpus_id,
                doc_id=doc.doc_id,
                source=doc.source,
                uri=doc.uri,
                content_sha256=content_hash,
                s3_raw_key=s3_key,
                metadata=dict(doc.metadata),
            )
            self.job_store.upsert_document(doc_record)  # type: ignore[union-attr]

            task = IngestTask(
                job_id=job.job_id,
                task_id=uuid.uuid4(),
                doc_id=doc.doc_id,
                status=TaskStatus.PENDING,
            )
            tasks.append(task)

            messages.append(
                {
                    "job_id": str(job.job_id),
                    "corpus_id": corpus_id,
                    "doc_id": doc.doc_id,
                }
            )

        # 2) Bulk create tasks in DB
        self.job_store.create_tasks(tasks)  # type: ignore[union-attr]

        # 3) Enqueue messages for workers
        self.task_queue.send_batch(messages)  # type: ignore[union-attr]

        # 4) Transition job to RUNNING
        self.job_store.update_job_status(job.job_id, JobStatus.RUNNING)  # type: ignore[union-attr]

        from dataclasses import replace

        return replace(job, status=JobStatus.RUNNING)
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/app/ingestion/test_enumerator.py -v`
Expected: All PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check src/rag/app/ingestion/`

**Step 6: Suggested commit**

```
feat: add ingestion enumerator — walks corpus, stores to S3, enqueues tasks

Files: src/rag/app/ingestion/enumerator.py, tests/app/ingestion/test_enumerator.py
```

---

## Task 9: Worker Service

**Files:**
- Create: `src/rag/app/ingestion/worker.py`
- Test: `tests/app/ingestion/test_worker.py`

The worker processes a single document task: loads raw doc from S3, chunks, embeds, writes to chunk store + vector store, and marks the task complete.

**Step 1: Write the failing test**

```python
# tests/app/ingestion/test_worker.py
"""Tests for the ingestion worker."""
from __future__ import annotations

import uuid

import pytest

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.app.ingestion.worker import Worker
from rag.domain.ingestion import IngestJob, IngestTask, JobStatus, TaskStatus
from tests.app.ingestion.test_enumerator import FakeRawDocumentStore, FakeTaskQueue
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
        s3_key = f"corpus/c/raw/doc-1.json"

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
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/app/ingestion/test_worker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/rag/app/ingestion/worker.py
"""Worker: processes a single-document ingestion task.

Pulls a raw document from S3, chunks it, embeds, and writes to both
the chunk store (S3 shards) and vector store (Qdrant).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from rag.domain.ingestion import IngestTask
from rag.domain.models import Document

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    """Processes individual document ingestion tasks."""

    job_store: object  # IngestJobStore protocol
    raw_document_store: object  # RawDocumentStore protocol
    chunker: object  # Chunker protocol
    embedder: object  # Embedder protocol
    vector_store: object  # VectorStore protocol
    chunk_store: object  # ChunkStore protocol
    worker_id: str = "worker-0"

    def process_task(
        self,
        *,
        task: IngestTask,
        s3_raw_key: str,
        corpus_id: str,
    ) -> bool:
        """Process a single document task.

        Returns True on success, False on failure.
        """
        try:
            # 1) Load raw doc from S3
            doc: Document = self.raw_document_store.get_document(s3_raw_key)  # type: ignore[union-attr]
            if doc is None:
                raise ValueError(f"Raw document not found: {s3_raw_key}")

            # 2) Chunk
            chunks = self.chunker.chunk(doc)  # type: ignore[union-attr]
            if not chunks:
                logger.warning("No chunks produced for doc %s", task.doc_id)
                self.job_store.complete_task(task.task_id)  # type: ignore[union-attr]
                return True

            # 3) Embed
            vectors = self.embedder.embed_texts([c.text for c in chunks])  # type: ignore[union-attr]

            # 4) Write to vector store
            self.vector_store.upsert(chunks=chunks, vectors=vectors)  # type: ignore[union-attr]

            # 5) Write to chunk store (dual-write)
            self.chunk_store.store_chunks(chunks)  # type: ignore[union-attr]

            # 6) Mark task succeeded
            self.job_store.complete_task(task.task_id)  # type: ignore[union-attr]

            logger.info(
                "Worker %s completed doc %s: %d chunks",
                self.worker_id,
                task.doc_id,
                len(chunks),
            )
            return True

        except Exception:
            logger.exception(
                "Worker %s failed doc %s", self.worker_id, task.doc_id
            )
            self.job_store.fail_task(  # type: ignore[union-attr]
                task.task_id, error=str(Exception), retryable=True
            )
            return False
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/app/ingestion/test_worker.py -v`
Expected: All PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check src/rag/app/ingestion/worker.py`

**Step 6: Suggested commit**

```
feat: add ingestion worker — processes single-doc tasks from SQS

Files: src/rag/app/ingestion/worker.py, tests/app/ingestion/test_worker.py
```

---

## Task 10: Worker SQS Loop

**Files:**
- Create: `src/rag/app/ingestion/worker_loop.py`
- Test: `tests/app/ingestion/test_worker_loop.py`

The long-running loop that polls SQS, acquires a DB lease, processes the task, and acks/nacks the message.

**Step 1: Write the failing test**

```python
# tests/app/ingestion/test_worker_loop.py
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
from tests.app.ingestion.test_enumerator import FakeRawDocumentStore, FakeTaskQueue
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
        queue = FakeTaskQueue()
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
```

**Step 2: Run test to verify it fails**

Run: `./scripts/py -m pytest tests/app/ingestion/test_worker_loop.py -v`
Expected: FAIL

**Step 3: Write minimal implementation**

```python
# src/rag/app/ingestion/worker_loop.py
"""SQS worker loop: poll messages, process tasks, ack/nack.

This module provides the ``process_one_message`` function (testable
without real SQS) and a ``run_loop`` entry point for the worker
service.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from rag.app.ingestion.worker import Worker

logger = logging.getLogger(__name__)


def process_one_message(
    *,
    message: dict[str, Any],
    worker: Worker,
    job_store: object,
    on_ack: Callable[[str], None],
    on_nack: Callable[[str], None],
) -> bool:
    """Process a single SQS message.

    Acquires a DB lease, processes the document, and calls on_ack or
    on_nack depending on the outcome. Returns True on success.
    """
    body = message["body"]
    receipt_handle = message["receipt_handle"]
    doc_id = body["doc_id"]
    corpus_id = body["corpus_id"]

    # Look up the document record for the S3 key
    doc_record = job_store.get_document(corpus_id, doc_id)  # type: ignore[union-attr]
    if doc_record is None:
        logger.error("No document record for %s/%s — nacking", corpus_id, doc_id)
        on_nack(receipt_handle)
        return False

    # Acquire a task lease from the DB
    import uuid

    job_id = uuid.UUID(body["job_id"])
    task = job_store.acquire_task(job_id, lease_owner=worker.worker_id)  # type: ignore[union-attr]
    if task is None:
        # No claimable task — might already be processed. Ack to remove from queue.
        logger.info("No claimable task for doc %s — acking (already done?)", doc_id)
        on_ack(receipt_handle)
        return True

    success = worker.process_task(
        task=task,
        s3_raw_key=doc_record.s3_raw_key,
        corpus_id=corpus_id,
    )

    if success:
        on_ack(receipt_handle)
    else:
        on_nack(receipt_handle)

    return success


def run_loop(
    *,
    worker: Worker,
    job_store: object,
    task_queue: object,
    max_iterations: int | None = None,
) -> None:
    """Long-running worker loop that polls SQS.

    Args:
        max_iterations: If set, stop after N iterations (for testing).
    """
    iterations = 0
    while max_iterations is None or iterations < max_iterations:
        messages = task_queue.receive(max_messages=1, wait_seconds=20)  # type: ignore[union-attr]

        for msg in messages:
            process_one_message(
                message=msg,
                worker=worker,
                job_store=job_store,
                on_ack=lambda h: task_queue.ack(h),  # type: ignore[union-attr]
                on_nack=lambda h: task_queue.nack(h, visibility_timeout=60),  # type: ignore[union-attr]
            )

        iterations += 1
```

**Step 4: Run test to verify it passes**

Run: `./scripts/py -m pytest tests/app/ingestion/test_worker_loop.py -v`
Expected: All PASS

**Step 5: Lint**

Run: `./scripts/py -m ruff check src/rag/app/ingestion/worker_loop.py`

**Step 6: Suggested commit**

```
feat: add SQS worker loop — polls, acquires lease, processes, acks

Files: src/rag/app/ingestion/worker_loop.py, tests/app/ingestion/test_worker_loop.py
```

---

## Task 11: CLI Entry Points

**Files:**
- Create: `scripts/start_ingestion.py`
- Create: `scripts/run_worker.py`

These are thin CLI scripts that wire up the distributed ingestion components and invoke the enumerator/worker.

**Step 1: Write the enumerator CLI**

```python
# scripts/start_ingestion.py
"""CLI: Start a distributed ingestion job.

Usage:
    ./scripts/py scripts/start_ingestion.py \
        --corpus /path/to/vault \
        --corpus-id regulations_v1 \
        --index-name regulations
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from rag import settings
from rag.app.container import ContainerOverrides, build_container
from rag.app.ingestion.enumerator import Enumerator

log = logging.getLogger("start_ingestion")


def main() -> None:
    ap = argparse.ArgumentParser(description="Start a distributed ingestion job.")
    ap.add_argument("--corpus", required=True, help="Path to corpus directory.")
    ap.add_argument("--corpus-id", required=True, help="Unique corpus identifier.")
    ap.add_argument("--index-name", required=True, help="Index name for manifest.")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs (0=all).")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[ingest] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()

    if not cfg.distributed_ingestion.enabled:
        log.error("distributed_ingestion.enabled must be true in settings.toml")
        raise SystemExit(1)

    container = build_container()

    # Ingest documents from local filesystem
    vault_root = Path(args.corpus).expanduser().resolve()
    docs, report = container.ingestor.ingest([str(vault_root)])

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    log.info("Ingested %d docs, now enumerating...", len(docs))

    # Build adapters from settings
    from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
    from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
    from rag.adapters.queue.sqs_task_queue import SQSTaskQueue

    job_store = PostgresIngestJobStore(postgres_dsn=cfg.distributed_ingestion.postgres_dsn)
    job_store.ensure_schema()

    raw_store = S3RawDocumentStore(
        bucket=cfg.distributed_ingestion.corpus_s3_bucket,
        prefix=f"{cfg.distributed_ingestion.corpus_s3_prefix}/{args.corpus_id}/raw",
    )
    queue = SQSTaskQueue(queue_url=cfg.distributed_ingestion.sqs_queue_url)

    enumerator = Enumerator(
        job_store=job_store,
        raw_document_store=raw_store,
        task_queue=queue,
    )

    # Build index_id
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    index_id = f"{args.index_name}_{cfg.chunking.backend}_{cfg.embeddings.model}_{ts}"

    job = enumerator.enumerate(
        docs=docs,
        corpus_id=args.corpus_id,
        index_id=index_id,
        chunking_strategy=cfg.chunking.backend,
        embedder_model=cfg.embeddings.model,
        qdrant_collection=cfg.vectorstore.qdrant_collection,
    )

    log.info("Job created: %s (status=%s, docs=%d)", job.job_id, job.status.value, len(docs))


if __name__ == "__main__":
    main()
```

**Step 2: Write the worker CLI**

```python
# scripts/run_worker.py
"""CLI: Run a distributed ingestion worker.

Usage:
    ./scripts/py scripts/run_worker.py --worker-id worker-1
"""
from __future__ import annotations

import argparse
import logging
import uuid

from dotenv import load_dotenv

from rag import settings
from rag.app.container import build_container
from rag.app.ingestion.worker import Worker
from rag.app.ingestion.worker_loop import run_loop

log = logging.getLogger("worker")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a distributed ingestion worker.")
    ap.add_argument(
        "--worker-id",
        default=f"worker-{uuid.uuid4().hex[:8]}",
        help="Unique worker identifier.",
    )
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[worker] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()

    if not cfg.distributed_ingestion.enabled:
        log.error("distributed_ingestion.enabled must be true in settings.toml")
        raise SystemExit(1)

    container = build_container()

    from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
    from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
    from rag.adapters.queue.sqs_task_queue import SQSTaskQueue

    job_store = PostgresIngestJobStore(postgres_dsn=cfg.distributed_ingestion.postgres_dsn)
    raw_store = S3RawDocumentStore(
        bucket=cfg.distributed_ingestion.corpus_s3_bucket,
        prefix=cfg.distributed_ingestion.corpus_s3_prefix,
    )
    queue = SQSTaskQueue(queue_url=cfg.distributed_ingestion.sqs_queue_url)

    worker = Worker(
        job_store=job_store,
        raw_document_store=raw_store,
        chunker=container.chunker,
        embedder=container.embedder,
        vector_store=container.store,
        chunk_store=container.chunk_store,
        worker_id=args.worker_id,
    )

    log.info("Worker %s starting SQS loop...", args.worker_id)
    run_loop(worker=worker, job_store=job_store, task_queue=queue)


if __name__ == "__main__":
    main()
```

**Step 3: Suggested commit**

```
feat: add CLI scripts for distributed ingestion

Files: scripts/start_ingestion.py, scripts/run_worker.py
```

---

## Task 12: Terraform — RDS, SQS, Worker ECS Service

**Files:**
- Create: `infra/modules/rds/main.tf`, `variables.tf`, `outputs.tf`
- Create: `infra/modules/sqs/main.tf`, `variables.tf`, `outputs.tf`
- Modify: `infra/main.tf` — add RDS and SQS modules
- Modify: `infra/modules/ecs/main.tf` — add ingest-worker task definition and service
- Modify: `infra/modules/ecs/iam.tf` — add SQS + RDS permissions
- Modify: `infra/modules/ecs/variables.tf` — add worker variables

This is infrastructure-only work. No Python tests, but verify with `terraform validate`.

**Step 1: Create RDS module**

```hcl
# infra/modules/rds/main.tf
resource "aws_db_subnet_group" "this" {
  name       = "${var.name_prefix}-db-subnet"
  subnet_ids = var.subnet_ids
  tags       = var.tags
}

resource "aws_db_instance" "this" {
  identifier              = "${var.name_prefix}-ingest"
  engine                  = "postgres"
  engine_version          = "16.4"
  instance_class          = var.instance_class
  allocated_storage       = 20
  storage_type            = "gp3"
  db_name                 = "rag"
  username                = var.db_username
  password                = var.db_password
  db_subnet_group_name    = aws_db_subnet_group.this.name
  vpc_security_group_ids  = var.security_group_ids
  skip_final_snapshot     = true
  publicly_accessible     = false
  backup_retention_period = 7
  tags                    = var.tags
}
```

```hcl
# infra/modules/rds/variables.tf
variable "name_prefix"        { type = string }
variable "subnet_ids"         { type = list(string) }
variable "security_group_ids" { type = list(string) }
variable "instance_class"     { type = string; default = "db.t4g.micro" }
variable "db_username"        { type = string; default = "rag" }
variable "db_password"        { type = string; sensitive = true }
variable "tags"               { type = map(string); default = {} }
```

```hcl
# infra/modules/rds/outputs.tf
output "endpoint" {
  description = "RDS endpoint (host:port)"
  value       = aws_db_instance.this.endpoint
}

output "dsn" {
  description = "Postgres connection string"
  value       = "postgresql://${var.db_username}:${var.db_password}@${aws_db_instance.this.endpoint}/rag"
  sensitive   = true
}
```

**Step 2: Create SQS module**

```hcl
# infra/modules/sqs/main.tf
resource "aws_sqs_queue" "ingest_tasks" {
  name                       = "${var.name_prefix}-ingest-tasks"
  visibility_timeout_seconds = var.visibility_timeout
  message_retention_seconds  = 1209600  # 14 days
  receive_wait_time_seconds  = 20       # long polling
  tags                       = var.tags
}

resource "aws_sqs_queue" "ingest_dlq" {
  name = "${var.name_prefix}-ingest-tasks-dlq"
  tags = var.tags
}

resource "aws_sqs_queue_redrive_policy" "this" {
  queue_url = aws_sqs_queue.ingest_tasks.id
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.ingest_dlq.arn
    maxReceiveCount     = var.max_receive_count
  })
}
```

```hcl
# infra/modules/sqs/variables.tf
variable "name_prefix"        { type = string }
variable "visibility_timeout" { type = number; default = 300 }
variable "max_receive_count"  { type = number; default = 5 }
variable "tags"               { type = map(string); default = {} }
```

```hcl
# infra/modules/sqs/outputs.tf
output "queue_url" {
  value = aws_sqs_queue.ingest_tasks.url
}

output "queue_arn" {
  value = aws_sqs_queue.ingest_tasks.arn
}

output "dlq_url" {
  value = aws_sqs_queue.ingest_dlq.url
}
```

**Step 3: Add worker task definition to ECS module**

Add to `infra/modules/ecs/main.tf`:

```hcl
# --- Ingest worker task definition ---
resource "aws_ecs_task_definition" "ingest_worker" {
  family                   = "${var.cluster_name}-ingest-worker"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.worker_cpu
  memory                   = var.worker_memory
  execution_role_arn       = aws_iam_role.task_execution.arn
  task_role_arn            = aws_iam_role.task.arn

  container_definitions = jsonencode([
    {
      name      = "ingest-worker"
      image     = var.app_image
      essential = true
      command   = ["python", "scripts/run_worker.py"]
      environment = [
        { name = "RAG_VECTORSTORE__BACKEND", value = "qdrant" },
        { name = "RAG_VECTORSTORE__QDRANT_URL", value = "http://qdrant.${var.cluster_name}.local:6333" },
        { name = "RAG_DISTRIBUTED_INGESTION__ENABLED", value = "true" },
        { name = "RAG_DISTRIBUTED_INGESTION__SQS_QUEUE_URL", value = var.sqs_queue_url },
        { name = "RAG_DISTRIBUTED_INGESTION__CORPUS_S3_BUCKET", value = var.s3_bucket_name },
      ]
      secrets = [
        { name = "OPENAI_API_KEY", valueFrom = var.openai_api_key_arn },
        { name = "RAG_DISTRIBUTED_INGESTION__POSTGRES_DSN", valueFrom = var.rds_dsn_arn },
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.worker.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "worker"
        }
      }
    }
  ])

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "worker" {
  name              = "/ecs/${var.cluster_name}/ingest-worker"
  retention_in_days = 30
  tags              = var.tags
}

# --- Ingest worker service ---
resource "aws_ecs_service" "ingest_worker" {
  name            = "${var.cluster_name}-ingest-worker"
  cluster         = aws_ecs_cluster.this.id
  task_definition = aws_ecs_task_definition.ingest_worker.arn
  desired_count   = var.worker_desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = var.subnet_ids
    security_groups  = var.security_group_ids
    assign_public_ip = true
  }

  tags = var.tags
}
```

**Step 4: Add SQS + RDS IAM policies**

Add to `infra/modules/ecs/iam.tf`:

```hcl
resource "aws_iam_role_policy" "task_sqs" {
  name = "${var.cluster_name}-sqs-access"
  role = aws_iam_role.task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:SendMessageBatch",
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:ChangeMessageVisibility",
          "sqs:GetQueueAttributes"
        ]
        Resource = [var.sqs_queue_arn]
      }
    ]
  })
}
```

**Step 5: Validate Terraform**

Run: `cd infra && terraform validate`
Expected: Success

**Step 6: Suggested commit**

```
infra: add RDS, SQS, and worker ECS task definition

Files: infra/modules/rds/*, infra/modules/sqs/*, infra/modules/ecs/main.tf, infra/modules/ecs/iam.tf, infra/modules/ecs/variables.tf, infra/main.tf
```

---

## Task 13: Documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md` — add distributed ingestion section
- Create: `docs/specs/05-distributed-ingestion.md` — formal spec

**Step 1: Add architecture diagram**

Add a new section to `docs/ARCHITECTURE.md` with a mermaid diagram showing the distributed ingestion flow:

```mermaid
flowchart LR
    subgraph Enumerator
        E[start_ingestion.py]
    end
    subgraph S3
        RAW[(Raw Docs)]
        CHUNKS[(Chunk Shards)]
    end
    subgraph Postgres
        JOBS[(ingest_jobs)]
        TASKS[(ingest_tasks)]
        DOCS[(corpus_documents)]
        CI[(chunk_index)]
    end
    subgraph SQS
        Q[ingest-tasks queue]
    end
    subgraph Workers["ECS Workers (N replicas)"]
        W1[run_worker.py]
        W2[run_worker.py]
    end
    subgraph Qdrant
        V[(vectors + thin payload)]
    end

    E -->|write raw docs| RAW
    E -->|create job + tasks| JOBS
    E -->|create doc records| DOCS
    E -->|enqueue messages| Q
    Q --> W1
    Q --> W2
    W1 -->|read raw doc| RAW
    W1 -->|chunk + embed| CHUNKS
    W1 -->|upsert vectors| V
    W1 -->|update task status| TASKS
    W2 -->|read raw doc| RAW
    W2 -->|chunk + embed| CHUNKS
    W2 -->|upsert vectors| V
    W2 -->|update task status| TASKS
```

**Step 2: Suggested commit**

```
docs: add distributed ingestion architecture documentation

Files: docs/ARCHITECTURE.md, docs/specs/05-distributed-ingestion.md
```

---

## Task 14: Full Integration Smoke Test

**Files:**
- Test: `tests/integration/test_distributed_ingestion.py`

End-to-end test using all fakes (no real AWS/Postgres). Verifies the full enumerator → worker loop.

**Step 1: Write the test**

```python
# tests/integration/test_distributed_ingestion.py
"""Integration test: full enumerator → worker pipeline with fakes."""
from __future__ import annotations

import uuid

import pytest

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.app.ingestion.enumerator import Enumerator
from rag.app.ingestion.worker import Worker
from rag.app.ingestion.worker_loop import process_one_message
from rag.domain.ingestion import JobStatus, TaskStatus
from tests.app.ingestion.test_enumerator import FakeRawDocumentStore, FakeTaskQueue
from tests.conftest import FakeChunkStore, make_document
from tests.fakes.fake_ingest_job_store import FakeIngestJobStore


class TestDistributedIngestionEndToEnd:
    def test_enumerate_then_process_all(self) -> None:
        # Setup fakes
        job_store = FakeIngestJobStore()
        job_store.ensure_schema()
        raw_store = FakeRawDocumentStore()
        queue = FakeTaskQueue()
        chunk_store = FakeChunkStore()
        vector_store = InMemoryVectorStore()

        # Create documents
        docs = [
            make_document(doc_id=f"doc-{i}", text=f"Content for document {i}. " * 20)
            for i in range(5)
        ]

        # 1) Enumerate
        enumerator = Enumerator(
            job_store=job_store,
            raw_document_store=raw_store,
            task_queue=queue,
        )
        job = enumerator.enumerate(
            docs=docs,
            corpus_id="test",
            index_id="test_index",
            chunking_strategy="fixed",
            embedder_model="dummy",
            qdrant_collection="test",
        )

        assert job.status == JobStatus.RUNNING
        assert len(queue.sent) == 5

        # 2) Process all queued messages
        worker = Worker(
            job_store=job_store,
            raw_document_store=raw_store,
            chunker=FixedChunker(chunk_size=100, overlap=20),
            embedder=DummyEmbedder(dim=128),
            vector_store=vector_store,
            chunk_store=chunk_store,
            worker_id="test-worker",
        )

        acked = []
        for msg_body in queue.sent:
            message = {"body": msg_body, "receipt_handle": f"handle-{msg_body['doc_id']}"}
            result = process_one_message(
                message=message,
                worker=worker,
                job_store=job_store,
                on_ack=lambda h: acked.append(h),
                on_nack=lambda h: None,
            )
            assert result is True

        # All 5 messages should be acked
        assert len(acked) == 5

        # All tasks should be SUCCEEDED
        counts = job_store.get_task_counts(job.job_id)
        assert counts.get(TaskStatus.SUCCEEDED, 0) == 5
        assert counts.get(TaskStatus.PENDING, 0) == 0

        # Chunks should be stored
        assert len(chunk_store.stored_chunks) > 0
```

**Step 2: Run test**

Run: `./scripts/py -m pytest tests/integration/test_distributed_ingestion.py -v`
Expected: All PASS

**Step 3: Suggested commit**

```
test: add end-to-end integration test for distributed ingestion

Files: tests/integration/test_distributed_ingestion.py
```

---

## Task 15: Run Full Test Suite + Lint + Typecheck

**Step 1: Run all tests**

Run: `./scripts/py -m pytest --tb=short -q`
Expected: All PASS, no regressions

**Step 2: Lint**

Run: `./scripts/py -m ruff check .`
Expected: Clean

**Step 3: Format**

Run: `./scripts/py -m ruff format .`

**Step 4: Typecheck**

Run: `./scripts/py -m mypy src/rag`

**Step 5: Suggested commit (if any fixes needed)**

```
fix: address lint/type issues from Phase 3 implementation
```

---

## Summary of All Suggested Commits

| # | Message | Key Files |
|---|---------|-----------|
| 1 | `feat(domain): add ingestion orchestration models` | `src/rag/domain/ingestion.py` |
| 2 | `feat(ports): add IngestJobStore protocol` | `src/rag/ports/ingest_job_store.py` |
| 3 | `test: add FakeIngestJobStore with contract tests` | `tests/fakes/fake_ingest_job_store.py` |
| 4 | `feat: add RawDocumentStore port and S3 adapter` | `src/rag/adapters/corpus/s3_raw_document_store.py` |
| 5 | `feat: add TaskQueue port and SQS adapter` | `src/rag/adapters/queue/sqs_task_queue.py` |
| 6 | `feat: add PostgresIngestJobStore adapter` | `src/rag/adapters/persistence/postgres_ingest_job_store.py` |
| 7 | `feat(settings): add distributed ingestion configuration` | `src/rag/settings.py`, `settings.toml` |
| 8 | `feat: add ingestion enumerator` | `src/rag/app/ingestion/enumerator.py` |
| 9 | `feat: add ingestion worker` | `src/rag/app/ingestion/worker.py` |
| 10 | `feat: add SQS worker loop` | `src/rag/app/ingestion/worker_loop.py` |
| 11 | `feat: add CLI scripts for distributed ingestion` | `scripts/start_ingestion.py`, `scripts/run_worker.py` |
| 12 | `infra: add RDS, SQS, and worker ECS task definition` | `infra/modules/rds/*`, `infra/modules/sqs/*` |
| 13 | `docs: add distributed ingestion architecture documentation` | `docs/ARCHITECTURE.md` |
| 14 | `test: add end-to-end integration test` | `tests/integration/test_distributed_ingestion.py` |

## Acceptance Criteria (from distributed-architecture-plan.md)

- [ ] Raw docs stored in S3 under `corpus/{corpus_id}/raw/...`
- [ ] Ingestion jobs and tasks tracked in Postgres with resumability
- [ ] SQS distributes work to N workers
- [ ] Workers chunk/embed/store independently (one doc per task)
- [ ] Failed workers don't block the job (lease expiry + retry)
- [ ] Idempotent: re-running doesn't duplicate chunks (stable IDs + upserts)
- [ ] Existing `build_index.py` still works unchanged (Phase 3 is additive)
- [ ] All existing tests pass
