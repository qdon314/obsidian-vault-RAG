"""Contract tests for IngestJobStore implementations.

These tests verify behavior, not implementation. They can be
parameterized to run against FakeIngestJobStore and (later)
PostgresIngestJobStore.
"""
from __future__ import annotations

import uuid
from dataclasses import replace
from datetime import UTC, datetime, timedelta

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

    def test_acquire_task_filters_by_doc_id(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        t1 = _make_task(job.job_id, "doc_1")
        t2 = _make_task(job.job_id, "doc_2")
        store.create_tasks([t1, t2])

        acquired = store.acquire_task(job.job_id, lease_owner="worker-1", doc_id="doc_2")
        assert acquired is not None
        assert acquired.doc_id == "doc_2"

    def test_expired_running_task_can_be_reacquired(self, store: FakeIngestJobStore) -> None:
        job = _make_job()
        store.create_job(job)
        task = _make_task(job.job_id, "doc_1")
        store.create_tasks([task])

        leased = store.acquire_task(job.job_id, lease_owner="worker-1")
        assert leased is not None
        expired = replace(leased, lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        store._tasks[expired.task_id] = expired

        reacquired = store.acquire_task(job.job_id, lease_owner="worker-2")
        assert reacquired is not None
        assert reacquired.task_id == leased.task_id
        assert reacquired.lease_owner == "worker-2"


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
