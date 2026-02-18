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
        doc_id: str | None = None,
    ) -> IngestTask | None:
        now = datetime.now(UTC)
        for tid, t in self._tasks.items():
            if t.job_id != job_id:
                continue
            if doc_id is not None and t.doc_id != doc_id:
                continue
            is_ready = t.status in (TaskStatus.PENDING, TaskStatus.RETRYABLE)
            is_expired_running = (
                t.status == TaskStatus.RUNNING
                and t.lease_expires_at is not None
                and t.lease_expires_at < now
            )
            if not (is_ready or is_expired_running):
                continue
            updated = replace(
                t,
                status=TaskStatus.RUNNING,
                lease_owner=lease_owner,
                lease_expires_at=now + timedelta(seconds=lease_duration_s),
                attempt=t.attempt + 1,
                updated_at=now,
            )
            self._tasks[tid] = updated
            return updated
        return None

    def complete_task(self, task_id: uuid.UUID) -> None:
        t = self._tasks[task_id]
        self._tasks[task_id] = replace(t, status=TaskStatus.SUCCEEDED, updated_at=datetime.now(UTC))

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
