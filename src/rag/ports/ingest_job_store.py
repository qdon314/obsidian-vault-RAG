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
        self,
        job_id: uuid.UUID,
        *,
        lease_owner: str,
        lease_duration_s: int = 300,
        doc_id: str | None = None,
    ) -> IngestTask | None:
        """Atomically claim the next task for a job.

        Claims tasks in PENDING/RETRYABLE, plus RUNNING tasks whose lease
        has expired. When ``doc_id`` is provided, only that document's task
        is eligible.
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
