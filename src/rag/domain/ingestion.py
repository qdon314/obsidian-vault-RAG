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
