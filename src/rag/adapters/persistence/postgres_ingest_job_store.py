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
                        "UPDATE ingest_jobs SET status = %s, updated_at = %s WHERE job_id = %s",
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
        doc_id: str | None = None,
    ) -> IngestTask | None:
        now = datetime.now(UTC)
        expires = now + timedelta(seconds=lease_duration_s)
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
                    "  WHERE job_id = %s "
                    "    AND (%s IS NULL OR doc_id = %s) "
                    "    AND ("
                    "      status IN (%s, %s) "
                    "      OR (status = %s AND lease_expires_at IS NOT NULL AND lease_expires_at < %s)"
                    "    ) "
                    "  ORDER BY updated_at ASC LIMIT 1 "
                    "  FOR UPDATE SKIP LOCKED"
                    ") "
                    "RETURNING task_id, job_id, doc_id, status, attempt, "
                    "          lease_owner, lease_expires_at, last_error, updated_at",
                    (
                        TaskStatus.RUNNING.value,
                        lease_owner,
                        expires,
                        now,
                        str(job_id),
                        doc_id,
                        doc_id,
                        TaskStatus.PENDING.value,
                        TaskStatus.RETRYABLE.value,
                        TaskStatus.RUNNING.value,
                        now,
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
                    "UPDATE ingest_tasks SET status = %s, updated_at = %s WHERE task_id = %s",
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
                    "SELECT status, COUNT(*) FROM ingest_tasks WHERE job_id = %s GROUP BY status",
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
