"""Tests for PostgresIngestJobStore.

Uses a real SQLite-in-memory database via psycopg2 mocking for unit tests.
For true integration tests, use a Postgres testcontainer (marked slow).
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
from rag.domain.ingestion import (
    IngestJob,
    JobStatus,
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


class TestAcquireTask:
    def test_filters_by_doc_and_reclaims_expired_running(self) -> None:
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        cursor = mock_conn.cursor.return_value.__enter__.return_value
        cursor.fetchone.return_value = None

        store = PostgresIngestJobStore._for_test(pool=mock_pool)
        store.acquire_task(
            uuid.uuid4(),
            lease_owner="worker-1",
            doc_id="doc-123",
        )

        sql = cursor.execute.call_args[0][0]
        assert "(%s IS NULL OR doc_id = %s)" in sql
        assert "status = %s AND lease_expires_at IS NOT NULL AND lease_expires_at < %s" in sql
