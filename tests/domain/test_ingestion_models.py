"""Tests for ingestion domain models."""
from __future__ import annotations

import uuid
from typing import runtime_checkable

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)
from rag.ports.ingest_job_store import IngestJobStore


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
            raise AssertionError("Should be frozen")
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
            raise AssertionError("Should be frozen")
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
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestIngestJobStoreProtocol:
    def test_is_runtime_checkable(self) -> None:
        assert runtime_checkable(IngestJobStore)
