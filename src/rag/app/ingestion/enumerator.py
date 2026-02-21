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
from typing import TYPE_CHECKING

from rag.domain.ingestion import (
    DocumentRecord,
    IngestJob,
    IngestTask,
    JobStatus,
    TaskStatus,
)
from rag.domain.models import Document

if TYPE_CHECKING:
    from rag.ports import IngestJobStore, RawDocumentStore, TaskQueue

logger = logging.getLogger(__name__)


@dataclass
class Enumerator:
    """Enumerates documents and dispatches ingestion tasks."""

    job_store: IngestJobStore
    raw_document_store: RawDocumentStore
    task_queue: TaskQueue

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

        # 1) Store raw docs in S3 and record in DB (skip unchanged)
        tasks: list[IngestTask] = []
        messages: list[dict[str, str]] = []
        skipped = 0

        for doc in docs:
            content_hash = sha256(doc.text.encode("utf-8")).hexdigest()

            existing = self.job_store.get_document(corpus_id, doc.doc_id)  # type: ignore[union-attr]
            if existing is not None and existing.content_sha256 == content_hash:
                skipped += 1
                continue

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

        logger.info(
            "Enumerated %d docs: %d to process, %d unchanged (skipped)",
            len(docs),
            len(tasks),
            skipped,
        )

        # All docs unchanged — nothing to do
        if not tasks:
            from dataclasses import replace

            self.job_store.update_job_status(  # type: ignore[union-attr]
                job.job_id, JobStatus.COMPLETED, stats={"docs": 0, "skipped": skipped}
            )
            return replace(job, status=JobStatus.COMPLETED)

        # 2) Bulk create tasks in DB
        self.job_store.create_tasks(tasks)  # type: ignore[union-attr]

        # 3) Enqueue messages for workers
        self.task_queue.send_batch(messages)  # type: ignore[union-attr]

        # 4) Transition job to RUNNING
        self.job_store.update_job_status(job.job_id, JobStatus.RUNNING)  # type: ignore[union-attr]

        from dataclasses import replace

        return replace(job, status=JobStatus.RUNNING)
