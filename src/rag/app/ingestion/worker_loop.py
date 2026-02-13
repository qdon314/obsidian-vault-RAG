"""SQS worker loop: poll messages, process tasks, ack/nack.

This module provides the ``process_one_message`` function (testable
without real SQS) and a ``run_loop`` entry point for the worker
service.
"""
from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rag.app.ingestion.worker import Worker

if TYPE_CHECKING:
    from rag.ports import IngestJobStore, TaskQueue

logger = logging.getLogger(__name__)


def process_one_message(
    *,
    message: dict[str, Any],
    worker: Worker,
    job_store: IngestJobStore,
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
    task = job_store.acquire_task(  # type: ignore[union-attr]
        job_id,
        lease_owner=worker.worker_id,
        doc_id=doc_id,
    )
    if task is None:
        # No claimable task — might already be processed. Ack to remove from queue.
        logger.info("No claimable task for doc %s — acking (already done?)", doc_id)
        on_ack(receipt_handle)
        return True
    if task.doc_id != doc_id:
        # Defensive guard in case a store implementation ignores doc_id filtering.
        logger.error(
            "Acquired mismatched task doc_id=%s for message doc_id=%s — nacking",
            task.doc_id,
            doc_id,
        )
        job_store.fail_task(task.task_id, error="leased task/message doc mismatch", retryable=True)  # type: ignore[union-attr]
        on_nack(receipt_handle)
        return False

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
    job_store: IngestJobStore,
    task_queue: TaskQueue,
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
