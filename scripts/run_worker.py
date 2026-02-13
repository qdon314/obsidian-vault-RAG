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
from rag.app.container import ContainerOverrides, build_container
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
    ap.add_argument(
        "--qdrant-collection",
        type=str,
        default=None,
        help="Qdrant collection (default: from settings).",
    )
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[worker] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()

    if not cfg.distributed_ingestion.enabled:
        log.error("distributed_ingestion.enabled must be true in settings.toml")
        raise SystemExit(1)

    if cfg.distributed_ingestion.postgres_dsn is None:
        log.error("distributed_ingestion.postgres_dsn must be set")
        raise SystemExit(1)
    if cfg.distributed_ingestion.corpus_s3_bucket is None:
        log.error("distributed_ingestion.corpus_s3_bucket must be set")
        raise SystemExit(1)
    if cfg.distributed_ingestion.sqs_queue_url is None:
        log.error("distributed_ingestion.sqs_queue_url must be set")
        raise SystemExit(1)

    overrides = ContainerOverrides(
        qdrant_collection=args.qdrant_collection,
    ) if args.qdrant_collection else None
    container = build_container(overrides=overrides)

    from rag.adapters.chunk_storage.s3_chunk_store import S3ChunkStore
    from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
    from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
    from rag.adapters.queue.sqs_task_queue import SQSTaskQueue

    job_store = PostgresIngestJobStore(postgres_dsn=cfg.distributed_ingestion.postgres_dsn)
    raw_store = S3RawDocumentStore(
        bucket=cfg.distributed_ingestion.corpus_s3_bucket,
        prefix=cfg.distributed_ingestion.corpus_s3_prefix,
    )
    queue = SQSTaskQueue(queue_url=cfg.distributed_ingestion.sqs_queue_url)

    job_store.ensure_schema()
    if isinstance(container.chunk_store, S3ChunkStore):
        container.chunk_store.ensure_schema()

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
