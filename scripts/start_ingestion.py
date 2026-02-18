"""CLI: Start a distributed ingestion job.

Usage:
    ./scripts/py scripts/start_ingestion.py \
        --corpus /path/to/vault \
        --corpus-id regulations_v1 \
        --index-name regulations
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from rag import settings
from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
from rag.adapters.queue.sqs_task_queue import SQSTaskQueue
from rag.app.container import build_container
from rag.app.ingestion.enumerator import Enumerator

log = logging.getLogger("start_ingestion")


def main() -> None:
    ap = argparse.ArgumentParser(description="Start a distributed ingestion job.")
    ap.add_argument("--corpus", required=True, help="Path to corpus directory.")
    ap.add_argument("--corpus-id", required=True, help="Unique corpus identifier.")
    ap.add_argument("--index-name", required=True, help="Index name for manifest.")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs (0=all).")
    ap.add_argument(
        "--qdrant-collection",
        type=str,
        default=None,
        help="Qdrant collection (default: from settings).",
    )
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[ingest] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()

    if not cfg.distributed_ingestion.enabled:
        log.error("distributed_ingestion.enabled must be true in settings.toml")
        raise SystemExit(1)

    container = build_container()

    # Ingest documents from local filesystem
    vault_root = Path(args.corpus).expanduser().resolve()
    docs, _report = container.ingestor.ingest([str(vault_root)])

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    log.info("Ingested %d docs, now enumerating...", len(docs))

    if cfg.distributed_ingestion.postgres_dsn is None:
        log.error("distributed_ingestion.postgres_dsn must be set")
        raise SystemExit(1)
    if cfg.distributed_ingestion.corpus_s3_bucket is None:
        log.error("distributed_ingestion.corpus_s3_bucket must be set")
        raise SystemExit(1)
    if cfg.distributed_ingestion.sqs_queue_url is None:
        log.error("distributed_ingestion.sqs_queue_url must be set")
        raise SystemExit(1)

    job_store = PostgresIngestJobStore(postgres_dsn=cfg.distributed_ingestion.postgres_dsn)
    job_store.ensure_schema()

    raw_store = S3RawDocumentStore(
        bucket=cfg.distributed_ingestion.corpus_s3_bucket,
        prefix=f"{cfg.distributed_ingestion.corpus_s3_prefix}/{args.corpus_id}/raw",
    )
    queue = SQSTaskQueue(queue_url=cfg.distributed_ingestion.sqs_queue_url)

    enumerator = Enumerator(
        job_store=job_store,
        raw_document_store=raw_store,
        task_queue=queue,
    )

    # Build index_id
    from datetime import UTC, datetime

    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%S")
    index_id = f"{args.index_name}_{cfg.chunking.backend}_{cfg.embeddings.model}_{ts}"

    job = enumerator.enumerate(
        docs=docs,
        corpus_id=args.corpus_id,
        index_id=index_id,
        chunking_strategy=cfg.chunking.backend,
        embedder_model=cfg.embeddings.model,
        qdrant_collection=args.qdrant_collection or cfg.vectorstore.qdrant_collection,
    )

    log.info("Job created: %s (status=%s, docs=%d)", job.job_id, job.status.value, len(docs))


if __name__ == "__main__":
    main()
