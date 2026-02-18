"""CLI: Finalize a distributed ingestion job.

Validates all tasks succeeded, counts chunks, builds an IndexManifest,
uploads it to S3, and marks the job COMPLETED in Postgres.

Usage:
    ./scripts/py scripts/finalize_job.py --job-id <uuid>
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid

import boto3  # type: ignore[import-untyped]
import psycopg2  # type: ignore[import-untyped]
from dotenv import load_dotenv

from rag import settings
from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
from rag.domain.index_manifest import IndexManifest
from rag.domain.ingestion import JobStatus, TaskStatus

log = logging.getLogger("finalize-job")


def _count_chunks(dsn: str) -> int:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunk_index")
            return cur.fetchone()[0]  # type: ignore[index]
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Finalize a distributed ingestion job.")
    ap.add_argument("--job-id", required=True, help="Job UUID from start_ingestion.")
    ap.add_argument("--force", action="store_true", help="Finalize even if some tasks failed.")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[finalize] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()
    if cfg.distributed_ingestion.postgres_dsn is None:
        log.error("distributed_ingestion.postgres_dsn must be set")
        raise SystemExit(1)

    job_store = PostgresIngestJobStore(postgres_dsn=cfg.distributed_ingestion.postgres_dsn)
    job_id = uuid.UUID(args.job_id)

    # 1) Fetch job
    job = job_store.get_job(job_id)
    if job is None:
        log.error("Job %s not found", job_id)
        raise SystemExit(1)
    log.info("Job %s: status=%s, index_id=%s", job.job_id, job.status.value, job.index_id)

    # 2) Check task counts
    counts = job_store.get_task_counts(job_id)
    total = sum(counts.values())
    succeeded = counts.get(TaskStatus.SUCCEEDED, 0)
    failed = counts.get(TaskStatus.FAILED, 0)
    retryable = counts.get(TaskStatus.RETRYABLE, 0)
    pending = counts.get(TaskStatus.PENDING, 0)

    log.info(
        "Tasks: %d total, %d succeeded, %d failed, %d retryable, %d pending",
        total,
        succeeded,
        failed,
        retryable,
        pending,
    )

    if succeeded < total and not args.force:
        log.error(
            "Not all tasks succeeded (%d/%d). Use --force to finalize anyway.",
            succeeded,
            total,
        )
        raise SystemExit(1)

    # 3) Count chunks
    chunk_count = _count_chunks(cfg.distributed_ingestion.postgres_dsn)
    log.info("Chunk index: %d chunks", chunk_count)

    # 4) Build manifest
    manifest = IndexManifest.create(
        index_name=job.index_id.rsplit("_", 1)[0] if "_" in job.index_id else job.index_id,
        corpus=job.corpus_id,
        doc_count=succeeded,
        chunk_count=chunk_count,
        chunking={"backend": job.chunking_strategy},
        embedding={"model": job.embedder_model},
        store={
            "type": "s3+qdrant",
            "bucket": cfg.distributed_ingestion.corpus_s3_bucket,
            "collection": job.qdrant_collection,
        },
    )

    # 5) Upload to S3
    bucket = cfg.distributed_ingestion.corpus_s3_bucket
    prefix = cfg.distributed_ingestion.corpus_s3_prefix or ""
    s3_key = (
        f"{prefix}/manifests/{job.index_id}/manifest.json"
        if prefix
        else f"manifests/{job.index_id}/manifest.json"
    )

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(manifest.to_dict(), indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info("Uploaded manifest to s3://%s/%s", bucket, s3_key)

    # 6) Mark job completed
    job_store.update_job_status(
        job_id,
        JobStatus.COMPLETED,
        stats={
            "doc_count": succeeded,
            "chunk_count": chunk_count,
            "failed_tasks": failed,
            "manifest_s3_key": s3_key,
        },
    )
    log.info("Job %s marked COMPLETED", job_id)


if __name__ == "__main__":
    main()
