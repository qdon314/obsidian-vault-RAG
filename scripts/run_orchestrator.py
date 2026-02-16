"""CLI: Run the full distributed ingestion lifecycle.

Combines: enumerate docs → poll for worker completion → finalize job.
Designed to run as a single ECS task (ingest-orchestrator).

Usage:
    ./scripts/py scripts/run_orchestrator.py \
        --corpus /path/to/vault \
        --corpus-id regulations_v1 \
        --index-name regulatory
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import boto3  # type: ignore[import-untyped]
import psycopg2  # type: ignore[import-untyped]
from dotenv import load_dotenv

from rag import settings
from rag.adapters.corpus.s3_raw_document_store import S3RawDocumentStore
from rag.adapters.persistence.postgres_ingest_job_store import PostgresIngestJobStore
from rag.adapters.queue.sqs_task_queue import SQSTaskQueue
from rag.app.container import build_container
from rag.app.ingestion.enumerator import Enumerator
from rag.app.ingestion.orchestrator import poll_until_complete
from rag.domain.index_manifest import IndexManifest
from rag.domain.ingestion import JobStatus
from rag.domain.models import Document

log = logging.getLogger("orchestrator")


def _count_chunks(dsn: str) -> int:
    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM chunk_index")
            return cur.fetchone()[0]  # type: ignore[index]
    finally:
        conn.close()


def _download_s3_prefix(bucket: str, prefix: str, local_dir: Path) -> list[Path]:
    """Download all objects under an S3 prefix to a local directory."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: list[Path] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix) :].lstrip("/")
            if not rel:
                continue
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            downloaded.append(local_path)
    return downloaded


def _canonicalize_s3_docs(
    *,
    docs: list[Document],
    local_root: Path,
    bucket: str,
    prefix: str,
) -> list[Document]:
    """Rewrite temp local URIs/doc_ids to stable s3:// URIs/doc_ids."""
    root = local_root.resolve()
    prefix = prefix.strip("/")
    canonical: list[Document] = []

    for doc in docs:
        path = Path(doc.uri).resolve()
        try:
            rel = path.relative_to(root).as_posix()
        except ValueError:
            # Defensive fallback: keep basename-stable key if path is outside root.
            rel = path.name
        key = "/".join(part for part in (prefix, rel) if part)
        s3_uri = f"s3://{bucket}/{key}"
        text_hash = sha256(doc.text.encode("utf-8")).hexdigest()
        doc_id = sha256(f"{s3_uri}|{text_hash}".encode()).hexdigest()
        metadata = dict(doc.metadata)
        metadata["uri"] = s3_uri
        metadata["s3_key"] = key
        metadata["source_uri"] = s3_uri
        canonical.append(replace(doc, uri=s3_uri, doc_id=doc_id, metadata=metadata))

    return canonical


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full ingestion lifecycle.")
    ap.add_argument("--corpus", required=True, help="Path to corpus directory.")
    ap.add_argument(
        "--corpus-s3-prefix",
        type=str,
        default=None,
        help="S3 prefix containing corpus files to ingest (overrides --corpus path input).",
    )
    ap.add_argument(
        "--corpus-s3-bucket",
        type=str,
        default=None,
        help="S3 bucket for --corpus-s3-prefix (default: distributed_ingestion.corpus_s3_bucket).",
    )
    ap.add_argument("--corpus-id", required=True, help="Unique corpus identifier.")
    ap.add_argument("--index-name", required=True, help="Index name for manifest.")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit docs (0=all).")
    ap.add_argument("--qdrant-collection", type=str, default=None)
    ap.add_argument("--poll-interval", type=float, default=30.0, help="Poll interval in seconds.")
    ap.add_argument("--timeout", type=float, default=7200.0, help="Max wait time in seconds.")
    ap.add_argument("--force-finalize", action="store_true", help="Finalize even if tasks failed.")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[orchestrator] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()

    if not cfg.distributed_ingestion.enabled:
        log.error("distributed_ingestion.enabled must be true")
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

    # ── Phase 1: Enumerate ─────────────────────────────────────────
    log.info("Phase 1: Enumerating documents...")

    container = build_container()

    if args.corpus_s3_prefix:
        bucket = args.corpus_s3_bucket or cfg.distributed_ingestion.corpus_s3_bucket
        if not bucket:
            log.error("No S3 bucket configured for --corpus-s3-prefix")
            raise SystemExit(1)
        prefix = args.corpus_s3_prefix.strip("/")
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_root = Path(tmpdir) / "corpus"
            corpus_root.mkdir(parents=True, exist_ok=True)
            downloaded = _download_s3_prefix(bucket, prefix, corpus_root)
            log.info(
                "Downloaded %d corpus files from s3://%s/%s to %s",
                len(downloaded),
                bucket,
                prefix,
                corpus_root,
            )
            docs, _report = container.ingestor.ingest([str(corpus_root)])
            docs = _canonicalize_s3_docs(
                docs=docs,
                local_root=corpus_root,
                bucket=bucket,
                prefix=prefix,
            )
    else:
        vault_root = Path(args.corpus).expanduser().resolve()
        docs, _report = container.ingestor.ingest([str(vault_root)])

    if args.max_docs > 0:
        docs = docs[: args.max_docs]

    if len(docs) == 0:
        if args.corpus_s3_prefix:
            log.error(
                "No documents found under s3://%s/%s",
                args.corpus_s3_bucket or cfg.distributed_ingestion.corpus_s3_bucket,
                args.corpus_s3_prefix.strip("/"),
            )
        else:
            log.error("No documents found under corpus path: %s", args.corpus)
        raise SystemExit(1)

    log.info("Ingested %d docs, now creating job...", len(docs))

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

    log.info("Job %s created (status=%s, tasks=%d)", job.job_id, job.status.value, len(docs))

    # ── Phase 2: Poll ──────────────────────────────────────────────
    log.info("Phase 2: Polling for task completion...")

    result = poll_until_complete(
        job_id=job.job_id,
        job_store=job_store,
        total_tasks=len(docs),
        poll_interval_s=args.poll_interval,
        timeout_s=args.timeout,
    )

    if result.timed_out:
        log.error(
            "Timed out: %d succeeded, %d failed, %d still in-flight",
            result.succeeded, result.failed,
            result.pending + result.running + result.retryable,
        )
        job_store.update_job_status(job.job_id, JobStatus.FAILED, stats={
            "reason": "timeout",
            "succeeded": result.succeeded,
            "failed": result.failed,
        })
        raise SystemExit(1)

    log.info("All tasks terminal: %d succeeded, %d failed", result.succeeded, result.failed)

    if result.failed > 0 and not args.force_finalize:
        log.error(
            "%d tasks failed. Use --force-finalize to proceed anyway.", result.failed,
        )
        job_store.update_job_status(job.job_id, JobStatus.FAILED, stats={
            "succeeded": result.succeeded,
            "failed": result.failed,
        })
        raise SystemExit(1)

    # ── Phase 3: Finalize ──────────────────────────────────────────
    log.info("Phase 3: Finalizing job...")

    chunk_count = _count_chunks(cfg.distributed_ingestion.postgres_dsn)
    log.info("Chunk index: %d chunks", chunk_count)

    manifest = IndexManifest.create(
        index_name=args.index_name,
        corpus=args.corpus_id,
        doc_count=result.succeeded,
        chunk_count=chunk_count,
        chunking={"backend": cfg.chunking.backend},
        embedding={"model": cfg.embeddings.model},
        store={
            "type": "s3+qdrant",
            "bucket": cfg.distributed_ingestion.corpus_s3_bucket,
            "collection": args.qdrant_collection or cfg.vectorstore.qdrant_collection,
        },
    )

    bucket = cfg.distributed_ingestion.corpus_s3_bucket
    corpus_prefix = (cfg.distributed_ingestion.corpus_s3_prefix or "").strip("/")
    manifests_prefix = os.environ.get("RAG_MANIFESTS_S3_PREFIX", "manifests").strip("/")
    key_parts = [part for part in (corpus_prefix, manifests_prefix, index_id, "manifest.json") if part]
    s3_key = "/".join(key_parts)

    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(manifest.to_dict(), indent=2).encode("utf-8"),
        ContentType="application/json",
    )
    log.info("Uploaded manifest to s3://%s/%s", bucket, s3_key)

    job_store.update_job_status(
        job.job_id,
        JobStatus.COMPLETED,
        stats={
            "doc_count": result.succeeded,
            "chunk_count": chunk_count,
            "failed_tasks": result.failed,
            "manifest_s3_key": s3_key,
        },
    )
    log.info("Job %s marked COMPLETED", job.job_id)


if __name__ == "__main__":
    main()
