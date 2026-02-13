"""Worker: processes a single-document ingestion task.

Pulls a raw document from S3, chunks it, embeds, and writes to both
the chunk store (S3 shards) and vector store (Qdrant).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.adapters.ingestion.regulatory.enrichment import enrich_regulatory_chunks
from rag.domain.ingestion import IngestTask
from rag.domain.models import Document

if TYPE_CHECKING:
    from rag.ports import (
        Chunker,
        ChunkStore,
        Embedder,
        IngestJobStore,
        RawDocumentStore,
        VectorStore,
    )

logger = logging.getLogger(__name__)


@dataclass
class Worker:
    """Processes individual document ingestion tasks."""

    job_store: IngestJobStore
    raw_document_store: RawDocumentStore
    chunker: Chunker
    embedder: Embedder
    vector_store: VectorStore
    chunk_store: ChunkStore
    worker_id: str = "worker-0"

    def process_task(
        self,
        *,
        task: IngestTask,
        s3_raw_key: str,
        corpus_id: str,
    ) -> bool:
        """Process a single document task.

        Returns True on success, False on failure.
        """
        try:
            # 1) Load raw doc from S3
            doc: Document = self.raw_document_store.get_document(s3_raw_key)  # type: ignore[union-attr]
            if doc is None:
                raise ValueError(f"Raw document not found: {s3_raw_key}")

            # 2) Chunk
            chunks = self.chunker.chunk(doc)  # type: ignore[union-attr]
            if not chunks:
                logger.warning("No chunks produced for doc %s", task.doc_id)
                self.job_store.complete_task(task.task_id)  # type: ignore[union-attr]
                return True

            # 2b) Enrich regulatory citation metadata (no-op for non-regulatory)
            chunks = enrich_regulatory_chunks(chunks)

            # 3) Embed
            vectors = self.embedder.embed_texts([c.text for c in chunks])  # type: ignore[union-attr]

            # 4) Write to vector store
            self.vector_store.upsert(chunks=chunks, vectors=vectors)  # type: ignore[union-attr]

            # 5) Write to chunk store (dual-write)
            self.chunk_store.store_chunks(chunks)  # type: ignore[union-attr]

            # 6) Mark task succeeded
            self.job_store.complete_task(task.task_id)  # type: ignore[union-attr]

            logger.info(
                "Worker %s completed doc %s: %d chunks",
                self.worker_id,
                task.doc_id,
                len(chunks),
            )
            return True

        except Exception as exc:
            logger.exception(
                "Worker %s failed doc %s", self.worker_id, task.doc_id
            )
            err = str(exc) or exc.__class__.__name__
            self.job_store.fail_task(  # type: ignore[union-attr]
                task.task_id, error=err, retryable=True
            )
            return False
