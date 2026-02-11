"""S3 + Postgres chunk store.

Stores chunk content in S3 JSONL shard files (one per document) and
maintains a Postgres index for fast batch lookup by chunk_id.

S3 layout::

    s3://{bucket}/{prefix}/shards/{doc_id_hash[:4]}/{doc_id_hash}.jsonl

Each JSONL shard contains one JSON object per line — the full
``Chunk.to_dict()`` output.  Sharding by document provides cache
locality (related chunks are colocated in the same object).

Postgres schema::

    CREATE TABLE chunk_index (
        chunk_id    TEXT PRIMARY KEY,
        doc_id      TEXT NOT NULL,
        s3_key      TEXT NOT NULL,
        line_offset INT  NOT NULL
    );
    CREATE INDEX idx_chunk_index_doc_id ON chunk_index(doc_id);
    CREATE INDEX idx_chunk_index_s3_key ON chunk_index(s3_key);
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from rag.domain.models import Chunk

logger = logging.getLogger(__name__)


def _shard_key(prefix: str, doc_id: str) -> str:
    """Deterministic S3 key for a document's chunk shard."""
    h = sha256(doc_id.encode("utf-8")).hexdigest()
    parts = [prefix, "shards", h[:4], f"{h}.jsonl"] if prefix else ["shards", h[:4], f"{h}.jsonl"]
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class S3ChunkStore:
    """Chunk content store backed by S3 JSONL shards + Postgres index.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    prefix:
        Key prefix inside the bucket (e.g. ``"obsidian"``).
    postgres_dsn:
        Postgres connection string.
    max_s3_workers:
        Max threads for parallel S3 fetches during ``get_chunks``.
    """

    bucket: str
    prefix: str
    postgres_dsn: str
    max_s3_workers: int = 4
    _s3: Any = field(init=False, repr=False, compare=False, hash=False)
    _pool: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import boto3  # type: ignore[import-untyped]
        import psycopg2  # type: ignore[import-untyped]
        import psycopg2.pool  # type: ignore[import-untyped]

        object.__setattr__(self, "_s3", boto3.client("s3"))
        object.__setattr__(
            self,
            "_pool",
            psycopg2.pool.SimpleConnectionPool(1, 10, self.postgres_dsn),
        )

    # ------------------------------------------------------------------
    # ChunkStore protocol
    # ------------------------------------------------------------------

    def get_chunks(
        self,
        chunk_ids: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> dict[str, Chunk]:
        """Batch fetch chunks from S3 via Postgres index."""
        if not chunk_ids:
            return {}

        # 1) Postgres lookup: chunk_id -> (s3_key, line_offset)
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT chunk_id, s3_key, line_offset "
                    "FROM chunk_index WHERE chunk_id = ANY(%s)",
                    (list(chunk_ids),),
                )
                rows = cur.fetchall()
        finally:
            self._pool.putconn(conn)

        if not rows:
            logger.warning("No chunk index entries for %d requested IDs", len(chunk_ids))
            return {}

        # 2) Group by S3 key to minimise fetches
        by_key: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for chunk_id, s3_key, line_offset in rows:
            by_key[s3_key].append((chunk_id, line_offset))

        # 3) Fetch S3 objects in parallel
        result: dict[str, Chunk] = {}

        def _fetch_shard(s3_key: str, targets: list[tuple[str, int]]) -> dict[str, Chunk]:
            out: dict[str, Chunk] = {}
            try:
                obj = self._s3.get_object(Bucket=self.bucket, Key=s3_key)
                lines = obj["Body"].read().decode("utf-8").splitlines()
                for chunk_id, offset in targets:
                    if 0 <= offset < len(lines):
                        out[chunk_id] = Chunk.from_dict(json.loads(lines[offset]))
                    else:
                        logger.error(
                            "Invalid line_offset %d for chunk %s in %s",
                            offset,
                            chunk_id,
                            s3_key,
                        )
            except Exception:
                logger.exception("Failed to fetch shard %s", s3_key)
            return out

        with ThreadPoolExecutor(max_workers=self.max_s3_workers) as pool:
            futures = {
                pool.submit(_fetch_shard, key, targets): key
                for key, targets in by_key.items()
            }
            for future in as_completed(futures):
                result.update(future.result())

        return result

    def store_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        """Write chunks to S3 shards and upsert Postgres index rows."""
        if not chunks:
            return

        # Group chunks by doc_id (one shard per document)
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for ch in chunks:
            by_doc[ch.doc_id].append(ch)

        index_rows: list[tuple[str, str, str, int]] = []

        for doc_id, doc_chunks in by_doc.items():
            s3_key = _shard_key(self.prefix, doc_id)
            lines = [json.dumps(ch.to_dict(), ensure_ascii=False) for ch in doc_chunks]
            body = "\n".join(lines)

            self._s3.put_object(
                Bucket=self.bucket,
                Key=s3_key,
                Body=body.encode("utf-8"),
                ContentType="application/x-jsonlines",
            )

            for offset, ch in enumerate(doc_chunks):
                index_rows.append((ch.chunk_id, ch.doc_id, s3_key, offset))

        # Bulk upsert to Postgres
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunk_index (chunk_id, doc_id, s3_key, line_offset) "
                    "VALUES (%s, %s, %s, %s) "
                    "ON CONFLICT (chunk_id) DO UPDATE SET "
                    "  doc_id = EXCLUDED.doc_id, "
                    "  s3_key = EXCLUDED.s3_key, "
                    "  line_offset = EXCLUDED.line_offset",
                    index_rows,
                )
            conn.commit()
        finally:
            self._pool.putconn(conn)

        logger.info(
            "Stored %d chunks across %d shards", len(index_rows), len(by_doc)
        )

    def list_all_chunk_ids(
        self,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[str]:
        """List every chunk_id in the Postgres index."""
        conn = self._pool.getconn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT chunk_id FROM chunk_index ORDER BY chunk_id")
                return [row[0] for row in cur.fetchall()]
        finally:
            self._pool.putconn(conn)
