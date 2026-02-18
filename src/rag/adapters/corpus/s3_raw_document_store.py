"""S3-backed raw document store for the corpus-of-record.

Stores normalized documents as JSON objects in S3 with hash-prefix
sharding for even key distribution.

S3 layout::

    s3://{bucket}/{prefix}/{hash[:4]}/{hash}.json

Each JSON object contains the full document text, metadata, and
content hash for idempotency checks.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any

from rag.domain.models import Document


def _doc_s3_key(prefix: str, doc_id: str) -> str:
    """Deterministic S3 key for a raw document."""
    h = sha256(doc_id.encode("utf-8")).hexdigest()
    parts = [p for p in [prefix, h[:4], f"{h}.json"] if p]
    return "/".join(parts)


@dataclass(frozen=True, slots=True)
class S3RawDocumentStore:
    """Stores raw documents in S3 as JSON."""

    bucket: str
    prefix: str  # e.g. "corpus/{corpus_id}/raw"
    _s3: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import boto3  # type: ignore[import-untyped]

        object.__setattr__(self, "_s3", boto3.client("s3"))

    @classmethod
    def _for_test(cls, *, bucket: str, prefix: str, s3_client: Any) -> S3RawDocumentStore:
        """Create an instance with an injected S3 client (for testing)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "bucket", bucket)
        object.__setattr__(obj, "prefix", prefix)
        object.__setattr__(obj, "_s3", s3_client)
        return obj

    def store_document(
        self,
        doc: Document,
        *,
        corpus_id: str,
        content_sha256: str,
    ) -> str:
        """Persist a raw document to S3. Returns the S3 key."""
        s3_key = _doc_s3_key(self.prefix, doc.doc_id)

        body = json.dumps(
            {
                "corpus_id": corpus_id,
                "doc_id": doc.doc_id,
                "source": doc.source,
                "uri": doc.uri,
                "content_sha256": content_sha256,
                "metadata": dict(doc.metadata),
                "text": doc.text,
            },
            ensure_ascii=False,
        )

        self._s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=body.encode("utf-8"),
            ContentType="application/json",
        )
        return s3_key

    def get_document(self, key: str) -> Document:
        """Retrieve a raw document from S3 by key."""
        obj = self._s3.get_object(Bucket=self.bucket, Key=key)
        data = json.loads(obj["Body"].read().decode("utf-8"))
        return Document(
            doc_id=data["doc_id"],
            text=data["text"],
            source=data["source"],
            uri=data["uri"],
            metadata=data.get("metadata", {}),
        )
