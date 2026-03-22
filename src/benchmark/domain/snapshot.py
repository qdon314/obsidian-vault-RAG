"""Corpus snapshot identity — content-addressable hash of the full corpus.

The snapshot ID is a SHA-256 of sorted ``(doc_id, content_hash)`` pairs.
Since ``rag.domain.models.Document`` has no ``content_hash`` field, we
compute SHA-256 of each document's text at snapshot time.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from typing import Protocol


class _HasDocIdAndText(Protocol):
    """Structural type for anything with doc_id and text."""

    @property
    def doc_id(self) -> str: ...
    @property
    def text(self) -> str: ...


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def compute_snapshot_id(corpus: Sequence[_HasDocIdAndText]) -> str:
    """SHA-256 of sorted (doc_id, content_hash) pairs."""
    pairs = sorted((_doc.doc_id, _content_hash(_doc.text)) for _doc in corpus)
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def verify_snapshot(corpus: Sequence[_HasDocIdAndText], expected_id: str) -> bool:
    """Confirm the current corpus matches the claimed snapshot."""
    return compute_snapshot_id(corpus) == expected_id
