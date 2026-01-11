from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from rag.domain.models import Chunk

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JsonlChunkLoader:
    """
    Loads chunks from a JSONL vector store file.

    Expected format: one JSON object per line with {"chunk": {...}, "vector": [...]}
    """

    def load_all(self, path: Path) -> list[Chunk]:
        """Load all chunks from the JSONL file."""
        chunks: list[Chunk] = []

        if not path.exists():
            logger.warning(f"Chunk store not found: {path}")
            return chunks

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    chunks.append(Chunk.from_dict(data["chunk"]))
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(f"Error loading chunk at line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(chunks)} chunks from {path}")
        return chunks

    def load_by_doc_id(self, path: Path, doc_id: str) -> list[Chunk]:
        """Load chunks for a specific document."""
        all_chunks = self.load_all(path)
        return [c for c in all_chunks if c.doc_id == doc_id]

    def get_doc_ids(self, path: Path) -> list[str]:
        """Get all unique document IDs."""
        all_chunks = self.load_all(path)
        return sorted({c.doc_id for c in all_chunks})

    def get_chunks_by_doc(self, path: Path) -> dict[str, list[Chunk]]:
        """Group all chunks by document ID."""
        all_chunks = self.load_all(path)
        by_doc: dict[str, list[Chunk]] = defaultdict(list)
        for chunk in all_chunks:
            by_doc[chunk.doc_id].append(chunk)
        return dict(by_doc)
