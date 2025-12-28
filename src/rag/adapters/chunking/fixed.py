from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from rag.domain.models import Chunk, Document


@dataclass(frozen=True, slots=True)
class FixedChunker:
    """
    Simple character-based chunker (good enough to validate architecture).

    Strategy:
      - split doc.text into chunks of size `chunk_size`
      - optional overlap in chars
      - emits stable-ish chunk ids (you'll replace with hashing later)
    """
    chunk_size: int = 1200
    overlap: int = 150
    strategy_name: str = "fixed_chars_v1"

    def chunk(self, doc: Document, *, metadata: Optional[Mapping[str, object]] = None) -> list[Chunk]:
        text = doc.text or ""
        if not text.strip():
            return []

        step = max(1, self.chunk_size - self.overlap)
        chunks: list[Chunk] = []

        i = 0
        chunk_index = 0
        while i < len(text):
            start = i
            end = min(len(text), i + self.chunk_size)
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_id = f"{doc.doc_id}:{self.strategy_name}:{chunk_index}:{start}-{end}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        start_char=start,
                        end_char=end,
                        metadata={
                            **(dict(doc.metadata) if doc.metadata else {}),
                            "chunking_strategy": self.strategy_name,
                            "chunk_size": self.chunk_size,
                            "overlap": self.overlap,
                            **(dict(metadata) if metadata else {}),
                        },
                    )
                )
                chunk_index += 1

            i += step

        return chunks
