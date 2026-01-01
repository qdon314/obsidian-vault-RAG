from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from rag.domain.models import Candidate, Chunk
from rag.utils.json_sanitize import json_sanitize

Vector = list[float]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return sqrt(sum(x * x for x in a)) or 1.0


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def _chunk_to_dict(ch: Chunk) -> dict[str, Any]:
    return {
        "chunk_id": ch.chunk_id,
        "doc_id": ch.doc_id,
        "text": ch.text,
        "chunk_index": ch.chunk_index,
        "start_char": ch.start_char,
        "end_char": ch.end_char,
        "section_heading": ch.section_heading,
        "section_path": ch.section_path,
        "language": ch.language,
        "metadata": dict(ch.metadata),
    }


def _chunk_from_dict(d: Mapping[str, Any]) -> Chunk:
    return Chunk(
        chunk_id=str(d["chunk_id"]),
        doc_id=str(d["doc_id"]),
        text=str(d["text"]),
        chunk_index=int(d["chunk_index"]),
        start_char=d.get("start_char"),
        end_char=d.get("end_char"),
        section_heading=d.get("section_heading"),
        section_path=d.get("section_path"),
        language=d.get("language"),
        metadata=d.get("metadata", {}) or {},
    )


@dataclass(slots=True)
class JsonlVectorStore:
    """
    Disk-persisted store, loaded into memory for search.

    Files:
      - chunks.jsonl  (one row per chunk+vector)
    """
    path: Path
    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[Vector] = field(default_factory=list)

    @property
    def data_file(self) -> Path:
        return self.path / "chunks.jsonl"

    def load(self) -> None:
        self._chunks.clear()
        self._vectors.clear()

        if not self.data_file.exists():
            return

        for i, line in enumerate(self.data_file.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                snippet = line[:200].replace("\n", "\\n")
                raise RuntimeError(
                    f"Invalid JSON on {self.data_file} line {i}: {e}\n"
                    f"Snippet: {snippet}"
                ) from e
            self._chunks.append(_chunk_from_dict(row["chunk"]))
            self._vectors.append(list(row["vector"]))

    def save(self) -> None:
        """
        Persist the in-memory chunks+vectors to disk as JSONL.

        Portfolio-grade behaviors:
        - sanitize the entire row so everything is JSON-serializable
        - write to a temp file first
        - atomic replace to avoid corrupted/truncated files
        """

        self.path.mkdir(parents=True, exist_ok=True)

        tmp_file = self.data_file.with_suffix(".jsonl.tmp")

        with tmp_file.open("w", encoding="utf-8") as f:
            for ch, vec in zip(self._chunks, self._vectors):
                row = {"chunk": _chunk_to_dict(ch), "vector": vec}
                safe_row = json_sanitize(row)
                f.write(json.dumps(safe_row, ensure_ascii=False))
                f.write("\n")

            # Ensure buffers are flushed to OS before replace (extra safety)
            f.flush()

        # Atomic replace (macOS/Linux). On Windows this also works in most cases.
        tmp_file.replace(self.data_file)


    def upsert(
        self,
        *,
        chunks: Sequence[Chunk],
        vectors: Sequence[Vector],
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        self._chunks.extend(list(chunks))
        self._vectors.extend([list(v) for v in vectors])

    def search(
        self,
        *,
        query_vector: Vector,
        top_k: int,
        filters: Optional[Mapping[str, object]] = None,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> list[Candidate]:
        def allowed(c: Chunk) -> bool:
            if not filters:
                return True
            for k, v in filters.items():
                if c.metadata.get(k) != v:
                    return False
            return True

        scored: list[Candidate] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            if not allowed(chunk):
                continue
            score = _cosine(query_vector, vec)
            scored.append(Candidate(chunk=chunk, score=score))

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:top_k]

    def count(self) -> int:
        return len(self._chunks)
