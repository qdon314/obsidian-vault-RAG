from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Mapping, Optional, Sequence

from rag.ports import Embedder

Vector = list[float]


def _key_for_text(model_name: str, text: str) -> str:
    h = sha256(text.encode("utf-8")).hexdigest()
    return sha256(f"{model_name}|{h}".encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class CachedEmbedder:
    """
    Wraps any Embedder with a disk cache (SQLite).
    Keyed by (model_name, text hash) so it is stable across runs.

    Great for:
      - repeatable experiments
      - avoiding repeated embedding costs
    """
    embedder: Embedder
    db_path: Path

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS embeddings (
              key TEXT PRIMARY KEY,
              model TEXT NOT NULL,
              vector_json TEXT NOT NULL
            )
            """
        )
        return conn

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Vector]:
        keys = [_key_for_text(self.model_name, t) for t in texts]

        with self._connect() as conn:
            found: dict[str, Vector] = {}
            if keys:
                qmarks = ",".join(["?"] * len(keys))
                cur = conn.execute(f"SELECT key, vector_json FROM embeddings WHERE key IN ({qmarks})", keys)
                for k, vjson in cur.fetchall():
                    found[str(k)] = list(json.loads(vjson))

            missing_idx: list[int] = [i for i, k in enumerate(keys) if k not in found]
            if missing_idx:
                missing_texts = [texts[i] for i in missing_idx]
                new_vecs = self.embedder.embed_texts(missing_texts, metadata=metadata)

                rows = []
                for i, vec in zip(missing_idx, new_vecs):
                    k = keys[i]
                    found[k] = vec
                    rows.append((k, self.model_name, json.dumps(vec)))

                conn.executemany("INSERT OR REPLACE INTO embeddings(key, model, vector_json) VALUES (?, ?, ?)", rows)

        return [found[k] for k in keys]
