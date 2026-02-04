from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256

Vector = list[float]


@dataclass(frozen=True, slots=True)
class DummyEmbedder:
    """
    Deterministic fake embeddings for wiring tests.
    Not semantically meaningful, but stable across runs.
    """

    dim: int = 128
    model: str = "dummy-embedder-v1"

    @property
    def model_name(self) -> str:
        return self.model

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Vector]:
        out: list[Vector] = []
        for text in texts:
            text_hash = sha256(text.encode("utf-8")).digest()
            # expand digest to dim floats in [-1,1]
            vector: list[float] = []
            i = 0
            while len(vector) < self.dim:
                block = text_hash[i % len(text_hash)]
                vector.append((block / 127.5) - 1.0)  # map byte to [-1,1]
                i += 1
            out.append(vector)
        return out
