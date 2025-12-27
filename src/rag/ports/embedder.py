from __future__ import annotations

from typing import Mapping, Protocol, Sequence

Vector = list[float]


class Embedder(Protocol):
    """
    Turns text into dense vectors.

    Keep this intentionally small: you can later add batch sizing, retries, etc.
    """

    @property
    def model_name(self) -> str: ...

    def embed_texts(self, texts: Sequence[str], *, metadata: Mapping[str, object] | None = None) -> list[Vector]:
        ...
