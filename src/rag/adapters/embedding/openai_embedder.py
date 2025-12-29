from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

# Requires: pip install openai, and set OPENAI_API_KEY in env
from openai import OpenAI


Vector = list[float]


@dataclass(frozen=True, slots=True)
class OpenAIEmbedder:
    """
    OpenAI embeddings adapter.

    Notes:
      - uses the official OpenAI Python client
      - returns List[List[float]] in the same order as inputs
    """
    api_key: str
    model: str = "text-embedding-3-small"

    @property
    def model_name(self) -> str:
        return self.model

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> list[Vector]:
        client = OpenAI(api_key=self.api_key)
        resp = client.embeddings.create(model=self.model, input=list(texts))
        # OpenAI returns embeddings in resp.data in order
        return [list(item.embedding) for item in resp.data]
