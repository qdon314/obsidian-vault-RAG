from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

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
    timeout: float = 30.0
    max_retries: int = 3
    _client: OpenAI = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_client",
            OpenAI(
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            ),
        )

    @property
    def model_name(self) -> str:
        return self.model

    def embed_texts(
        self,
        texts: Sequence[str],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> list[Vector]:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        # OpenAI returns embeddings in resp.data in order
        return [list(item.embedding) for item in resp.data]
