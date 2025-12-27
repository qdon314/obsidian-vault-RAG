from __future__ import annotations

from typing import Mapping, Protocol

from rag.domain.models import Answer, ContextPack


class Generator(Protocol):
    """
    Produces an answer from the query + context.
    """

    @property
    def model_name(self) -> str: ...

    def generate(
        self,
        query: str,
        context: ContextPack,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> Answer:
        ...
