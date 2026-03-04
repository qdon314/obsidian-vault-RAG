from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from rag.domain.models import CompressionResult, ContextPack


class PromptCompressor(Protocol):
    """Compress a ContextPack before it reaches the Generator.

    Implementations must return a CompressionResult containing:
    - context_pack: the (possibly compressed) ContextPack to pass to the Generator
    - observability metrics (tokens_before/after, savings_pct, latency_ms, etc.)

    Implementations should be fail-open: on error, return the original ContextPack
    with successful=False rather than raising.
    """

    @property
    def name(self) -> str: ...

    def compress(
        self,
        context_pack: ContextPack,
        *,
        query: str,
        metadata: Mapping[str, object] | None = None,
    ) -> CompressionResult: ...
