from __future__ import annotations

from collections.abc import Mapping

from rag.domain.models import CompressionResult, ContextPack


class NoOpCompressor:
    """Pass-through compressor used when compression is disabled.

    Returns the original ContextPack unchanged with zero savings recorded.
    """

    @property
    def name(self) -> str:
        return "noop"

    def compress(
        self,
        context_pack: ContextPack,
        *,
        query: str,
        metadata: Mapping[str, object] | None = None,
    ) -> CompressionResult:
        tokens = int(context_pack.metadata.get("tokens_used_est", 0))
        return CompressionResult(
            context_pack=context_pack,
            successful=True,
            tokens_before=tokens,
            tokens_after=tokens,
            savings_pct=0.0,
            latency_ms=0,
            adapter="noop",
        )
