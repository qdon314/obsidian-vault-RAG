from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.adapters.context_building._shared import _estimate_tokens, _normalize_for_dedupe
from rag.domain.models import Candidate, Chunk, Citation, ContextPack


@dataclass(frozen=True, slots=True)
class SimpleContextBuilder:
    """
    Builds a context pack from candidates:
      - optional score thresholding
      - dedupe near-identical chunks
      - pack chunks into a token budget
      - produce citations for provenance
    """

    min_score: float | None = None
    max_chunks: int = 12
    dedupe: bool = True
    include_scores: bool = False

    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        token_budget: int,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextPack:
        # Use candidates in the order provided (caller handles ordering).
        # Fall back to sorting by score if needed for backwards compat.
        def candidate_key(c: Candidate) -> float:
            return c.rerank_score if c.rerank_score is not None else c.score

        ordered = sorted(candidates, key=candidate_key, reverse=True)

        chosen: list[Chunk] = []
        citations: list[Citation] = []
        seen: set[str] = set()

        tokens_used = 0
        header_tokens = _estimate_tokens("Context:\n")
        tokens_used += header_tokens

        for cand in ordered:
            score = candidate_key(cand)
            if self.min_score is not None and score < self.min_score:
                continue

            chunk = cand.chunk
            if self.dedupe:
                sig = _normalize_for_dedupe(chunk.text)[:500]
                if sig in seen:
                    continue
                seen.add(sig)

            label = f"[{len(chosen) + 1}]"
            if self.include_scores:
                label += f" score={score:.4f}"
            label += "\n"

            chunk_tokens = (
                _estimate_tokens(label) + _estimate_tokens(chunk.text) + _estimate_tokens("\n\n")
            )
            if tokens_used + chunk_tokens > token_budget:
                break

            chosen.append(chunk)
            tokens_used += chunk_tokens

            # Citation: URI typically lives in chunk.metadata (copied from Document.metadata)
            uri = str(chunk.metadata.get("uri") or chunk.metadata.get("source_uri") or "")
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    uri=uri,
                    quote=chunk.text[:240] if len(chunk.text) > 240 else chunk.text,
                    section_heading=chunk.section_heading,
                    section_path=chunk.section_path,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                    metadata={"rank": len(chosen), "score": score},
                )
            )

            if len(chosen) >= self.max_chunks:
                break

        rendered = self._render_context(
            chosen, ordered_scores=ordered[: len(chosen)] if self.include_scores else None
        )

        return ContextPack(
            query=query,
            chunks=tuple(chosen),
            rendered_context=rendered,
            citations=tuple(citations),
            token_budget=token_budget,
            metadata={**(dict(metadata) if metadata else {}), "tokens_used_est": tokens_used},
        )

    def _render_context(
        self, chunks: Sequence[Chunk], ordered_scores: Sequence[Candidate] | None = None
    ) -> str:
        lines: list[str] = []
        lines.append("CONTEXT:\n")

        for i, ch in enumerate(chunks, start=1):
            lines.append(f"[{i}]")
            # Include a tiny provenance header if helpful
            title = ch.metadata.get("title")
            uri = ch.metadata.get("uri") or ch.metadata.get("source_uri")
            if title or uri:
                lines.append(f"Source: {title or ''} {uri or ''}".strip())

            lines.append(ch.text.strip())
            lines.append("")  # blank line

        return "\n".join(lines).strip() + "\n"
