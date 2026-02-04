"""
Proposition-aware context builder for RAG query pipelines.

This module extends the context-building step to handle proposition chunks
produced by ``ObsidianPropositionChunker``.  Propositions are short
(~50-150 chars), so embedding them directly as context would waste the
LLM's token budget.  Instead, this builder *expands* each proposition
back to its parent passage before rendering, giving the LLM enough
surrounding context to answer accurately.

Expansion Behavior
==================

When ``expand_propositions=True`` and ``expansion_mode="passage"``
(both defaults):

- If a chunk has ``chunk_kind="proposition"`` and carries a
  ``parent_passage_text`` in its metadata, the rendered context shows
  the full parent passage (optionally prefixed with the proposition).
- All other chunk kinds (``para``, ``code``, ``list``, etc.) are
  rendered as-is, identical to ``SimpleContextBuilder``.

Deduplication
=============

Proposition expansion introduces a new deduplication concern: multiple
propositions can originate from the *same* passage.  Expanding all of
them would repeat the passage text.  This builder applies two dedup
layers:

1. **Passage-identity dedupe** (strong): For proposition chunks in
   passage expansion mode, track ``doc_id:start_char:end_char`` of the
   parent passage.  Skip if already seen.
2. **Text-signature dedupe** (fallback): Normalize and truncate the
   rendered text, skip if the signature was already seen.  This covers
   non-proposition chunks and any edge cases.

See Also
========

- rag.adapters.context_building.simple_context_builder:
  The simpler context builder without proposition expansion.
- rag.adapters.chunking.proposition:
  The chunker that produces proposition chunks with parent passage metadata.
- rag.adapters.context_building._shared:
  Shared utilities (_estimate_tokens, _normalize_for_dedupe).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rag.adapters.context_building._shared import _estimate_tokens, _normalize_for_dedupe
from rag.domain.models import Candidate, Chunk, Citation, ContextPack


@dataclass(frozen=True, slots=True)
class PropositionAwareContextBuilder:
    """
    Context builder that expands proposition chunks to their parent passages.

    Follows the same algorithm as ``SimpleContextBuilder`` (sort, filter,
    dedupe, pack, cite, render) but adds proposition-specific logic:

    - **Expansion**: proposition chunks are rendered as their parent passage
      text, optionally prefixed with a ``Proposition: ...`` header.
    - **Dual-layer deduplication**: passage identity + text signature to
      avoid repeating the same passage when multiple propositions from it
      are retrieved.
    - **Enhanced citations**: each citation records ``rendered_kind``,
      ``passage_index``, and ``prop_index`` for traceability.

    Attributes:
        min_score: Minimum score threshold.  Candidates below this are skipped.
        max_chunks: Maximum number of chunks to include in context.
        dedupe: Enable deduplication (default True).
        include_scores: Show scores in the rendered context.
        expand_propositions: Whether to expand proposition chunks to their
            parent passages (default True).
        expansion_mode: ``"passage"`` to show the full parent passage, or
            ``"none"`` to show only the proposition text.
        include_prop_header: When expanding, prefix the passage with
            ``"Proposition: <text>"`` so the LLM sees what was retrieved.
    """

    min_score: float | None = None
    max_chunks: int = 12
    dedupe: bool = True
    include_scores: bool = False

    expand_propositions: bool = True
    expansion_mode: str = "passage"
    include_prop_header: bool = True

    def build(
        self,
        query: str,
        candidates: Sequence[Candidate],
        *,
        token_budget: int,
        metadata: Mapping[str, object] | None = None,
    ) -> ContextPack:
        """
        Build a ContextPack from ranked candidates with proposition expansion.

        Args:
            query: The user query string.
            candidates: Scored candidates (typically from retrieval + reranking).
            token_budget: Maximum estimated tokens for the rendered context.
            metadata: Optional metadata to attach to the ContextPack.

        Returns:
            A ContextPack containing the chosen chunks, rendered context
            string, and citations with provenance metadata.
        """

        def candidate_key(c: Candidate) -> float:
            return c.rerank_score if c.rerank_score is not None else c.score

        ordered = sorted(candidates, key=candidate_key, reverse=True)

        chosen: list[Chunk] = []
        citations: list[Citation] = []
        seen_text: set[str] = set()
        seen_passages: set[str] = set()

        tokens_used = _estimate_tokens("Context:\n")

        for cand in ordered:
            score = candidate_key(cand)
            if self.min_score is not None and score < self.min_score:
                continue

            chunk = cand.chunk
            ck = str(chunk.metadata.get("chunk_kind") or "")

            rendered_text = self._render_chunk_text(chunk)

            if self.dedupe:
                # Layer 1: passage-identity dedupe for proposition chunks
                if (
                    ck == "proposition"
                    and self.expand_propositions
                    and self.expansion_mode == "passage"
                ):
                    ps = chunk.metadata.get("parent_start_char")
                    pe = chunk.metadata.get("parent_end_char")

                    if isinstance(ps, int) and isinstance(pe, int):
                        passage_key = f"{chunk.doc_id}:{ps}:{pe}"
                        if passage_key in seen_passages:
                            continue
                        seen_passages.add(passage_key)

                # Layer 2: text-signature dedupe (covers all chunk kinds)
                sig = _normalize_for_dedupe(rendered_text)[:800]
                if sig in seen_text:
                    continue
                seen_text.add(sig)

            label = f"[{len(chosen) + 1}]"
            if self.include_scores:
                label += f" score={score:.4f}"
            label += "\n"

            chunk_tokens = (
                _estimate_tokens(label) + _estimate_tokens(rendered_text) + _estimate_tokens("\n\n")
            )
            if tokens_used + chunk_tokens > token_budget:
                break

            chosen.append(chunk)
            tokens_used += chunk_tokens

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
                    metadata={
                        "rank": len(chosen),
                        "score": score,
                        "rendered_kind": self._rendered_kind(chunk),
                        "parent_start_char": chunk.metadata.get("parent_start_char"),
                        "parent_end_char": chunk.metadata.get("parent_end_char"),
                        "passage_index": chunk.metadata.get("passage_index"),
                        "prop_index": chunk.metadata.get("prop_index"),
                    },
                )
            )

            if len(chosen) >= self.max_chunks:
                break

        rendered = self._render_context(chosen, ordered, token_budget=token_budget)

        return ContextPack(
            query=query,
            chunks=tuple(chosen),
            rendered_context=rendered,
            citations=tuple(citations),
            token_budget=token_budget,
            metadata={**(dict(metadata) if metadata else {}), "tokens_used_est": tokens_used},
        )

    def _rendered_kind(self, ch: Chunk) -> str:
        """Classify how a chunk was rendered: expanded to passage or as-is."""
        ck = str(ch.metadata.get("chunk_kind") or "")
        if (
            self.expand_propositions
            and ck == "proposition"
            and self.expansion_mode == "passage"
            and ch.metadata.get("parent_passage_text")
        ):
            return "proposition_expanded_to_passage"
        return ck or "unknown"

    def _render_chunk_text(self, ch: Chunk) -> str:
        """
        Render a chunk's text for inclusion in the LLM context.

        For proposition chunks with expansion enabled, returns the parent
        passage text (optionally prefixed with the proposition).  For all
        other chunks, returns the chunk text as-is.
        """
        ck = str(ch.metadata.get("chunk_kind") or "")
        if not (self.expand_propositions and ck == "proposition"):
            return ch.text.strip()

        if self.expansion_mode == "none":
            return ch.text.strip()

        parent = ch.metadata.get("parent_passage_text")
        if isinstance(parent, str) and parent.strip():
            if self.include_prop_header:
                return f"Proposition: {ch.text.strip()}\n\nPassage:\n{parent.strip()}"
            return parent.strip()

        # Fallback if metadata missing
        return ch.text.strip()

    def _render_context(
        self,
        chunks: Sequence[Chunk],
        ordered: Sequence[Candidate],
        *,
        token_budget: int,
    ) -> str:
        """Render the final context string sent to the LLM."""
        lines: list[str] = []
        lines.append(
            "You are given CONTEXT chunks from a document corpus. Answer the QUESTION using only the CONTEXT.\n"
        )
        lines.append("If the answer is not supported by the CONTEXT, say you don't know.\n")
        lines.append("CONTEXT:\n")

        score_by_id: dict[str, float] = {}
        if self.include_scores:

            def candidate_key(c: Candidate) -> float:
                return c.rerank_score if c.rerank_score is not None else c.score

            for c in ordered:
                score_by_id[c.chunk.chunk_id] = candidate_key(c)

        for i, ch in enumerate(chunks, start=1):
            header = f"[{i}]"
            if self.include_scores and ch.chunk_id in score_by_id:
                header += f" score={score_by_id[ch.chunk_id]:.4f}"
            lines.append(header)

            title = ch.metadata.get("title")
            uri = ch.metadata.get("uri") or ch.metadata.get("source_uri")
            if title or uri:
                lines.append(f"Source: {title or ''} {uri or ''}".strip())

            lines.append(self._render_chunk_text(ch))
            lines.append("")

        return "\n".join(lines).strip() + "\n"
