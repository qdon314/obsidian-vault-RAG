"""
Proposition-based chunker for RAG document processing.

Decomposes markdown documents into atomic, self-contained propositions
using a seq2seq model, following the approach described in
`Dense X Retrieval <https://arxiv.org/abs/2312.06648>`_.

Architecture Overview
=====================

The chunker reuses the shared markdown parsing pipeline from ``_markdown.py``
(Stages 1-3), then replaces the structural chunk assembly (Stage 4) with a
proposition extraction step::

    Document Text
         |
         v
    Stage 1-3: Shared Markdown Parsing (_sections_from_markdown)
         |   -> hierarchical sections with typed blocks and offsets
         v
    Stage 4a: Code blocks emitted as-is (no propositionization)
         |
         v
    Stage 4b: Non-code blocks packed into passages
         |   -> passages sized for the propositionizer input window
         v
    Stage 4c: Batch propositionization (Propositionizer)
         |   -> each passage yields a list of atomic propositions
         v
    One Chunk per proposition

Comparison with ObsidianStructuralChunker
=========================================

The structural chunker produces multi-sentence chunks (~4000 chars) that
preserve document structure.  The proposition chunker produces single-sentence
chunks (~50-150 chars) that are self-contained and independently meaningful.

+--------------------------+-----------------------------+
| ObsidianStructuralChunker| ObsidianPropositionChunker  |
+--------------------------+-----------------------------+
| Multi-sentence chunks    | Single-proposition chunks   |
| ~4000 chars target       | ~50-150 chars per chunk     |
| Structure-preserving     | Meaning-preserving          |
| Direct embedding of text | Each proposition embeddable |
| chunk_kind: para/list/...| chunk_kind: proposition     |
+--------------------------+-----------------------------+

Provenance
==========

Because a proposition is derived from a *passage* (a group of blocks),
exact character offsets for the proposition text within the source document
are not available.  Instead:

- ``start_char`` / ``end_char`` on the Chunk reflect the **parent passage**
  span in the original document.
- ``metadata["parent_passage_text"]`` stores the full passage text that
  the proposition was extracted from.
- ``metadata["passage_index"]`` and ``metadata["prop_index"]`` identify
  which passage and which proposition within that passage.

This allows downstream components (e.g. ``PropositionAwareContextBuilder``)
to expand a proposition back to its parent passage for richer LLM context.

Usage Example
=============

::

    from rag.adapters.chunking.proposition import (
        ObsidianPropositionChunker,
        Propositionizer,
    )
    from rag.domain.models import Document

    propositionizer = Propositionizer()  # loads model on init
    chunker = ObsidianPropositionChunker(
        propositionizer=propositionizer,
        passage_target_chars=900,
        passage_hard_max_chars=1400,
    )

    doc = Document(
        doc_id="note-001",
        text="# Overview\\n\\nPython is a programming language.\\nIt was created by Guido.",
        source="obsidian",
        uri="/vault/python.md",
        metadata={"title": "Python"},
    )

    chunks = chunker.chunk(doc)
    # Each chunk.text is a single self-contained proposition, e.g.:
    # "Python is a programming language."
    # "Python was created by Guido van Rossum."

See Also
========

- rag.adapters.chunking._markdown: Shared markdown parsing pipeline
- rag.adapters.chunking.obsidian_structural: Structure-preserving chunker
- rag.adapters.context_building.propositional_context_builder:
  Context builder that expands propositions back to passages
- Dense X Retrieval paper: https://arxiv.org/abs/2312.06648
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rag.adapters.chunking._markdown import (
    _Block,
    _extract_code_language,
    _sections_from_markdown,
    _stable_hash,
)
from rag.adapters.chunking.proposition.backends import Propositionizer
from rag.domain.models import Chunk, Document

# =============================================================================
# Chunker
# =============================================================================


@dataclass(frozen=True, slots=True)
class ObsidianPropositionChunker:
    """
    Proposition-based chunker for Obsidian markdown documents.

    Reuses the shared markdown parsing pipeline (Stages 1-3) from
    ``_markdown.py``, then replaces the structural assembly step with
    proposition extraction via a seq2seq model.

    Pipeline
    --------

    For each parsed section:

    1. **Code blocks** are emitted as chunks unchanged (no propositionization).
    2. **Non-code blocks** (paragraphs, lists, callouts, tables) are packed
       into *passages* of ``passage_target_chars`` (soft) /
       ``passage_hard_max_chars`` (hard).
    3. Each passage is fed to the ``Propositionizer``, which returns a list
       of atomic propositions.
    4. Each proposition becomes one Chunk.

    All propositionizer calls are batched across the entire document for
    throughput, then mapped back to their source passages.

    Metadata
    --------

    Each proposition chunk carries extra metadata keys:

    - ``chunk_kind``: always ``"proposition"``
    - ``passage_index``: sequential index of the parent passage within its section
    - ``prop_index``: index of this proposition within the parent passage
    - ``parent_passage_text``: full text of the passage the proposition was
      extracted from (enables passage-level expansion at context-building time)
    - ``parent_start_char`` / ``parent_end_char``: character offsets of the
      parent passage in the source document

    Attributes:
        propositionizer: The ``Propositionizer`` instance to use.
        passage_target_chars: Soft target for passage size in characters.
            When the block buffer exceeds this, flush as a passage.
        passage_hard_max_chars: Hard maximum for passage size.  If adding
            a block would exceed this, flush first.
        include_heading_preamble: Whether to prefix each proposition chunk
            with ``"Title: X\\nPath: Y"`` (default False, since preambles
            can pollute short proposition embeddings).
        overlap_blocks: Number of trailing blocks to carry forward between
            consecutive passages (default 0).
        strategy_name: Version identifier included in chunk IDs.

    See Also:
        - Module docstring for full architecture and provenance details.
        - ``ObsidianStructuralChunker`` for the structure-preserving alternative.
        - ``PropositionAwareContextBuilder`` for expanding propositions at
          context-building time.
    """

    propositionizer: Propositionizer

    passage_target_chars: int = 900
    """Soft target for passage size fed to the propositionizer."""

    passage_hard_max_chars: int = 1400
    """Hard maximum for passage size."""

    include_heading_preamble: bool = False
    """Prefix chunks with Title/Path preamble.  Default False."""

    overlap_blocks: int = 0
    """Trailing blocks to carry forward between passages."""

    strategy_name: str = "obsidian_proposition_v1"
    """Version string included in chunk IDs for reproducibility."""

    def chunk(self, doc: Document, *, metadata: Mapping[str, object] | None = None) -> list[Chunk]:
        """
        Split a Document into proposition-level Chunks.

        Orchestrates the full pipeline: parse markdown into sections,
        emit code blocks as-is, pack non-code blocks into passages,
        run the propositionizer in batch, and emit one Chunk per proposition.

        Args:
            doc: Document to chunk.
            metadata: Optional additional metadata to merge into each chunk.

        Returns:
            List of Chunk objects.  Code blocks have ``chunk_kind="code"``;
            propositions have ``chunk_kind="proposition"`` with parent
            passage metadata.
        """
        # Merge metadata layers: caller overrides doc.metadata
        merged_meta: dict[str, Any] = dict(doc.metadata)
        if metadata:
            merged_meta.update(dict(metadata))

        note_title = str(merged_meta.get("title") or merged_meta.get("file_name") or doc.uri)

        sections = _sections_from_markdown(doc.text)
        if not sections:
            return []

        # We batch all propositionizer calls for throughput, but we still need to map outputs back.
        prop_inputs: list[str] = []
        prop_map: list[dict[str, Any]] = []

        # We also accumulate code chunks directly.
        chunks: list[Chunk] = []
        chunk_index = 0

        for sec in sections:
            section_path = " > ".join(sec.path) if sec.path else None
            section_heading = sec.path[-1] if sec.path else None

            preamble = ""
            if self.include_heading_preamble:
                if section_path:
                    preamble = f"Title: {note_title}\nPath: {section_path}\n\n"
                else:
                    preamble = f"Title: {note_title}\n\n"

            # ---- Pass 1: emit code blocks as chunks ----
            for b in sec.blocks:
                if not b.text.strip():
                    continue
                if b.kind != "code":
                    continue

                lang = _extract_code_language(b.text)
                text = (
                    (preamble + b.text).strip() if self.include_heading_preamble else b.text.strip()
                )

                chunk_id = _stable_hash(
                    [
                        doc.doc_id,
                        self.strategy_name,
                        "code",
                        str(chunk_index),
                        str(b.start),
                        str(b.end),
                        section_path or "",
                    ]
                )

                ch_meta = dict(merged_meta)
                ch_meta.update(
                    {
                        "chunk_kind": "code",
                        "chunk_strategy": self.strategy_name,
                    }
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=text,
                        chunk_index=chunk_index,
                        start_char=b.start,
                        end_char=b.end,
                        section_heading=section_heading,
                        section_path=section_path,
                        language=lang,
                        metadata=ch_meta,
                    )
                )
                chunk_index += 1

            # ---- Pass 2: pack non-code blocks into passages for propositionization ----
            noncode_blocks = [b for b in sec.blocks if b.kind != "code" and b.text.strip()]
            if not noncode_blocks:
                continue

            buf: list[_Block] = []
            buf_len = 0
            passage_index = 0

            def flush_passage(
                *,
                section_heading=section_heading,
                section_path=section_path,
                preamble=preamble,
            ) -> None:
                nonlocal buf, buf_len, passage_index
                if not buf:
                    return

                start_char = min(x.start for x in buf)
                end_char = max(x.end for x in buf)
                passage_text = "".join(x.text for x in buf).strip()
                if not passage_text:
                    buf = []
                    buf_len = 0
                    return

                # Propositionizer input format
                sec_label = section_heading or ""
                inp = f"Title: {note_title}. Section: {sec_label}. Content: {passage_text}"

                prop_inputs.append(inp)
                prop_map.append(
                    {
                        "section_heading": section_heading,
                        "section_path": section_path,
                        "preamble": preamble,
                        "start_char": start_char,
                        "end_char": end_char,
                        "passage_text": passage_text,
                        "passage_index": passage_index,
                    }
                )
                passage_index += 1

                if self.overlap_blocks > 0:
                    buf = buf[-self.overlap_blocks :]
                    buf_len = sum(len(x.text) for x in buf)
                else:
                    buf = []
                    buf_len = 0

            for b in noncode_blocks:
                if buf and (buf_len + len(b.text) > self.passage_hard_max_chars):
                    flush_passage()

                buf.append(b)
                buf_len += len(b.text)

                if buf_len >= self.passage_target_chars:
                    flush_passage()

            flush_passage()

        # ---- Run propositionizer in batch ----
        if prop_inputs:
            prop_lists = self.propositionizer.propositionize_many(prop_inputs)

            for m, props in zip(prop_map, prop_lists, strict=False):
                # Fallback: if propositionizer returns nothing, keep the passage
                if not props:
                    props = [m["passage_text"]]

                for prop_index, prop in enumerate(props):
                    prop = prop.strip()
                    if not prop:
                        continue

                    text = (m["preamble"] + prop).strip() if self.include_heading_preamble else prop

                    chunk_id = _stable_hash(
                        [
                            doc.doc_id,
                            self.strategy_name,
                            "prop",
                            str(chunk_index),
                            str(m["start_char"]),
                            str(m["end_char"]),
                            m["section_path"] or "",
                            _stable_hash([prop]),
                        ]
                    )

                    ch_meta = dict(merged_meta)
                    ch_meta.update(
                        {
                            "chunk_kind": "proposition",
                            "chunk_strategy": self.strategy_name,
                            "passage_index": m["passage_index"],
                            "prop_index": prop_index,
                            "parent_passage_text": m["passage_text"],
                            "parent_start_char": m["start_char"],
                            "parent_end_char": m["end_char"],
                        }
                    )

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            doc_id=doc.doc_id,
                            text=text,
                            chunk_index=chunk_index,
                            start_char=m["start_char"],
                            end_char=m["end_char"],
                            section_heading=m["section_heading"],
                            section_path=m["section_path"],
                            language="markdown",
                            metadata=ch_meta,
                        )
                    )
                    chunk_index += 1

        return chunks

    def get_config(self) -> dict[str, Any]:
        """Return the chunker configuration for manifest/introspection."""
        return {
            "backend": "obsidian_proposition",
            "strategy_name": self.strategy_name,
            "propositionizer": self.propositionizer.get_config(),
            "passage_target_chars": self.passage_target_chars,
            "passage_hard_max_chars": self.passage_hard_max_chars,
            "overlap_blocks": self.overlap_blocks,
            "include_heading_preamble": self.include_heading_preamble,
        }
