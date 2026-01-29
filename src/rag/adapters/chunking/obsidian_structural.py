"""
Obsidian-aware structural chunker for RAG document processing.

This module implements a chunking strategy specifically designed for
Obsidian markdown notes and similar structured documents. Unlike simple
fixed-size chunkers, it preserves semantic boundaries by respecting
markdown structure.

The parsing pipeline (Stages 1-3) is implemented in ``_markdown.py``.
This module provides Stage 4: chunk assembly, which packs parsed blocks
into size-constrained chunks with optional overlap.

Size Constraints
================

The chunker uses two size parameters:

    target_chars (default: 4000)
        Soft target size. When buffer exceeds this, flush current chunk
        and start a new one. Blocks are never split at this boundary.

    hard_max_chars (default: 5200)
        Hard maximum. If adding a block would exceed this, flush first.
        Only PARAGRAPHS can be split if they individually exceed hard_max.
        Code blocks, lists, tables, and callouts are never split.

    Visual representation of size handling:

    |<---------- target_chars ---------->|<-- buffer -->|
    |                                     |               |
    +-----------+-------------------------+               |
    |         Current Chunk               |   Block N     |
    |         (flushed when target        |  (triggers    |
    |          is exceeded)               |   flush)      |
    +-------------------------------------+---------------+

    |<-------------------- hard_max_chars ----------------->|
    |                                                        |
    |  Absolute maximum - flush before adding if exceeded    |


Overlap Behavior
================

When overlap_blocks > 0, the last N blocks from each chunk are carried
forward to the next chunk. This provides context continuity:

    overlap_blocks=1:

    Chunk 1: [Block A] [Block B] [Block C]
                                    |
                                    v
    Chunk 2:               [Block C] [Block D] [Block E]
                                                   |
                                                   v
    Chunk 3:                             [Block E] [Block F]

    This helps with:
    - Retrieval quality (context spans chunk boundaries)
    - LLM comprehension (no abrupt context switches)


See Also
========

- rag.adapters.chunking._markdown: Parsing pipeline (Stages 1-3)
- rag.ports.chunker.Chunker: Protocol this class implements
- rag.domain.models.Chunk: Output data structure
- rag.domain.models.Document: Input data structure
- tests.adapters.chunking.test_obsidian_structural_chunker: Comprehensive tests
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from rag.adapters.chunking._markdown import (
    _Block,
    _chunk_kind,
    _extract_code_language,
    _sections_from_markdown,
    _stable_hash,
)

# =============================================================================
# Main Chunker Class
# =============================================================================


@dataclass(frozen=True, slots=True)
class ObsidianStructuralChunker:
    """
    Structure-aware chunker optimized for Obsidian markdown notes.

    This chunker implements the Chunker protocol and provides intelligent
    document splitting that respects markdown semantic boundaries. Unlike
    naive fixed-size chunking, it preserves:

    - Section hierarchy (heading structure)
    - Block integrity (lists, tables, code blocks, callouts)
    - Document context (title, section path in preamble)
    - Stable identifiers (deterministic chunk IDs)

    Key Features:
        1. **Heading-based sectioning**: Splits at heading boundaries while
           maintaining the full path (e.g., "Chapter 1 > Section A").

        2. **Block-aware packing**: Groups related content (lists, tables,
           code blocks) and avoids splitting them mid-structure.

        3. **Flexible sizing**: Soft target with hard maximum, allowing
           natural content boundaries within reasonable size constraints.

        4. **Overlap support**: Optional block overlap between consecutive
           chunks for improved retrieval context.

        5. **Rich metadata**: Each chunk includes content type, section info,
           detected language (for code), and character offsets.

    Attributes:
        target_chars: Soft target size in characters (default: 4000).
            When the buffer exceeds this, the chunker flushes and starts
            a new chunk. Blocks are never split at this boundary.

        hard_max_chars: Hard maximum size in characters (default: 5200).
            If adding a block would exceed this, flush first. Only
            paragraphs can be split if they individually exceed this limit.
            Code blocks, lists, tables, and callouts are never split.

        overlap_blocks: Number of trailing blocks to carry forward to the
            next chunk (default: 1). Provides context continuity across
            chunk boundaries. Set to 0 for no overlap.

        include_heading_preamble: Whether to prepend contextual header
            (default: True). When True, chunks start with:
                Title: <document title>
                Path: <section path>

        strategy_name: Version identifier for this chunking strategy
            (default: "obsidian_structural_v1"). Included in chunk IDs
            for reproducibility tracking.

    Example:
        >>> from rag.adapters.chunking.obsidian_structural import ObsidianStructuralChunker
        >>> from rag.domain.models import Document
        >>>
        >>> chunker = ObsidianStructuralChunker(
        ...     target_chars=2000,
        ...     hard_max_chars=3000,
        ...     overlap_blocks=1,
        ... )
        >>>
        >>> doc = Document(
        ...     doc_id="note-001",
        ...     text="# Getting Started\\n\\nWelcome to the guide...\\n\\n## Installation\\n...",
        ...     source="obsidian",
        ...     uri="/vault/guide.md",
        ...     metadata={"title": "User Guide"},
        ... )
        >>>
        >>> chunks = chunker.chunk(doc)
        >>> for chunk in chunks:
        ...     print(f"{chunk.chunk_id}: {chunk.section_path} ({len(chunk.text)} chars)")

    See Also:
        - Module docstring for full architecture overview
        - rag.ports.chunker.Chunker: Protocol definition
        - rag.domain.models.Chunk: Output data structure
    """

    target_chars: int = 4000
    """Soft target size. Flush when buffer exceeds this. Default: 4000."""

    hard_max_chars: int = 5200
    """Hard maximum. Flush before adding if exceeded. Default: 5200."""

    overlap_blocks: int = 1
    """Trailing blocks to repeat in next chunk. Default: 1."""

    include_heading_preamble: bool = True
    """Prepend 'Title: X\\nPath: Y' to each chunk. Default: True."""

    strategy_name: str = "obsidian_structural_v1"
    """Version string for chunk ID stability. Default: 'obsidian_structural_v1'."""

    def chunk(self, doc, *, metadata: Mapping[str, object] | None = None) -> list:
        """
        Split a Document into semantically coherent Chunks.

        This is the main entry point implementing the Chunker protocol. It
        orchestrates the full chunking pipeline:

        1. Parse markdown into sections and blocks
        2. Iterate through sections, packing blocks into chunks
        3. Apply size constraints (target_chars, hard_max_chars)
        4. Handle special cases (oversize paragraphs, code blocks)
        5. Apply overlap if configured
        6. Generate stable chunk IDs and rich metadata

        Args:
            doc: Document to chunk. Expected fields:
                - doc_id: Unique document identifier
                - text: Full document content (markdown)
                - uri: Document location (used in preamble fallback)
                - metadata: Document-level metadata to propagate

            metadata: Optional additional metadata to merge. Takes precedence
                over doc.metadata for conflicting keys.

        Returns:
            List of Chunk objects. Each chunk includes:
                - chunk_id: Stable hash-based identifier
                - doc_id: Reference to source document
                - text: Chunk content (with optional preamble)
                - chunk_index: Sequential position (0, 1, 2, ...)
                - start_char: Start offset in source document
                - end_char: End offset in source document
                - section_heading: Current section heading (or None)
                - section_path: Full path like "Parent > Child" (or None)
                - language: Detected language for code chunks (or None)
                - metadata: Merged metadata including chunk_kind, chunk_strategy

        Raises:
            No exceptions are raised; malformed markdown is handled gracefully.

        Example:
            >>> doc = Document(
            ...     doc_id="my-doc",
            ...     text="# Hello\\n\\nWorld",
            ...     source="test",
            ...     uri="/test.md",
            ...     metadata={},
            ... )
            >>> chunks = chunker.chunk(doc)
            >>> len(chunks)
            1
            >>> chunks[0].section_heading
            'Hello'
        """
        # Merge metadata layers: caller overrides doc.metadata if provided
        merged_meta: dict[str, Any] = dict(doc.metadata)
        if metadata:
            merged_meta.update(dict(metadata))

        # Stage 1-3: Parse document into sections containing blocks
        sections = _sections_from_markdown(doc.text)

        # Determine title for preamble (priority: title > file_name > uri)
        note_title = str(merged_meta.get("title") or merged_meta.get("file_name") or doc.uri)

        # Stage 4: Chunk Assembly
        # --------------------------
        # Iterate through sections, packing blocks into chunks while
        # respecting size constraints and maintaining overlap.

        chunks = []
        chunk_index = 0

        for sec in sections:
            # Build section context for preamble and metadata
            section_path = " > ".join(sec.path) if sec.path else None
            section_heading = sec.path[-1] if sec.path else None

            # Construct preamble (prepended to each chunk for context)
            preamble = ""
            if self.include_heading_preamble:
                if sec.path:
                    preamble = f"Title: {note_title}\nPath: {section_path}\n\n"
                else:
                    preamble = f"Title: {note_title}\n\n"

            # Buffer to accumulate blocks until we hit a size boundary
            buf: list[_Block] = []
            buf_len = len(preamble)  # Track buffer size including preamble

            def flush(preamble=preamble, section_path=section_path, section_heading=section_heading):
                """
                Emit current buffer as a chunk and optionally preserve overlap.

                This closure captures the current section context and handles:
                1. Computing character offsets from buffered blocks
                2. Detecting language for code-only chunks
                3. Generating stable chunk ID
                4. Building chunk metadata
                5. Managing overlap for the next chunk
                """
                nonlocal buf, buf_len, chunk_index

                if not buf:
                    return

                # Compute character range spanned by buffered blocks
                start_char = min(b.start for b in buf)
                end_char = max(b.end for b in buf)

                # Combine block text, stripping outer whitespace
                body = "".join(b.text for b in buf).strip()
                if not body:
                    buf = []
                    buf_len = len(preamble)
                    return

                text = preamble + body

                # Detect language for pure code chunks (single code block)
                lang = None
                if _chunk_kind(buf) == "code" and len(buf) == 1:
                    lang = _extract_code_language(buf[0].text)

                # Generate deterministic chunk ID from content characteristics
                chunk_id = _stable_hash(
                    [
                        doc.doc_id,
                        self.strategy_name,
                        str(chunk_index),
                        str(start_char),
                        str(end_char),
                        section_path or "",
                    ]
                )

                # Build chunk metadata (merge doc metadata + chunker additions)
                ch_meta = dict(merged_meta)
                ch_meta.update(
                    {
                        "chunk_kind": _chunk_kind(buf),
                        "chunk_strategy": self.strategy_name,
                    }
                )

                # Import here to avoid circular dependency issues
                from rag.domain.models import Chunk

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=doc.doc_id,
                        text=text,
                        chunk_index=chunk_index,
                        start_char=start_char,
                        end_char=end_char,
                        section_heading=section_heading,
                        section_path=section_path,
                        language=lang,
                        metadata=ch_meta,
                    )
                )

                chunk_index += 1

                # Handle overlap: carry last N blocks forward to next chunk
                if self.overlap_blocks > 0:
                    buf = buf[-self.overlap_blocks :]
                    buf_len = len(preamble) + sum(len(b.text) for b in buf)
                else:
                    buf = []
                    buf_len = len(preamble)

            # Process each block in this section
            for b in sec.blocks:
                # Skip empty/whitespace-only blocks
                if not b.text.strip():
                    continue

                # SPECIAL CASE: Oversize paragraph
                # Only paragraphs can be split; other block types (code, list,
                # table, callout) are kept intact even if they exceed hard_max
                if b.kind == "para" and len(b.text) > self.hard_max_chars:
                    flush()
                    start = 0
                    while start < len(b.text):
                        piece = b.text[start : start + self.target_chars]
                        piece_start = b.start + start
                        piece_end = piece_start + len(piece)

                        text = (preamble + piece).strip()

                        chunk_id = _stable_hash(
                            [
                                doc.doc_id,
                                self.strategy_name,
                                str(chunk_index),
                                str(piece_start),
                                str(piece_end),
                                section_path or "",
                                "oversize_para",
                            ]
                        )

                        from rag.domain.models import Chunk

                        ch_meta = dict(merged_meta)
                        ch_meta.update(
                            {
                                "chunk_kind": "para",
                                "chunk_strategy": self.strategy_name,
                                "split_reason": "oversize_paragraph",
                            }
                        )

                        chunks.append(
                            Chunk(
                                chunk_id=chunk_id,
                                doc_id=doc.doc_id,
                                text=text,
                                chunk_index=chunk_index,
                                start_char=piece_start,
                                end_char=piece_end,
                                section_heading=section_heading,
                                section_path=section_path,
                                language=None,
                                metadata=ch_meta,
                            )
                        )

                        chunk_index += 1
                        start += self.target_chars

                    continue

                # HARD MAX CHECK: If adding this block would exceed hard_max,
                # flush current buffer first to avoid oversized chunks
                if buf and (buf_len + len(b.text) > self.hard_max_chars):
                    flush()

                # Add block to buffer
                buf.append(b)
                buf_len += len(b.text)

                # SOFT TARGET CHECK: If we've exceeded target_chars, flush
                # This creates naturally-sized chunks at block boundaries
                if buf_len >= self.target_chars:
                    flush()

            # Flush any remaining blocks in this section
            flush()

        return chunks

    def get_config(self) -> dict[str, Any]:
        """
        Return the chunker's configuration for manifest/introspection.

        This method is part of the Chunker protocol and enables configuration
        tracking for reproducibility and debugging.

        Returns:
            Dictionary containing:
                - backend: "obsidian_structural"
                - target_chars: Soft target size
                - hard_max_chars: Hard maximum size
                - overlap_blocks: Overlap setting
                - include_heading_preamble: Preamble setting
                - strategy_name: Version identifier

        Example:
            >>> chunker = ObsidianStructuralChunker(target_chars=2000)
            >>> chunker.get_config()
            {
                'backend': 'obsidian_structural',
                'target_chars': 2000,
                'hard_max_chars': 5200,
                'overlap_blocks': 1,
                'include_heading_preamble': True,
                'strategy_name': 'obsidian_structural_v1'
            }
        """
        return {
            "backend": "obsidian_structural",
            "target_chars": self.target_chars,
            "hard_max_chars": self.hard_max_chars,
            "overlap_blocks": self.overlap_blocks,
            "include_heading_preamble": self.include_heading_preamble,
            "strategy_name": self.strategy_name,
        }
