"""
Obsidian-aware structural chunker for RAG document processing.

This module implements a sophisticated chunking strategy specifically designed for
Obsidian markdown notes and similar structured documents. Unlike simple fixed-size
chunkers, it preserves semantic boundaries by respecting markdown structure.

Architecture Overview
=====================

The chunking pipeline follows this flow:

    Document Text
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 1: Code Block Isolation                                  │
    │  _split_fenced_code_blocks()                                    │
    │  - Splits text into code/non-code segments                      │
    │  - Preserves ``` fences and tracks character offsets            │
    └─────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 2: Section Parsing                                       │
    │  _sections_from_markdown()                                      │
    │  - Identifies markdown headings (# to ######)                   │
    │  - Builds hierarchical section paths (e.g., "Parent > Child")   │
    │  - Groups content under appropriate sections                    │
    └─────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 3: Block Detection                                       │
    │  _blocks_from_noncode_text()                                    │
    │  - Identifies structural blocks: para, list, callout, table     │
    │  - Tracks precise character offsets for each block              │
    └─────────────────────────────────────────────────────────────────┘
         │
         ▼
    ┌─────────────────────────────────────────────────────────────────┐
    │  Stage 4: Chunk Assembly                                        │
    │  ObsidianStructuralChunker.chunk()                              │
    │  - Packs blocks into chunks respecting size constraints         │
    │  - Adds contextual preambles (Title, Section Path)              │
    │  - Generates stable chunk IDs via content hashing               │
    │  - Handles overlap between consecutive chunks                   │
    └─────────────────────────────────────────────────────────────────┘
         │
         ▼
    List[Chunk]


Block Types
===========

The chunker recognizes these markdown structural elements:

    +----------+---------------------------+----------------------------+
    | Type     | Detection Pattern         | Example                    |
    +----------+---------------------------+----------------------------+
    | para     | Default text content      | Regular paragraph text     |
    | list     | Lines starting with       | - Item one                 |
    |          | -, *, +, or 1. / 1)       | - Item two                 |
    | callout  | Lines starting with >     | > [!note] This is a note   |
    |          | (blockquotes & Obsidian   | > Callout content          |
    |          | callouts like > [!note])  |                            |
    | table    | Lines matching |...|      | | Col A | Col B |          |
    |          |                           | |-------|-------|          |
    | code     | Content within ``` fences | ```python                  |
    |          |                           | print("hello")             |
    |          |                           | ```                        |
    | mixed    | Multiple block types in   | Paragraph + list + code    |
    |          | a single chunk            | combined in one chunk      |
    +----------+---------------------------+----------------------------+


Chunk ID Generation
===================

Chunk IDs are generated using BLAKE2b hashing for stability and collision resistance.
The hash incorporates:

    - doc_id: Source document identifier
    - strategy_name: Chunking strategy version (for reproducibility)
    - chunk_index: Sequential position in output
    - start_char / end_char: Character offsets in source document
    - section_path: Hierarchical section location

This ensures:
    1. Same input always produces same chunk IDs (deterministic)
    2. Different documents never collide (doc_id included)
    3. Strategy changes produce new IDs (versioned)


Usage Example
=============

    from rag.adapters.chunking.obsidian_structural import ObsidianStructuralChunker
    from rag.domain.models import Document

    # Create chunker with custom settings
    chunker = ObsidianStructuralChunker(
        target_chars=4000,      # Soft target for chunk size
        hard_max_chars=5200,    # Never exceed this (except code blocks)
        overlap_blocks=1,       # Include last block from previous chunk
        include_heading_preamble=True,  # Add "Title: X\\nPath: Y" prefix
    )

    # Create a document
    doc = Document(
        doc_id="my-note-001",
        text="# Introduction\\n\\nThis is my note content...\\n\\n## Details\\n...",
        source="obsidian",
        uri="/vault/my-note.md",
        metadata={"title": "My Note", "tags": ["python", "tutorial"]},
    )

    # Chunk the document
    chunks = chunker.chunk(doc)

    # Each chunk contains:
    # - chunk_id: Stable identifier
    # - doc_id: Reference to source document
    # - text: Chunk content with optional preamble
    # - chunk_index: Position (0, 1, 2, ...)
    # - start_char / end_char: Offsets into original text
    # - section_heading: Current heading (e.g., "Details")
    # - section_path: Full path (e.g., "Introduction > Details")
    # - language: Detected language for code blocks
    # - metadata: Merged doc + caller metadata + chunk_kind


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

    │◄────────── target_chars ──────────►│◄── buffer ──►│
    │                                     │               │
    ├─────────────────────────────────────┤               │
    │         Current Chunk               │   Block N     │
    │         (flushed when target        │  (triggers    │
    │          is exceeded)               │   flush)      │
    └─────────────────────────────────────┴───────────────┘

    │◄──────────────────── hard_max_chars ─────────────────►│
    │                                                        │
    │  Absolute maximum - flush before adding if exceeded    │


Overlap Behavior
================

When overlap_blocks > 0, the last N blocks from each chunk are carried
forward to the next chunk. This provides context continuity:

    overlap_blocks=1:

    Chunk 1: [Block A] [Block B] [Block C]
                                    │
                                    ▼
    Chunk 2:               [Block C] [Block D] [Block E]
                                                   │
                                                   ▼
    Chunk 3:                             [Block E] [Block F]

    This helps with:
    - Retrieval quality (context spans chunk boundaries)
    - LLM comprehension (no abrupt context switches)


See Also
========

- rag.ports.chunker.Chunker: Protocol this class implements
- rag.domain.models.Chunk: Output data structure
- rag.domain.models.Document: Input data structure
- tests.adapters.chunking.test_obsidian_structural_chunker: Comprehensive tests
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

# =============================================================================
# Regex Patterns for Markdown Structure Detection
# =============================================================================

# Matches markdown headings: # Title, ## Subtitle, etc.
# Captures: group(1) = hashes (level), group(2) = heading text
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

# Matches code fence opening/closing: ```, ```python, etc.
_FENCE_RE = re.compile(r"^\s*```")

# Matches Obsidian callout syntax: > [!note] Title, > [!warning], etc.
# Captures: group(1) = callout type (note, warning, etc.), group(2) = title
_CALLOUT_START_RE = re.compile(r"^\s*>\s*\[!(\w+)\]\s*(.*)$")

# Matches any blockquote line: > text, > , etc.
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?.*$")

# Matches list items: -, *, +, or numbered (1., 2), 10., etc.)
# Captures: group(1) = number for ordered lists (if present)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|(\d+)[.)])\s+.+$")

# Matches markdown table rows: |col1|col2|, | col1 | col2 |, etc.
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")

# Extracts language from code fence: ```python -> "python", ``` -> None
_FENCE_LANG_RE = re.compile(r"^\s*```(\w+)?\s*$")


# =============================================================================
# Helper Functions
# =============================================================================


def _stable_hash(parts: list[str]) -> str:
    """
    Generate a stable, deterministic hash from a list of string parts.

    Uses BLAKE2b with a 12-byte digest (24 hex characters) for a good balance
    between collision resistance and ID length. Parts are separated by the
    ASCII unit separator (0x1F) to ensure ["a", "bc"] != ["ab", "c"].

    Args:
        parts: List of strings to hash together.

    Returns:
        24-character hexadecimal hash string.

    Example:
        >>> _stable_hash(["doc-001", "v1", "0", "100", "200"])
        'a1b2c3d4e5f6a1b2c3d4e5f6'
    """
    h = blake2b(digest_size=12)
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")  # Unit separator prevents collision between adjacent values
    return h.hexdigest()


def _split_fenced_code_blocks(text: str) -> list[tuple[str, bool, int]]:
    """
    Split markdown text into code and non-code segments.

    This is Stage 1 of the chunking pipeline. It identifies fenced code blocks
    (delimited by ```) and separates them from regular markdown content while
    tracking exact character positions.

    The function handles:
    - Opening and closing fences (```)
    - Language specifiers (```python, ```javascript)
    - Nested content that looks like fences but isn't (inside code)

    Args:
        text: Raw markdown text to split.

    Returns:
        List of tuples: (segment_text, is_code_block, segment_start_char)
        - segment_text: The text content of this segment
        - is_code_block: True if this segment is a fenced code block
        - segment_start_char: Character offset where this segment begins

    Example:
        Input:
            "Hello world.

            ```python
            print('hi')
            ```

            More text."

        Output:
            [
                ("Hello world.\\n\\n", False, 0),
                ("```python\\nprint('hi')\\n```\\n", True, 14),
                ("\\nMore text.", False, 42),
            ]
    """
    lines = text.splitlines(keepends=True)
    segs: list[tuple[str, bool, int]] = []

    buf: list[str] = []
    in_code = False
    seg_start = 0
    cursor = 0

    def flush():
        nonlocal buf, seg_start
        if buf:
            segs.append(("".join(buf), in_code, seg_start))
            buf = []
            seg_start = cursor

    for line in lines:
        if _FENCE_RE.match(line):
            if in_code:
                buf.append(line)
                cursor += len(line)
                flush()
                in_code = False
            else:
                flush()
                in_code = True
                buf.append(line)
                cursor += len(line)
            continue

        buf.append(line)
        cursor += len(line)

    flush()
    return segs


# =============================================================================
# Internal Data Structures
# =============================================================================


@dataclass(frozen=True, slots=True)
class _Block:
    """
    Represents a single structural block within a markdown section.

    A block is the smallest semantic unit that the chunker will not split
    (except for oversize paragraphs). Blocks are categorized by type to
    enable intelligent chunking decisions.

    Attributes:
        kind: Block type identifier. One of:
            - "para": Regular paragraph text
            - "list": Ordered or unordered list (including nested items)
            - "callout": Blockquote or Obsidian callout (> [!note])
            - "table": Markdown table (|col|col|)
            - "code": Fenced code block (```...```)
        text: The raw text content of this block, including any delimiters.
        start: Character offset where this block begins in the source document.
        end: Character offset where this block ends in the source document.

    Note:
        start and end enable precise mapping back to the original document,
        which is useful for highlighting, citations, and debugging.
    """

    kind: str  # "para" | "list" | "callout" | "table" | "code"
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Section:
    """
    Represents a markdown section delimited by headings.

    Sections form a hierarchy based on heading levels. A ## heading under a
    # heading creates a parent-child relationship tracked via the path tuple.

    Attributes:
        level: Heading level (1-6), or 0 for content before any heading.
        title: The heading text (e.g., "Introduction"), empty for pre-heading.
        path: Tuple of heading titles from root to current section.
            Example: ("Chapter 1", "Section A", "Subsection i")
        blocks: Tuple of _Block objects belonging to this section.

    Example:
        For markdown:
            # Chapter 1
            ## Overview
            Some text here.

        The "Overview" section would have:
            level=2
            title="Overview"
            path=("Chapter 1", "Overview")
            blocks=(Block(kind="para", text="Some text here.", ...),)
    """

    level: int
    title: str
    path: tuple[str, ...]
    blocks: tuple[_Block, ...]


def _extract_code_language(code_block_text: str) -> str | None:
    """
    Extract the programming language from a fenced code block.

    Looks at the opening fence line (e.g., ```python) to determine the language.

    Args:
        code_block_text: Full text of the code block including fences.

    Returns:
        Language identifier (e.g., "python", "javascript") or None if not specified.

    Example:
        >>> _extract_code_language("```python\\nprint('hi')\\n```")
        'python'
        >>> _extract_code_language("```\\nplain code\\n```")
        None
    """
    first_line = code_block_text.splitlines()[0] if code_block_text else ""
    m = _FENCE_LANG_RE.match(first_line)
    if not m:
        return None
    lang = (m.group(1) or "").strip()
    return lang or None


def _blocks_from_noncode_text(text: str, base_offset: int) -> list[_Block]:
    """
    Parse non-code markdown text into structural blocks.

    This is Stage 3 of the chunking pipeline. It uses a state machine to
    identify block boundaries based on line patterns:

    State Machine:
        ┌─────────────────────────────────────────────────────────────┐
        │                         START                               │
        └───────────────────────────┬─────────────────────────────────┘
                                    │
            ┌───────────────────────┼───────────────────────┐
            │                       │                       │
            ▼                       ▼                       ▼
        ┌───────┐              ┌────────┐              ┌─────────┐
        │ para  │◄────────────►│  list  │◄────────────►│ callout │
        └───┬───┘   blank line └────┬───┘   blank line └────┬────┘
            │                       │                       │
            │       ┌───────────────┼───────────────────────┤
            │       │               │                       │
            │       ▼               ▼                       ▼
            │   ┌───────┐      ┌────────┐              ┌─────────┐
            └──►│ table │      │ FLUSH  │◄─────────────│  FLUSH  │
                └───────┘      └────────┘              └─────────┘

    Detection rules:
        - Lists: Lines starting with -, *, +, or digit + . or )
        - Callouts: Lines starting with > (blockquotes, Obsidian callouts)
        - Tables: Lines matching |...|
        - Paragraphs: Everything else (default)

    Args:
        text: Non-code markdown text to parse.
        base_offset: Character offset of this text in the original document.

    Returns:
        List of _Block objects with precise character offsets.
    """
    lines = text.splitlines(keepends=True)

    blocks: list[_Block] = []
    buf: list[str] = []
    mode: str | None = None
    buf_start = base_offset
    cursor = base_offset

    def flush():
        nonlocal buf, mode, buf_start
        if not buf:
            return
        seg_text = "".join(buf)
        blocks.append(
            _Block(
                kind=mode or "para", text=seg_text, start=buf_start, end=buf_start + len(seg_text)
            )
        )
        buf = []
        mode = None
        buf_start = cursor

    def is_blank(ln: str) -> bool:
        return ln.strip() == ""

    for ln in lines:
        is_quote = bool(_BLOCKQUOTE_RE.match(ln))
        is_table = bool(_TABLE_ROW_RE.match(ln))
        is_list = bool(_LIST_ITEM_RE.match(ln))

        # Callouts / blockquotes
        if is_quote:
            if mode not in (None, "callout"):
                flush()
            if mode is None:
                mode = "callout"
                buf_start = cursor
            mode = "callout"
            buf.append(ln)
            cursor += len(ln)
            continue

        # Tables
        if is_table:
            if mode not in (None, "table"):
                flush()
            if mode is None:
                mode = "table"
                buf_start = cursor
            mode = "table"
            buf.append(ln)
            cursor += len(ln)
            continue

        # Lists: contiguous list items + indented continuations
        if is_list or (
            mode == "list" and (ln.startswith(" ") or ln.startswith("\t")) and not is_blank(ln)
        ):
            if mode not in (None, "list"):
                flush()
            if mode is None:
                mode = "list"
                buf_start = cursor
            mode = "list"
            buf.append(ln)
            cursor += len(ln)
            continue

        # Blank: paragraph boundary (and ends structured blocks)
        if is_blank(ln):
            if mode in ("callout", "table", "list"):
                # blank ends these blocks
                cursor += len(ln)
                flush()
                continue

            # paragraph: include blank then flush
            if mode != "para":
                mode = "para"
                buf_start = cursor
            buf.append(ln)
            cursor += len(ln)
            flush()
            continue

        # Default: paragraph line
        if mode not in (None, "para"):
            flush()
        if mode is None:
            mode = "para"
            buf_start = cursor
        mode = "para"
        buf.append(ln)
        cursor += len(ln)

    flush()
    return blocks


def _sections_from_markdown(text: str) -> list[_Section]:
    """
    Parse markdown text into hierarchical sections.

    This is Stage 2 of the chunking pipeline. It combines code block isolation
    with heading detection to build a section tree structure.

    The algorithm:
        1. Split text into code/non-code segments (_split_fenced_code_blocks)
        2. Scan non-code segments for headings
        3. Maintain a heading stack to track section hierarchy
        4. Parse non-heading content into blocks (_blocks_from_noncode_text)
        5. Group blocks under their containing section

    Section Hierarchy Example:
        # Chapter 1           <- level=1, path=("Chapter 1",)
        Some intro text.
        ## Section A          <- level=2, path=("Chapter 1", "Section A")
        Section A content.
        ## Section B          <- level=2, path=("Chapter 1", "Section B")
        Section B content.
        ### Subsection        <- level=3, path=("Chapter 1", "Section B", "Subsection")
        Subsection content.

    Args:
        text: Full markdown document text.

    Returns:
        List of _Section objects in document order.
        Empty sections (no non-whitespace content) are filtered out.
    """
    segments = _split_fenced_code_blocks(text)

    sections: list[_Section] = []
    heading_stack: list[tuple[int, str]] = []

    current_level = 0
    current_title = ""
    current_path: tuple[str, ...] = ()
    current_blocks: list[_Block] = []

    def flush_section():
        nonlocal current_blocks
        sections.append(
            _Section(
                level=current_level,
                title=current_title,
                path=current_path,
                blocks=tuple(current_blocks),
            )
        )
        current_blocks = []

    def start_section(level: int, title: str):
        nonlocal current_level, current_title, current_path

        # close existing section
        flush_section()

        # update heading stack
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()
        heading_stack.append((level, title))

        current_level = level
        current_title = title
        current_path = tuple(t for _, t in heading_stack)

    # initial implicit section (pre-heading content)
    # (we'll flush it once we see the first heading or at end)
    for seg_text, is_code, seg_start in segments:
        if is_code:
            current_blocks.append(
                _Block(kind="code", text=seg_text, start=seg_start, end=seg_start + len(seg_text))
            )
            continue

        # non-code: scan headings line-by-line, with offsets
        lines = seg_text.splitlines(keepends=True)
        buf: list[str] = []
        buf_start = seg_start
        cursor = seg_start

        def flush_buf_as_blocks(cursor=cursor):
            nonlocal buf, buf_start
            if not buf:
                return
            block_text = "".join(buf)
            current_blocks.extend(_blocks_from_noncode_text(block_text, base_offset=buf_start))
            buf = []
            buf_start = cursor

        for ln in lines:
            m = _HEADING_RE.match(ln)
            if m:
                flush_buf_as_blocks()
                level = len(m.group(1))
                title = m.group(2).strip()
                start_section(level, title)
                cursor += len(ln)
                buf_start = cursor
                continue

            # regular line
            buf.append(ln)
            cursor += len(ln)

        flush_buf_as_blocks()

    # flush final section
    flush_section()

    # drop completely empty sections
    return [s for s in sections if any(b.text.strip() for b in s.blocks)]


def _chunk_kind(blocks: list[_Block]) -> str:
    """
    Determine the overall content type of a chunk from its constituent blocks.

    This is used to set the chunk_kind metadata field, which enables
    type-specific retrieval strategies or filtering.

    Args:
        blocks: List of _Block objects in the chunk.

    Returns:
        One of:
        - "empty": No non-whitespace content
        - "para", "list", "callout", "table", "code": Single block type
        - "mixed": Multiple different block types

    Example:
        >>> _chunk_kind([_Block(kind="para", ...), _Block(kind="para", ...)])
        'para'
        >>> _chunk_kind([_Block(kind="para", ...), _Block(kind="list", ...)])
        'mixed'
        >>> _chunk_kind([_Block(kind="code", ...)])
        'code'
    """
    kinds = {b.kind for b in blocks if b.text.strip()}
    if not kinds:
        return "empty"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


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
