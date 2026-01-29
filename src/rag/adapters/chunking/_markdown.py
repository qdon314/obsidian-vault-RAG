"""
Shared markdown parsing utilities for chunking adapters.

This module provides the common parsing pipeline used by both
ObsidianStructuralChunker and ObsidianPropositionChunker:

    Document Text
         |
         v
    Stage 1: Code Block Isolation (_split_fenced_code_blocks)
         |
         v
    Stage 2: Section Parsing (_sections_from_markdown)
         |
         v
    Stage 3: Block Detection (_blocks_from_noncode_text)

Block Types
===========

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

Chunk IDs are generated using BLAKE2b hashing for stability and collision
resistance.  The hash incorporates doc_id, strategy_name, chunk_index,
character offsets, and section_path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import blake2b

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
        +-------------------------------------------------------------+
        |                         START                                |
        +----------------------------+--------------------------------+
                                     |
            +------------------------+------------------------+
            |                        |                        |
            v                        v                        v
        +-------+              +--------+              +---------+
        | para  |<------------>|  list  |<------------>| callout |
        +---+---+   blank line +----+---+   blank line +----+----+
            |                       |                       |
            |       +---------------+                       |
            |       |               |                       |
            |       v               v                       v
            |   +-------+      +--------+              +---------+
            +-->| table |      | FLUSH  |<-------------|  FLUSH  |
                +-------+      +--------+              +---------+

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
