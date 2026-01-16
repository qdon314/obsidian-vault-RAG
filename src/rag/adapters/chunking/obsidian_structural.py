from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import blake2b
from typing import Any

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_FENCE_RE = re.compile(r"^\s*```")
_CALLOUT_START_RE = re.compile(r"^\s*>\s*\[!(\w+)\]\s*(.*)$")  # > [!note] Title
_BLOCKQUOTE_RE = re.compile(r"^\s*>\s?.*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|(\d+)[.)])\s+.+$")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_FENCE_LANG_RE = re.compile(r"^\s*```(\w+)?\s*$")


def _stable_hash(parts: list[str]) -> str:
    h = blake2b(digest_size=12)
    for p in parts:
        h.update(p.encode("utf-8", errors="ignore"))
        h.update(b"\x1f")
    return h.hexdigest()


def _split_fenced_code_blocks(text: str) -> list[tuple[str, bool, int]]:
    """
    Split text into segments: (segment_text, is_code_block, segment_start_char).
    Preserves ``` fences and returns offsets into original text.
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


@dataclass(frozen=True, slots=True)
class _Block:
    kind: str  # "para" | "list" | "callout" | "table" | "code"
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Section:
    level: int
    title: str
    path: tuple[str, ...]
    blocks: tuple[_Block, ...]


def _extract_code_language(code_block_text: str) -> str | None:
    # Try to infer from opening fence line: ```python
    first_line = code_block_text.splitlines()[0] if code_block_text else ""
    m = _FENCE_LANG_RE.match(first_line)
    if not m:
        return None
    lang = (m.group(1) or "").strip()
    return lang or None


def _blocks_from_noncode_text(text: str, base_offset: int) -> list[_Block]:
    """
    Parse a non-code segment into structural blocks with offsets.
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
    kinds = {b.kind for b in blocks if b.text.strip()}
    if not kinds:
        return "empty"
    if len(kinds) == 1:
        return next(iter(kinds))
    return "mixed"


@dataclass(frozen=True, slots=True)
class ObsidianStructuralChunker:
    """
    Obsidian-aware chunker:
      - heading-based sections
      - block-aware packing (para/list/callout/table/code)
      - stable chunk ids
    """

    target_chars: int = 4000
    hard_max_chars: int = 5200
    overlap_blocks: int = 1
    include_heading_preamble: bool = True

    # naming this explicitly helps stability/debugging
    strategy_name: str = "obsidian_structural_v1"

    def chunk(self, doc, *, metadata: Mapping[str, object] | None = None) -> list:
        # Merge metadata layers: caller overrides doc.metadata if provided
        merged_meta: dict[str, Any] = dict(doc.metadata)
        if metadata:
            merged_meta.update(dict(metadata))

        sections = _sections_from_markdown(doc.text)

        # For preamble grounding
        note_title = str(merged_meta.get("title") or merged_meta.get("file_name") or doc.uri)

        chunks = []
        chunk_index = 0

        for sec in sections:
            section_path = " > ".join(sec.path) if sec.path else None
            section_heading = sec.path[-1] if sec.path else None

            preamble = ""
            if self.include_heading_preamble:
                if sec.path:
                    preamble = f"Title: {note_title}\nPath: {section_path}\n\n"
                else:
                    preamble = f"Title: {note_title}\n\n"

            buf: list[_Block] = []
            buf_len = len(preamble)

            def flush(preamble=preamble, section_path=section_path, section_heading=section_heading):
                nonlocal buf, buf_len, chunk_index

                if not buf:
                    return

                start_char = min(b.start for b in buf)
                end_char = max(b.end for b in buf)

                body = "".join(b.text for b in buf).strip()
                if not body:
                    buf = []
                    buf_len = len(preamble)
                    return

                text = preamble + body

                # language: if pure code chunk, infer language
                lang = None
                if _chunk_kind(buf) == "code" and len(buf) == 1:
                    lang = _extract_code_language(buf[0].text)

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

                ch_meta = dict(merged_meta)
                ch_meta.update(
                    {
                        "chunk_kind": _chunk_kind(buf),
                        "chunk_strategy": self.strategy_name,
                    }
                )

                from rag.domain.models import Chunk  # move import here if you have circulars

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

                # overlap by last N blocks
                if self.overlap_blocks > 0:
                    buf = buf[-self.overlap_blocks :]
                    buf_len = len(preamble) + sum(len(b.text) for b in buf)
                else:
                    buf = []
                    buf_len = len(preamble)

            for b in sec.blocks:
                if not b.text.strip():
                    continue

                # Oversize paragraph: allow splitting ONLY paragraphs (rare)
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

                # If adding would exceed hard max, flush first
                if buf and (buf_len + len(b.text) > self.hard_max_chars):
                    flush()

                buf.append(b)
                buf_len += len(b.text)

                # soft target boundary
                if buf_len >= self.target_chars:
                    flush()

            flush()

        return chunks

    def get_config(self) -> dict[str, Any]:
        return {
            "backend": "obsidian_structural",
            "target_chars": self.target_chars,
            "hard_max_chars": self.hard_max_chars,
            "overlap_blocks": self.overlap_blocks,
            "include_heading_preamble": self.include_heading_preamble,
            "strategy_name": self.strategy_name,
        }
