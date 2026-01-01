from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rag.adapters.ingestion.loaders.text_loader import TextLoader


_WIKILINK_RE = re.compile(r"\[\[([^\]]+)\]\]")          # [[target]] or [[target|alias]]
_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")           # ![[target]] or ![[target|alias]]
_TAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9/_-]+)")     # #tag #a/b etc.

_TEXT_NOTE_EXTS = {".md", ".txt"}
_ATTACHMENT_EXTS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".pdf",
    ".mp3", ".wav", ".m4a",
    ".mp4", ".mov",
}

def split_obsidian_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """
    Parse YAML-like frontmatter delimited by leading '---' lines.
    Returns (frontmatter_dict, content).
    """
    if not text.startswith("---"):
        return {}, text

    lines = text.splitlines()
    if len(lines) < 3:
        return {}, text

    # find closing ---
    try:
        end_idx = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    except StopIteration:
        return {}, text

    fm_text = "\n".join(lines[1:end_idx]).strip()
    content = "\n".join(lines[end_idx + 1 :])

    fm: dict[str, Any] = {}
    if not fm_text:
        return fm, content

    # Try PyYAML if available; else do a tiny parser for common cases
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(fm_text)
        if isinstance(loaded, dict):
            fm = loaded
        else:
            fm = {}
    except Exception:
        # Minimal parsing: key: value, tags: [a, b] / tags: a
        for line in fm_text.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if not k:
                continue
            # crude list handling
            if v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                fm[k] = [x.strip().strip("'\"") for x in inner.split(",") if x.strip()]
            else:
                fm[k] = v.strip().strip("'\"")

    return fm, content


def _extract_inline_tags(text: str) -> list[str]:
    return sorted(set(m.group(1) for m in _TAG_RE.finditer(text)))


def _normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        # Allow "a, b" or "a"
        parts = [p.strip() for p in value.split(",")]
        return [p for p in parts if p]
    if isinstance(value, list):
        out: list[str] = []
        for x in value:
            if isinstance(x, str) and x.strip():
                out.append(x.strip())
        return out
    return []


def classify_note(file_name: str, frontmatter: dict[str, Any], tags: list[str]) -> str:
    name = file_name.lower()
    if "moc" in name:
        return "moc"
    if str(frontmatter.get("type", "")).lower() == "moc":
        return "moc"
    if any(t.lower() == "moc" for t in tags):
        return "moc"
    return "note"


def _split_fenced_code_blocks(text: str) -> list[tuple[str, bool]]:
    """
    Splits text into segments: (segment_text, is_code_block).
    Preserves fenced code blocks delimited by ``` (Obsidian default).
    """
    lines = text.splitlines(keepends=True)
    segs: list[tuple[str, bool]] = []
    buf: list[str] = []
    in_code = False

    def flush():
        nonlocal buf
        if buf:
            segs.append(("".join(buf), in_code))
            buf = []

    for line in lines:
        if line.lstrip().startswith("```"):
            # fence line belongs to current segment
            buf.append(line)
            flush()
            in_code = not in_code
            continue
        buf.append(line)

    flush()
    return segs


def _strip_wikilinks_outside_code(text: str) -> str:
    """
    Replace [[target]] / [[target|alias]] with human-readable text (alias or target),
    but only outside fenced code blocks.
    """
    out: list[str] = []
    for seg, is_code in _split_fenced_code_blocks(text):
        if is_code:
            out.append(seg)
            continue

        def repl(m: re.Match[str]) -> str:
            inner = m.group(1)
            # inner can be "target|alias" or "target#heading|alias"
            if "|" in inner:
                target, alias = inner.split("|", 1)
                return alias.strip() or target.strip()
            return inner.strip()

        out.append(_WIKILINK_RE.sub(repl, seg))
    return "".join(out)


def _resolve_embed_target(vault_root: Path, current_file: Path, target: str) -> Optional[Path]:
    """
    Resolve Obsidian embed target:
      - 'Note Name' -> find 'Note Name.md' anywhere (best-effort)
      - 'path/to/note.md' -> relative to vault root or current dir
      - may include headings/aliases after # or |
    """
    target = target.split("|", 1)[0]
    target = target.split("#", 1)[0].strip()

    tpath = Path(target)

    candidates: list[Path] = []
    if tpath.suffix:
        candidates.append(tpath)
    else:
        # Prefer notes first
        candidates.append(tpath.with_suffix(".md"))
        # Also allow raw name (attachments)
        candidates.append(tpath)

    for cand in candidates:
        p1 = (current_file.parent / cand).resolve()
        if p1.exists() and p1.is_file():
            return p1

        p2 = (vault_root / cand).resolve()
        if p2.exists() and p2.is_file():
            return p2

        # Fallback: search by filename within vault
        name = cand.name if cand.suffix else cand.name  # same
        matches = list(vault_root.rglob(name))
        for m in matches:
            if m.is_file():
                return m.resolve()

    # If we tried "image" without suffix and didn't find it, also search for "image.*"
    if not tpath.suffix:
        matches = list(vault_root.rglob(f"{tpath.name}.*"))
        for m in matches:
            if m.is_file():
                return m.resolve()

    return None


@dataclass(frozen=True, slots=True)
class ObsidianMarkdownLoader:
    """
    Loads Obsidian-flavored markdown with:
      - frontmatter parsing
      - inline tag extraction
      - embed expansion (![[...]]), with recursion+cycle protection
      - wikilink stripping ([[...]] -> alias/target) outside code fences
      - preserves code blocks (included as-is)
    """
    vault_root: Path
    text_loader: TextLoader = TextLoader()
    expand_embeds: bool = True
    max_embed_depth: int = 4

    def load(self, path: Path) -> Optional[tuple[str, dict[str, Any]]]:
        raw = self.text_loader.load(path)
        if raw is None:
            return None

        frontmatter, content = split_obsidian_frontmatter(raw)
        fm_tags = _normalize_tags(frontmatter.get("tags"))

        if self.expand_embeds:
            content = self._expand_embeds(content, current_file=path, depth=0, seen=set())

        # Remove wikilinks outside code
        content = _strip_wikilinks_outside_code(content)

        inline_tags = _extract_inline_tags(content)

        meta: dict[str, Any] = {
            "frontmatter": frontmatter,
            "frontmatter_tags": fm_tags,   # list[str]
            "inline_tags": inline_tags,    # list[str]
            "classification": classify_note(path.name, frontmatter, fm_tags),
        }

        return content, meta

    def _expand_embeds(self, content: str, *, current_file: Path, depth: int, seen: set[str]) -> str:
        if depth >= self.max_embed_depth:
            return content

        out_lines: list[str] = []

        # expand embeds line-by-line; keep code fences untouched
        segments = _split_fenced_code_blocks(content)
        for seg_text, is_code in segments:
            if is_code:
                out_lines.append(seg_text)
                continue

            # Expand embeds within non-code text
            def repl(m: re.Match[str]) -> str:
                target = m.group(1)
                embed_path = _resolve_embed_target(self.vault_root, current_file, target)
                if embed_path is None:
                    return ""

                ext = embed_path.suffix.lower()

                # Attachments: do NOT inline; replace with a placeholder
                if ext in _ATTACHMENT_EXTS:
                    return f"\n[Embedded attachment: {embed_path.name}]\n"

                # Only inline known text-note types
                if ext not in _TEXT_NOTE_EXTS:
                    return f"\n[Embedded file (unsupported type): {embed_path.name}]\n"

                key = str(embed_path)
                if key in seen:
                    return ""

                seen.add(key)
                loaded = self.text_loader.load(embed_path)
                if loaded is None:
                    # This also covers binary-ish or too-large text loader cases
                    return f"\n[Embedded note unreadable: {embed_path.name}]\n"

                _, embedded_content = split_obsidian_frontmatter(loaded)
                embedded_expanded = self._expand_embeds(
                    embedded_content, current_file=embed_path, depth=depth + 1, seen=seen
                )

                return f"\n\n<!-- EMBED: {embed_path.name} -->\n{embedded_expanded}\n<!-- /EMBED -->\n\n"

            out_lines.append(_EMBED_RE.sub(repl, seg_text))

        return "".join(out_lines)
