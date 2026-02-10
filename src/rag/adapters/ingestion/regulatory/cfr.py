"""First-generation CFR normalizer (XML → markdown).

This was the original all-in-one module for ingesting eCFR XML.  It handles:

* Parsing ``DIV8 TYPE=SECTION`` elements out of raw eCFR XML.
* Rendering each section to Obsidian-compatible markdown with YAML frontmatter.
* Building a citation manifest that maps ``citation_key`` → file path.

Cross-reference detection and rewriting utilities have been extracted to
``cross_references.py``, which is the canonical home for those functions.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import split_obsidian_frontmatter
from rag.adapters.ingestion.regulatory.cross_references import (
    extract_cross_references,
    rewrite_cross_references_to_wikilinks,
    section_sort_key,
)

# Captures the ``<part>.<section><suffix>`` portion of a section number.
_SECTION_RE = re.compile(r"(?P<section>\d+\.\d+[A-Za-z0-9-]*)")

# Detects subsection markers like ``(a)``, ``(1)``, ``(iv)`` at the start of a paragraph.
_SUBSECTION_MARKER_RE = re.compile(r"^\((?P<marker>[A-Za-z0-9ivxlcdmIVXLCDM]+)\)\s*(?P<body>.*)$")


@dataclass(frozen=True, slots=True)
class _ParsedSection:
    """Lightweight intermediate representation of one CFR section extracted from XML."""

    section: str  # e.g. "50.36"
    title: str  # human-readable section title
    paragraphs: tuple[str, ...]  # raw paragraph texts in document order


# --- XML helpers ------------------------------------------------------------


def _local_name(tag: str) -> str:
    """Strip XML namespace prefix, returning the local tag name in uppercase."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1].upper()
    return tag.upper()


def _collapse_whitespace(text: str) -> str:
    """Normalize internal whitespace and non-breaking spaces to single spaces."""
    return " ".join(text.replace("\u00a0", " ").split()).strip()


# --- XML section extraction -------------------------------------------------


def _extract_section_number(section_elem: ElementTree.Element) -> str | None:
    """Pull the section number from an eCFR ``DIV8`` element.

    Tries the ``N`` attribute first, then falls back to the ``<SECTNO>`` child.
    """
    n_attr = section_elem.attrib.get("N")
    if n_attr:
        match = _SECTION_RE.search(n_attr)
        if match:
            return match.group("section")

    for child in section_elem:
        if _local_name(child.tag) != "SECTNO":
            continue
        sectno_text = _collapse_whitespace(" ".join(child.itertext()))
        match = _SECTION_RE.search(sectno_text)
        if match:
            return match.group("section")

    return None


def _extract_section_title(section_elem: ElementTree.Element, section_number: str) -> str:
    """Extract the human-readable title from a ``<HEAD>`` child element."""
    head_text = ""
    for child in section_elem:
        if _local_name(child.tag) == "HEAD":
            head_text = _collapse_whitespace(" ".join(child.itertext()))
            break

    if not head_text:
        return f"Section {section_number}"

    prefix_pattern = rf"^§\s*{re.escape(section_number)}\s*[-—.]?\s*"
    title = re.sub(prefix_pattern, "", head_text).strip()
    return title.rstrip(".") if title else f"Section {section_number}"


def _extract_section_paragraphs(section_elem: ElementTree.Element) -> tuple[str, ...]:
    """Collect all ``<P>``, ``<P-1>``, and ``<FP>`` text from a section element."""
    paragraphs: list[str] = []
    for node in section_elem.iter():
        if _local_name(node.tag) not in {"P", "P-1", "FP"}:
            continue
        paragraph = _collapse_whitespace(" ".join(node.itertext()))
        if paragraph:
            paragraphs.append(paragraph)
    return tuple(paragraphs)


def _parse_section(section_elem: ElementTree.Element, expected_part: int) -> _ParsedSection | None:
    """Parse a single ``DIV8`` element, returning ``None`` if it belongs to a different part."""
    section_number = _extract_section_number(section_elem)
    if not section_number:
        return None

    if not section_number.startswith(f"{expected_part}."):
        return None

    return _ParsedSection(
        section=section_number,
        title=_extract_section_title(section_elem, section_number),
        paragraphs=_extract_section_paragraphs(section_elem),
    )


# --- Markdown rendering -----------------------------------------------------


def _subsection_heading_level(marker: str) -> int | None:
    """Map a subsection marker to a markdown heading level.

    ``(a)`` → ``##``, ``(1)`` → ``###``, ``(iv)`` → ``####``.
    Returns ``None`` for unrecognized markers.
    """
    marker_lc = marker.lower()
    roman_numerals = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}

    if marker.isdigit():
        return 3
    if marker_lc in roman_numerals:
        return 4
    if len(marker) == 1 and marker.isalpha():
        return 2
    return None


def _render_frontmatter(frontmatter: dict[str, Any]) -> list[str]:
    """Serialize *frontmatter* dict to YAML-in-markdown ``---`` block lines."""
    lines = ["---"]
    for key in (
        "regime",
        "instrument",
        "instrument_version",
        "part",
        "section",
        "title",
        "citation_key",
        "source_url",
        "source_revision",
        "effective_date",
        "corpus",
    ):
        lines.append(f"{key}: {json.dumps(frontmatter[key], ensure_ascii=False)}")

    cross_refs = frontmatter.get("cross_references", [])
    refs_json = ", ".join(json.dumps(ref, ensure_ascii=False) for ref in cross_refs)
    lines.append(f"cross_references: [{refs_json}]")

    lines.append("---")
    lines.append("")
    return lines


def _render_markdown_document(
    *,
    section: _ParsedSection,
    regime: str,
    instrument: str,
    instrument_version: str,
    part: int,
    source_url: str,
    source_revision: str,
    effective_date: str,
) -> str:
    """Render a single parsed section to a complete markdown document.

    The output includes YAML frontmatter (with citation key, cross-references,
    provenance fields), a top-level heading, and body paragraphs.  Subsection
    markers like ``(a)`` are promoted to markdown headings so that the
    structural chunker can split on them.
    """
    instrument_display = instrument.replace("-", " ")
    citation_key = f"{instrument_display} §{section.section}"

    rewritten_paragraphs = [rewrite_cross_references_to_wikilinks(p) for p in section.paragraphs]

    cross_references: set[str] = set()
    for paragraph in rewritten_paragraphs:
        cross_references.update(extract_cross_references(paragraph))

    frontmatter = {
        "regime": regime,
        "instrument": instrument,
        "instrument_version": instrument_version,
        "part": str(part),
        "section": section.section,
        "title": section.title,
        "citation_key": citation_key,
        "source_url": source_url,
        "source_revision": source_revision,
        "effective_date": effective_date,
        "corpus": "regulatory",
        "cross_references": sorted(
            cross_references,
            key=lambda citation: section_sort_key(citation.split("§", 1)[1].strip()),
        ),
    }

    lines: list[str] = []
    lines.extend(_render_frontmatter(frontmatter))
    lines.append(f"# {citation_key} — {section.title}")
    lines.append("")

    for paragraph in rewritten_paragraphs:
        marker_match = _SUBSECTION_MARKER_RE.match(paragraph)
        if marker_match is not None:
            marker = marker_match.group("marker")
            level = _subsection_heading_level(marker)
            if level is not None:
                lines.append(f"{'#' * level} ({marker})")
                lines.append("")
                body = marker_match.group("body").strip()
                if body:
                    lines.append(body)
                    lines.append("")
                continue

        lines.append(paragraph)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --- Public API -------------------------------------------------------------


def normalize_cfr_part(
    *,
    raw_source_path: Path,
    output_dir: Path,
    regime: str,
    instrument: str,
    part: int,
    instrument_version: str,
    source_url: str,
    source_revision: str,
    effective_date: str,
) -> list[Path]:
    """Normalize a CFR part XML source into canonical section markdown files.

    Parses every ``DIV8 TYPE=SECTION`` element from *raw_source_path*, renders
    each to markdown under ``<output_dir>/part-<part>/``, and returns the list
    of written file paths.
    """
    xml_text = raw_source_path.read_text(encoding="utf-8")
    root = ElementTree.fromstring(xml_text)

    parsed_sections: list[_ParsedSection] = []
    for elem in root.iter():
        if _local_name(elem.tag) != "DIV8":
            continue
        if str(elem.attrib.get("TYPE", "")).upper() != "SECTION":
            continue

        parsed = _parse_section(elem, expected_part=part)
        if parsed is not None:
            parsed_sections.append(parsed)

    if not parsed_sections:
        raise ValueError(f"No CFR sections found for part {part} in {raw_source_path}")

    part_dir = output_dir / f"part-{part}"
    part_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for section in sorted(parsed_sections, key=lambda item: section_sort_key(item.section)):
        out_path = part_dir / f"{section.section}.md"
        markdown = _render_markdown_document(
            section=section,
            regime=regime,
            instrument=instrument,
            instrument_version=instrument_version,
            part=part,
            source_url=source_url,
            source_revision=source_revision,
            effective_date=effective_date,
        )
        out_path.write_text(markdown, encoding="utf-8")
        output_paths.append(out_path)

    return output_paths


def build_citation_manifest(output_dir: Path) -> dict[str, str]:
    """Map ``citation_key`` → relative file path for every markdown file under *output_dir*.

    Reads YAML frontmatter from each ``.md`` file and extracts the
    ``citation_key`` field.  The resulting dict is sorted by section number.
    """
    manifest: dict[str, str] = {}

    for path in sorted(output_dir.rglob("*.md")):
        raw = path.read_text(encoding="utf-8")
        frontmatter, _ = split_obsidian_frontmatter(raw)
        citation_key = frontmatter.get("citation_key")
        if not isinstance(citation_key, str) or not citation_key.strip():
            continue
        manifest[citation_key.strip()] = path.relative_to(output_dir).as_posix()

    return dict(sorted(manifest.items(), key=lambda item: section_sort_key(item[0].split("§", 1)[1].strip())))
