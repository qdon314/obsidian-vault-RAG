"""Second-generation CFR normalizer: ``ParsedSection`` → markdown.

Consumes the structured output of ``ecfr_parser.parse_ecfr_xml`` and renders
each section to an Obsidian-compatible markdown file with:

* YAML frontmatter carrying provenance and citation metadata.
* A top-level ``# <citation_key> — <title>`` heading.
* Subsection markers (``(a)``, ``(1)``, …) promoted to markdown headings so
  the structural chunker can split on them.
* Bare CFR references rewritten as wikilinks for Obsidian graph linking.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.adapters.ingestion.loaders.obsidian_markdown_loader import split_obsidian_frontmatter
from rag.adapters.ingestion.regulatory.cross_references import (
    extract_cross_references,
    rewrite_cross_references_to_wikilinks,
    section_sort_key,
)
from rag.adapters.ingestion.regulatory.ecfr_parser import ParsedSection

_SUBSECTION_PREFIX_RE = re.compile(r"^\([a-zA-Z0-9ivxlcdmIVXLCDM]+\)\s*")
# Maps ``ParsedParagraph.level`` → markdown heading prefix.
_HEADING_LEVEL = {1: "##", 2: "###", 3: "####", 4: "#####"}


@dataclass(frozen=True, slots=True)
class NormalizationConfig:
    """Provenance fields stamped into the YAML frontmatter of every output file."""

    regime: str  # e.g. "us-nrc"
    instrument: str  # e.g. "10-CFR"
    instrument_version: str  # e.g. "2025-01-17"
    source_url: str  # eCFR API URL the XML was fetched from
    source_revision: str  # git-friendly revision identifier
    effective_date: str  # regulatory effective date


def _strip_prefix(text: str) -> str:
    """Remove a leading subsection marker like ``(a) `` from paragraph text."""
    return _SUBSECTION_PREFIX_RE.sub("", text)


def _build_frontmatter(
    section: ParsedSection,
    config: NormalizationConfig,
    *,
    cross_references: list[str],
) -> list[str]:
    """Build YAML frontmatter lines (including delimiters) for a section."""
    citation_key = f"{config.instrument.replace('-', ' ')} §{section.section_number}"
    data: dict[str, object] = {
        "regime": config.regime,
        "instrument": config.instrument,
        "instrument_version": config.instrument_version,
        "part": section.part_number,
        "section": section.section_number,
        "title": section.title,
        "citation_key": citation_key,
        "source_url": config.source_url,
        "source_revision": config.source_revision,
        "effective_date": config.effective_date,
        "corpus": "regulatory",
        "cross_references": cross_references,
    }

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
        lines.append(f"{key}: {json.dumps(data[key], ensure_ascii=False)}")
    refs_json = ", ".join(json.dumps(ref, ensure_ascii=False) for ref in cross_references)
    lines.append(f"cross_references: [{refs_json}]")
    lines.append("---")
    lines.append("")
    return lines


def normalize_section_to_markdown(section: ParsedSection, config: NormalizationConfig) -> str:
    """Render a single ``ParsedSection`` to a complete markdown document string.

    Rewrites CFR references to wikilinks, extracts cross-references for
    frontmatter, and promotes subsection markers to markdown headings.
    """
    rewritten = [rewrite_cross_references_to_wikilinks(paragraph.text) for paragraph in section.paragraphs]
    cross_refs: set[str] = set()
    for paragraph in rewritten:
        cross_refs.update(extract_cross_references(paragraph))

    lines: list[str] = []
    lines.extend(
        _build_frontmatter(
            section,
            config,
            cross_references=sorted(
                cross_refs,
                key=lambda c: section_sort_key(c.split("§", 1)[1].strip()),
            ),
        )
    )

    citation_key = f"{config.instrument.replace('-', ' ')} §{section.section_number}"
    lines.append(f"# {citation_key} — {section.title}")
    lines.append("")

    for paragraph, parsed in zip(rewritten, section.paragraphs, strict=True):
        if parsed.level > 0 and parsed.prefix is not None:
            lines.append(f"{_HEADING_LEVEL.get(parsed.level, '##')} ({parsed.prefix})")
            lines.append("")
            body = _strip_prefix(paragraph).strip()
            if body:
                lines.append(body)
                lines.append("")
            continue

        lines.append(paragraph)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def normalize_part(
    sections: list[ParsedSection],
    config: NormalizationConfig,
    output_dir: Path,
) -> list[Path]:
    """Normalize all sections of a CFR part and write them to *output_dir*.

    Returns the list of written file paths, sorted by section number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for section in sorted(sections, key=lambda s: section_sort_key(s.section_number)):
        out_path = output_dir / f"{section.section_number}.md"
        out_path.write_text(normalize_section_to_markdown(section, config), encoding="utf-8")
        written.append(out_path)
    return written


def parse_frontmatter_citation_key(path: Path) -> str | None:
    """Read *path* and return its ``citation_key`` frontmatter value, or ``None``."""
    frontmatter, _ = split_obsidian_frontmatter(path.read_text(encoding="utf-8"))
    citation_key = frontmatter.get("citation_key")
    if isinstance(citation_key, str) and citation_key.strip():
        return citation_key.strip()
    return None
