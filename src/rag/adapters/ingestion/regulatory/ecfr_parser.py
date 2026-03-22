"""Second-generation eCFR XML parser.

Parses raw eCFR XML into ``ParsedSection`` / ``ParsedParagraph`` data objects
without performing any rendering.  This decouples parsing from the markdown
normalization step (handled by ``normalizer.py``), making both independently
testable.

The eCFR XML hierarchy is: ``DIV5 (PART) → DIV8 (SECTION) → P/FP (paragraph)``.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass

# Matches one or more leading subsection markers like ``(a)(1)(i)``.
_SUBSECTION_CHAIN_RE = re.compile(r"^\s*((?:\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)\s*)+)")
_SUBSECTION_TOKEN_RE = re.compile(r"\(([A-Za-z0-9ivxlcdmIVXLCDM]+)\)")

# Maps subsection prefix values to nesting levels.
# Level 1 = lowercase letter (a-z), Level 2 = digit, Level 3 = roman numeral,
# Level 4 = uppercase letter.  This mirrors the CFR subsection hierarchy.
_LEVEL_MAP: dict[str, int] = {}
for _char in "abcdefghijklmnopqrstuvwxyz":
    _LEVEL_MAP[_char] = 1
for _num in range(1, 100):
    _LEVEL_MAP[str(_num)] = 2
for _roman in ("i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"):
    _LEVEL_MAP[_roman] = 3
for _char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    _LEVEL_MAP[_char] = 4

# XML element tags to skip when extracting paragraph text (footnotes, superscripts).
_SKIP_TAGS = frozenset({"SU", "FTREF", "FTNT", "TNOTE"})


def _local_name(tag: str) -> str:
    """Strip XML namespace prefix, returning the local tag name in uppercase."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1].upper()
    return tag.upper()


# --- Domain objects ---------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CrossRef:
    """A cross-reference found in paragraph text."""

    target_citation: str  # canonical form, e.g. "10 CFR §50.55a" or "ASME BPV III"
    kind: str  # "cfr" | "incorporated_standard"


@dataclass(frozen=True, slots=True)
class SectionAmendment:
    """Amendment metadata from an XREF element at section level."""

    amendment_id: str  # XREF ID attribute (date-like, e.g. "20241230")
    ref_id: str  # XREF REFID attribute
    text: str  # human-readable amendment description


@dataclass(frozen=True, slots=True)
class ParsedParagraph:
    """One paragraph extracted from an eCFR section."""

    text: str  # full paragraph text with whitespace normalized
    level: int  # nesting depth (0 = no subsection prefix)
    prefix: str | None  # the raw prefix value, e.g. "a", "1", "iv"
    subsection_tokens: tuple[str, ...] = ()  # full leading chain, e.g. ("a", "1", "i")
    cross_references: tuple[CrossRef, ...] = ()  # cross-refs detected in text


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """One CFR section (e.g. § 50.36) with its paragraphs."""

    section_number: str  # e.g. "50.36"
    title: str  # human-readable title
    part_number: str  # e.g. "50"
    paragraphs: tuple[ParsedParagraph, ...] = ()
    amendments: tuple[SectionAmendment, ...] = ()  # XREF elements at section level


def _token_level(token: str) -> int:
    """Map one subsection token to its CFR nesting level."""
    if token in _LEVEL_MAP:
        return _LEVEL_MAP[token]
    return _LEVEL_MAP.get(token.lower(), 0)


def _classify_paragraph(text: str) -> tuple[int, str | None, tuple[str, ...]]:
    """Determine subsection level, prefix, and token chain for a paragraph."""
    match = _SUBSECTION_CHAIN_RE.match(text)
    if not match:
        return 0, None, ()

    tokens = tuple(_SUBSECTION_TOKEN_RE.findall(match.group(1)))
    if not tokens:
        return 0, None, ()

    prefix = tokens[-1]
    return _token_level(prefix), prefix, tokens


def _extract_text(elem: ET.Element) -> str:
    """Concatenate all text content of *elem*, skipping footnote/superscript tags."""
    parts: list[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        if _local_name(child.tag) in _SKIP_TAGS:
            if child.tail:
                parts.append(child.tail)
            continue
        if child.text:
            parts.append(child.text)
        if child.tail:
            parts.append(child.tail)
    return " ".join("".join(parts).split()).strip()


def _extract_part_number(part_elem: ET.Element) -> str:
    """Pull the numeric part number from a ``DIV5`` element's ``N`` attribute."""
    raw = part_elem.get("N", "")
    match = re.search(r"(\d+)", raw)
    return match.group(1) if match else raw


def _parse_section_head(head_text: str) -> tuple[str, str]:
    """Split a ``<HEAD>`` text like ``§ 50.36 Technical specifications.`` into ``("50.36", "Technical specifications")``."""
    cleaned = re.sub(r"^§\s*", "", head_text).strip()
    parts = cleaned.split(None, 1)
    section_number = parts[0].rstrip(".")
    title = parts[1].strip() if len(parts) > 1 else ""
    return section_number, title.rstrip(".")


def parse_ecfr_xml(xml_text: str) -> list[ParsedSection]:
    """Parse eCFR XML into a list of ``ParsedSection`` objects.

    Iterates over ``DIV5 TYPE=PART`` → ``DIV8 TYPE=SECTION`` elements,
    extracting the section number, title, and classified paragraphs.
    """
    root = ET.fromstring(xml_text)
    sections: list[ParsedSection] = []

    for part_elem in root.iter():
        if _local_name(part_elem.tag) != "DIV5" or part_elem.get("TYPE") != "PART":
            continue
        part_number = _extract_part_number(part_elem)

        for section_elem in part_elem.iter():
            if _local_name(section_elem.tag) != "DIV8" or section_elem.get("TYPE") != "SECTION":
                continue

            head_elem: ET.Element | None = None
            for child in section_elem:
                if _local_name(child.tag) == "HEAD":
                    head_elem = child
                    break
            if head_elem is None:
                continue

            section_number, title = _parse_section_head(_extract_text(head_elem))

            # Extract XREF amendment metadata.
            amendments: list[SectionAmendment] = []
            for child in section_elem:
                if _local_name(child.tag) == "XREF":
                    amendments.append(
                        SectionAmendment(
                            amendment_id=child.get("ID", ""),
                            ref_id=child.get("REFID", ""),
                            text=_extract_text(child),
                        )
                    )

            paragraphs: list[ParsedParagraph] = []

            for child in section_elem:
                if _local_name(child.tag) not in {
                    "P",
                    "P-1",
                    "P-2",
                    "FP",
                    "FP-1",
                    "FP-2",
                    "PSPACE",
                }:
                    continue
                text = _extract_text(child)
                if not text:
                    continue
                level, prefix, subsection_tokens = _classify_paragraph(text)
                paragraphs.append(
                    ParsedParagraph(
                        text=text,
                        level=level,
                        prefix=prefix,
                        subsection_tokens=subsection_tokens,
                    )
                )

            sections.append(
                ParsedSection(
                    section_number=section_number,
                    title=title,
                    part_number=part_number,
                    paragraphs=tuple(paragraphs),
                    amendments=tuple(amendments),
                )
            )

    return sections
