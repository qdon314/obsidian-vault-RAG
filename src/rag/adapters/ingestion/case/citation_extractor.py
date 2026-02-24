"""Composable citation span extractors for NRC case documents.

Each ``extract_*`` function takes normalized text and returns a list of
:class:`~rag.domain.citations.CitationSpan` objects.  The functions are
designed to be composed into a pipeline::

    spans = (
        extract_cfr_sections(text)
        + extract_cfr_parts(text)
        + extract_dockets(text)
        + extract_adams_accessions(text)
        + extract_nuregs(text)
        + extract_generic_communications(text)
    )
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from rag.adapters.ingestion.case.text_normalizer import normalize_for_citation_extraction
from rag.domain.citations import CitationSpan

# ---------------------------------------------------------------------------
# CFR section extraction
# ---------------------------------------------------------------------------

# Matches: "10 CFR 50.46(b)(1)(ii)", "10 CFR §50.46", "10CFR50.46"
# Groups: title, section (with letter suffix), subsections
_CFR_SECTION_RE = re.compile(
    r"(?P<title>\d{1,2})\s*CFR\s*§?\s*"
    r"(?P<section>\d+\.\d+[A-Za-z]?)"
    r"(?P<subs>(?:\([A-Za-z0-9]+\))*)"
)

_SUBSECTION_RE = re.compile(r"\(([A-Za-z0-9]+)\)")


def _parse_subsections(subs_raw: str) -> list[str]:
    """Parse '(b)(1)(ii)' into ['b', '1', 'ii']."""
    return _SUBSECTION_RE.findall(subs_raw)


def extract_cfr_sections(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract strong CFR section references with explicit title anchor.

    Returns spans for patterns like ``10 CFR 50.46(b)(1)`` — requires the
    title number (e.g. ``10``) to be present to avoid false positives on
    bare decimal numbers.
    """
    spans: list[CitationSpan] = []
    for m in _CFR_SECTION_RE.finditer(text):
        title = m.group("title")
        section = m.group("section")
        subs_raw = m.group("subs")
        subs = _parse_subsections(subs_raw)

        key = f"cfr:{title}:{section}"
        if subs_raw:
            key += subs_raw.lower()

        spans.append(
            CitationSpan(
                kind="cfr",
                raw=m.group(0),
                key=key,
                start=m.start(),
                end=m.end(),
                confidence=0.95,
                source_field=source_field,
                attrs={
                    "title": int(title),
                    "part": int(section.split(".")[0]),
                    "section": section.split(".")[1].rstrip("abcdefghijklmnopqrstuvwxyz"),
                    "section_full": section,
                    "subsections": subs,
                },
            )
        )
    return spans


# ---------------------------------------------------------------------------
# CFR part extraction
# ---------------------------------------------------------------------------

# "10 CFR Part 50" — with explicit title
_CFR_PART_STRONG_RE = re.compile(
    r"(?P<title>\d{1,2})\s*CFR\s+Part\s+(?P<part>\d+)",
    re.IGNORECASE,
)

# "Part 50" — without title (assumes Title 10)
# No lookbehind needed: the `covered` set in extract_cfr_parts() prevents
# double-matching "Part 50" inside an already-matched "10 CFR Part 50".
_CFR_PART_WEAK_RE = re.compile(
    r"Part\s+(?P<part>\d+)",
    re.IGNORECASE,
)


def extract_cfr_parts(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract CFR part references like '10 CFR Part 50' or bare 'Part 50'."""
    spans: list[CitationSpan] = []
    # Track which offsets are already covered by strong matches
    covered: set[int] = set()

    for m in _CFR_PART_STRONG_RE.finditer(text):
        title = m.group("title")
        part = m.group("part")
        spans.append(
            CitationSpan(
                kind="cfrpart",
                raw=m.group(0),
                key=f"cfrpart:{title}:{part}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"title": int(title), "part": int(part)},
            )
        )
        covered.update(range(m.start(), m.end()))

    for m in _CFR_PART_WEAK_RE.finditer(text):
        if m.start() in covered:
            continue
        part = m.group("part")
        spans.append(
            CitationSpan(
                kind="cfrpart",
                raw=m.group(0),
                key=f"cfrpart:10:{part}",
                start=m.start(),
                end=m.end(),
                confidence=0.70,
                source_field=source_field,
                attrs={"title": 10, "part": int(part)},
            )
        )

    return spans


# ---------------------------------------------------------------------------
# CFR appendix extraction
# ---------------------------------------------------------------------------

# "10 CFR Part 50, Appendix B" or "Appendix B to 10 CFR Part 100"
_CFR_APPENDIX_RE = re.compile(
    r"(?:"
    r"(?P<title1>\d{1,2})\s*CFR\s+Part\s+(?P<part1>\d+)\s*,?\s+Appendix\s+(?P<letter1>[A-Z])"
    r"|"
    r"Appendix\s+(?P<letter2>[A-Z])\s+to\s+(?P<title2>\d{1,2})\s*CFR\s+Part\s+(?P<part2>\d+)"
    r")",
    re.IGNORECASE,
)


def extract_cfr_appendices(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract CFR appendix references like '10 CFR Part 50, Appendix B'."""
    spans: list[CitationSpan] = []
    for m in _CFR_APPENDIX_RE.finditer(text):
        title = m.group("title1") or m.group("title2")
        part = m.group("part1") or m.group("part2")
        letter = (m.group("letter1") or m.group("letter2")).lower()

        spans.append(
            CitationSpan(
                kind="cfrapp",
                raw=m.group(0),
                key=f"cfrapp:{title}:{part}:appendix-{letter}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"title": int(title), "part": int(part), "appendix": letter},
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Docket number extraction
# ---------------------------------------------------------------------------

# "Docket No. 50-247" or "Docket Nos. 50-247 and 50-286"
_DOCKET_EXPLICIT_RE = re.compile(
    r"Docket\s+(?:Nos?\.?|Numbers?)\s+"
    r"(?P<docket>\d{1,2}-\d+)",
    re.IGNORECASE,
)

# Fixed-width 8-digit NRC docket: 05000247 → 50-247
# Valid facility-type prefixes: 050, 070, 072, 030, 040
_DOCKET_FIXED_RE = re.compile(r"\b(?P<docket>0[3457][02]\d{5})\b")

# Bare hyphenated form: 50-247, 70-7002
# Only match NRC facility-type prefixes to reduce false positives
_DOCKET_HYPHEN_RE = re.compile(r"\b(?P<docket>(?:50|70|72|30|40)-\d{3,5})\b")


def _normalize_fixed_docket(digits: str) -> str:
    """Convert 8-digit docket '05000247' to hyphenated '50-247'."""
    # Strip leading zero, split at position 3
    facility_type = str(int(digits[:3]))
    number = str(int(digits[3:]))
    return f"{facility_type}-{number}"


def extract_dockets(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract NRC docket number references."""
    spans: list[CitationSpan] = []
    covered: set[int] = set()

    # Explicit "Docket No." form (highest confidence)
    for m in _DOCKET_EXPLICIT_RE.finditer(text):
        docket = m.group("docket")
        spans.append(
            CitationSpan(
                kind="docket",
                raw=m.group(0),
                key=f"docket:{docket}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"docket_number": docket},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Fixed-width 8-digit form
    for m in _DOCKET_FIXED_RE.finditer(text):
        if m.start() in covered:
            continue
        docket_hyp = _normalize_fixed_docket(m.group("docket"))
        spans.append(
            CitationSpan(
                kind="docket",
                raw=m.group(0),
                key=f"docket:{docket_hyp}",
                start=m.start(),
                end=m.end(),
                confidence=0.85,
                source_field=source_field,
                attrs={"docket_number": docket_hyp, "raw_fixed": m.group("docket")},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Bare hyphenated form (lower confidence — more ambiguous)
    for m in _DOCKET_HYPHEN_RE.finditer(text):
        if m.start() in covered:
            continue
        docket = m.group("docket")
        spans.append(
            CitationSpan(
                kind="docket",
                raw=m.group(0),
                key=f"docket:{docket}",
                start=m.start(),
                end=m.end(),
                confidence=0.75,
                source_field=source_field,
                attrs={"docket_number": docket},
            )
        )

    return spans


# ---------------------------------------------------------------------------
# ADAMS accession number extraction
# ---------------------------------------------------------------------------

# Modern ADAMS: ML + exactly 9 alphanumeric chars
# Format: ML + 2-digit year + 3-digit Julian day + 1 alpha + 3-digit sequence
# e.g. ML021910673, ML20108D163, ML051600165
_ADAMS_MODERN_RE = re.compile(r"\b(?P<acc>[Mm][Ll][0-9A-Za-z]{9})\b")

# Legacy ADAMS: exactly 10 digits, first digit typically 7-9 (older era)
_ADAMS_LEGACY_RE = re.compile(r"\b(?P<acc>[789]\d{9})\b")


def extract_adams_accessions(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract ADAMS accession number references."""
    spans: list[CitationSpan] = []
    covered: set[int] = set()

    # Modern ML-prefixed accessions
    for m in _ADAMS_MODERN_RE.finditer(text):
        acc = m.group("acc").upper()
        spans.append(
            CitationSpan(
                kind="adams",
                raw=m.group(0),
                key=f"adams:{acc}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"accession_number": acc},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Legacy 10-digit numeric accessions
    for m in _ADAMS_LEGACY_RE.finditer(text):
        if m.start() in covered:
            continue
        acc = m.group("acc")
        spans.append(
            CitationSpan(
                kind="adams",
                raw=m.group(0),
                key=f"adamslegacy:{acc}",
                start=m.start(),
                end=m.end(),
                confidence=0.60,
                source_field=source_field,
                attrs={"accession_number": acc, "is_legacy": True},
            )
        )

    return spans


# ---------------------------------------------------------------------------
# NUREG extraction
# ---------------------------------------------------------------------------

# NUREG-0800, NUREG/CR-1234, NUREG/BR-1234, NUREG/CP-1234
# Optional revision: NUREG-0800 Rev. 5 or NUREG-0800 Revision 5
_NUREG_RE = re.compile(
    r"NUREG\s*[-/]\s*(?:(?P<series>CR|BR|CP)\s*[-/]\s*)?(?P<number>\d{3,4})"
    r"(?:\s+(?:Rev\.?|Revision)\s*(?P<rev>\d+))?",
    re.IGNORECASE,
)


def extract_nuregs(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract NUREG document references."""
    spans: list[CitationSpan] = []
    for m in _NUREG_RE.finditer(text):
        series = m.group("series")
        number = m.group("number")
        rev = m.group("rev")

        key = f"nureg:{series.lower()}:{number}" if series else f"nureg:{number}"
        if rev:
            key += f":rev{rev}"

        spans.append(
            CitationSpan(
                kind="nureg",
                raw=m.group(0),
                key=key,
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={
                    "series": series.upper() if series else None,
                    "number": number,
                    "revision": int(rev) if rev else None,
                },
            )
        )
    return spans


# ---------------------------------------------------------------------------
# Generic communication extraction (RIS, GL, IN)
# ---------------------------------------------------------------------------

# RIS 2004-03, GL 2004-01, IN 2004-05
# Also full names: "Regulatory Issue Summary 2004-03"
_GC_ABBREV_RE = re.compile(
    r"\b(?P<kind>RIS|GL|IN)\s+(?P<year>\d{4})-(?P<seq>\d{2,3})\b",
    re.IGNORECASE,
)

_GC_FULL_RE = re.compile(
    r"\b(?:Regulatory Issue Summary|Generic Letter|Information Notice)\s+(?P<year>\d{4})-(?P<seq>\d{2,3})\b",
    re.IGNORECASE,
)

_GC_KIND_MAP = {
    "regulatory issue summary": "ris",
    "generic letter": "gl",
    "information notice": "in",
    "ris": "ris",
    "gl": "gl",
    "in": "in",
}


def extract_generic_communications(
    text: str,
    *,
    source_field: str = "content",
) -> list[CitationSpan]:
    """Extract generic communication references (RIS, GL, IN)."""
    spans: list[CitationSpan] = []
    covered: set[int] = set()

    # Abbreviated forms (RIS, GL, IN)
    for m in _GC_ABBREV_RE.finditer(text):
        kind = m.group("kind").lower()
        year = m.group("year")
        seq = m.group("seq")

        spans.append(
            CitationSpan(
                kind=kind,
                raw=m.group(0),
                key=f"{kind}:{year}-{seq}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"kind": kind, "year": int(year), "sequence": seq},
            )
        )
        covered.update(range(m.start(), m.end()))

    # Full name forms
    for m in _GC_FULL_RE.finditer(text):
        if m.start() in covered:
            continue
        # Determine kind from matched text
        matched_text = m.group(0).lower()
        kind = None
        for name, abbrev in _GC_KIND_MAP.items():
            if name in matched_text:
                kind = abbrev
                break
        if kind is None:
            continue

        year = m.group("year")
        seq = m.group("seq")

        spans.append(
            CitationSpan(
                kind=kind,
                raw=m.group(0),
                key=f"{kind}:{year}-{seq}",
                start=m.start(),
                end=m.end(),
                confidence=0.90,
                source_field=source_field,
                attrs={"kind": kind, "year": int(year), "sequence": seq},
            )
        )

    return spans


# ---------------------------------------------------------------------------
# High-level extraction API
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CitationExtractionResult:
    """Result of running the full citation extraction pipeline on a document.

    Attributes:
        spans: All extracted citation spans (deduplicated by key).
        unique_keys: Set of unique canonical citation keys.
        by_kind: Spans grouped by citation kind (e.g., "cfr", "nureg", "docket").
    """

    spans: tuple[CitationSpan, ...] = ()
    unique_keys: frozenset[str] = field(default_factory=frozenset)
    by_kind: dict[str, tuple[CitationSpan, ...]] = field(default_factory=dict)


def extract_all_citations(
    text: str,
    *,
    source_field: str = "content",
    high_confidence_only: bool = False,
    confidence_threshold: float = 0.85,
) -> CitationExtractionResult:
    """Run the full citation extraction pipeline on *text*.

    This function normalizes the text, runs all extractors, deduplicates
    results by canonical key, and returns a structured result.

    Args:
        text: Raw document text to analyze.
        source_field: Which field this text came from ("title", "content", "metadata").
        high_confidence_only: If True, filter to spans with confidence >= threshold.
        confidence_threshold: Minimum confidence for high_confidence_only filter.

    Returns:
        A CitationExtractionResult containing all extracted spans.
    """
    # Normalize text for better extraction
    normalized = normalize_for_citation_extraction(text)

    # Run all extractors
    all_spans: list[CitationSpan] = []
    all_spans.extend(extract_cfr_sections(normalized, source_field=source_field))
    all_spans.extend(extract_cfr_parts(normalized, source_field=source_field))
    all_spans.extend(extract_cfr_appendices(normalized, source_field=source_field))
    all_spans.extend(extract_dockets(normalized, source_field=source_field))
    all_spans.extend(extract_adams_accessions(normalized, source_field=source_field))
    all_spans.extend(extract_nuregs(normalized, source_field=source_field))
    all_spans.extend(extract_generic_communications(normalized, source_field=source_field))

    # Filter by confidence if requested
    if high_confidence_only:
        all_spans = [s for s in all_spans if s.confidence >= confidence_threshold]

    # Deduplicate by key (keep highest confidence if duplicates)
    by_key: dict[str, CitationSpan] = {}
    for span in all_spans:
        if span.key not in by_key or span.confidence > by_key[span.key].confidence:
            by_key[span.key] = span

    deduped = tuple(by_key.values())
    unique_keys = frozenset(by_key.keys())

    # Group by kind
    by_kind: dict[str, tuple[CitationSpan, ...]] = {}
    kind_groups: dict[str, list[CitationSpan]] = defaultdict(list)
    for span in deduped:
        kind_groups[span.kind].append(span)
    by_kind = {k: tuple(v) for k, v in kind_groups.items()}

    return CitationExtractionResult(
        spans=deduped,
        unique_keys=unique_keys,
        by_kind=by_kind,
    )
