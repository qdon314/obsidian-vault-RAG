"""Case document normalizer: ``CaseDocument`` → Obsidian markdown.

Converts NRC case documents fetched from the ADAMS API into
Obsidian-compatible markdown files with:

* YAML frontmatter carrying provenance, classification, and citation metadata.
* A top-level ``# <accession_number> — <title>`` heading.
* Bare CFR references rewritten as wikilinks for Obsidian graph linking.
* Deterministic document categorisation via keyword rules.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from rag.adapters.ingestion.regulatory.cross_references import (
    extract_cross_references,
    rewrite_cross_references_to_wikilinks,
)
from rag.domain.case_documents import (
    CaseCategory,
    CaseClassification,
    CaseDocument,
    CaseMetadata,
    CaseSubcategory,
)

# =========================================================
# Configuration
# =========================================================


@dataclass(frozen=True, slots=True)
class CaseNormalizationConfig:
    """Provenance fields stamped into the YAML frontmatter of every output file."""

    regime: str = "us-nrc"
    corpus_label: str = "nrc-cases"
    source_url: str = ""
    source_revision: str = ""


# =========================================================
# Facility extraction
# =========================================================

# Matches common NRC title patterns:
#   "... - Facility Name, Unit 1"
#   "... - Facility Name Nuclear Generating Station"
#   "Facility Name - Inspection Report ..."
_FACILITY_TITLE_RE = re.compile(
    r"(?:^|\s-\s)"  # leading " - " separator or start
    r"(?P<facility>[A-Z][A-Za-z0-9 '.&/-]+?)"
    r"(?:\s*,\s*Units?\s*\d[\d,&\s]*)??"  # optional unit suffix
    r"(?:\s*-\s|$)",  # trailing separator or end
)

# Common NRC reactor/facility type keywords
_REACTOR_KEYWORDS: dict[str, str] = {
    "BWR": "BWR",
    "PWR": "PWR",
    "Boiling Water Reactor": "BWR",
    "Pressurized Water Reactor": "PWR",
    "AP1000": "AP1000",
    "ABWR": "ABWR",
    "Small Modular Reactor": "SMR",
    "SMR": "SMR",
    "Research Reactor": "Research Reactor",
    "Test Reactor": "Test Reactor",
    "Non-Power Reactor": "Non-Power Reactor",
}


def extract_facility_info(
    title: str,
    content: str | None = None,
) -> dict[str, str | None]:
    """Extract facility name and reactor type from a case document.

    Returns a dict with ``facility_name`` and ``reactor_type`` keys
    (values may be ``None`` when not determinable).
    """
    facility_name: str | None = None
    reactor_type: str | None = None

    # Try to pull facility from title
    m = _FACILITY_TITLE_RE.search(title)
    if m:
        name = m.group("facility").strip()
        # Reject matches that are clearly not facility names
        if len(name) > 3 and not name.startswith("Inspection"):
            facility_name = name

    # Scan for reactor type keywords in title + content
    search_text = title + (" " + content[:2000] if content else "")
    for keyword, rtype in _REACTOR_KEYWORDS.items():
        if keyword in search_text:
            reactor_type = rtype
            break

    return {"facility_name": facility_name, "reactor_type": reactor_type}


# =========================================================
# Document classification
# =========================================================

_WORD_BOUNDARY = r"(?<![A-Z0-9]){term}(?![A-Z0-9])"


def _has(text: str, term: str) -> bool:
    """Case-insensitive word-boundary match."""
    pat = _WORD_BOUNDARY.format(term=re.escape(term))
    return re.search(pat, text, flags=re.IGNORECASE) is not None


def _any_has(text: str, terms: list[str]) -> str | None:
    """Return the first matching term, or ``None``."""
    for t in terms:
        if _has(text, t):
            return t
    return None


def classify_case_document(
    *,
    title: str,
    doc_type: str | None = None,
    abstract: str | None = None,
) -> CaseClassification:
    """Deterministic, high-precision classifier.

    Prefers ``doc_type`` signals (when present), then falls back to
    title/abstract keyword rules.  Returns ``UNKNOWN`` if no
    high-confidence match is found.
    """
    t = " ".join([title or "", abstract or ""]).strip()
    doc_type_norm = (doc_type or "").strip().lower()

    # --- 1) doc_type-based rules (strongest signal) ---
    if doc_type_norm:
        if "licensee event report" in doc_type_norm or doc_type_norm == "ler":
            return CaseClassification(
                category=CaseCategory.INCIDENT,
                subcategory=CaseSubcategory.INCIDENT_LER,
                confidence=0.98,
                method="rules",
                reasons=(f"doc_type={doc_type_norm}",),
            )
        if "inspection report" in doc_type_norm:
            return CaseClassification(
                category=CaseCategory.INSPECTION,
                subcategory=CaseSubcategory.INSPECTION_ROUTINE_REPORT,
                confidence=0.95,
                method="rules",
                reasons=(f"doc_type={doc_type_norm}",),
            )
        if "notice of violation" in doc_type_norm:
            return CaseClassification(
                category=CaseCategory.ENFORCEMENT,
                subcategory=CaseSubcategory.ENFORCEMENT_NOTICE_OF_VIOLATION,
                confidence=0.97,
                method="rules",
                reasons=(f"doc_type={doc_type_norm}",),
            )

    # --- 2) Title/abstract keyword rules ---

    # Enforcement
    hit = _any_has(t, ["Notice of Violation", "NOV"])
    if hit:
        return CaseClassification(
            category=CaseCategory.ENFORCEMENT,
            subcategory=CaseSubcategory.ENFORCEMENT_NOTICE_OF_VIOLATION,
            confidence=0.96,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Civil Penalty"])
    if hit:
        return CaseClassification(
            category=CaseCategory.ENFORCEMENT,
            subcategory=CaseSubcategory.ENFORCEMENT_CIVIL_PENALTY,
            confidence=0.95,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Confirmatory Order"])
    if hit:
        return CaseClassification(
            category=CaseCategory.ENFORCEMENT,
            subcategory=CaseSubcategory.ENFORCEMENT_CONFIRMATORY_ORDER,
            confidence=0.95,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Demand for Information", "DFI"])
    if hit:
        return CaseClassification(
            category=CaseCategory.ENFORCEMENT,
            subcategory=CaseSubcategory.ENFORCEMENT_DEMAND_FOR_INFORMATION,
            confidence=0.95,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Adjudication
    hit = _any_has(t, ["Atomic Safety and Licensing Board", "ASLB"])
    if hit:
        return CaseClassification(
            category=CaseCategory.ADJUDICATION,
            subcategory=CaseSubcategory.ADJUDICATION_ASLB_DECISION,
            confidence=0.92,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Petition for Hearing", "Intervention"])
    if hit:
        return CaseClassification(
            category=CaseCategory.ADJUDICATION,
            subcategory=CaseSubcategory.ADJUDICATION_PETITION_FOR_HEARING,
            confidence=0.90,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Generic communications
    if _has(t, "Regulatory Issue Summary") or _has(t, "RIS"):
        return CaseClassification(
            category=CaseCategory.GENERIC_COMMUNICATION,
            subcategory=CaseSubcategory.GENERIC_COMMUNICATION_RIS,
            confidence=0.94,
            method="rules",
            reasons=("keyword=RIS",),
        )

    if _has(t, "Generic Letter") or _has(t, "GL"):
        return CaseClassification(
            category=CaseCategory.GENERIC_COMMUNICATION,
            subcategory=CaseSubcategory.GENERIC_COMMUNICATION_GENERIC_LETTER,
            confidence=0.93,
            method="rules",
            reasons=("keyword=Generic Letter/GL",),
        )

    if _has(t, "Information Notice") or _has(t, "IN"):
        return CaseClassification(
            category=CaseCategory.GENERIC_COMMUNICATION,
            subcategory=CaseSubcategory.GENERIC_COMMUNICATION_INFORMATION_NOTICE,
            confidence=0.92,
            method="rules",
            reasons=("keyword=Information Notice/IN",),
        )

    # Vendor / Part 21
    hit = _any_has(t, ["10 CFR Part 21", "Part 21"])
    if hit:
        return CaseClassification(
            category=CaseCategory.VENDOR_PART21,
            subcategory=CaseSubcategory.VENDOR_PART21_REPORT,
            confidence=0.93,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Inspection
    if _has(t, "Inspection Report"):
        return CaseClassification(
            category=CaseCategory.INSPECTION,
            subcategory=CaseSubcategory.INSPECTION_ROUTINE_REPORT,
            confidence=0.90,
            method="rules",
            reasons=("keyword=Inspection Report",),
        )

    hit = _any_has(t, ["95001", "95002", "95003"])
    if hit:
        sub = {
            "95001": CaseSubcategory.INSPECTION_SUPPLEMENTAL_95001,
            "95002": CaseSubcategory.INSPECTION_SUPPLEMENTAL_95002,
            "95003": CaseSubcategory.INSPECTION_SUPPLEMENTAL_95003,
        }[hit]
        return CaseClassification(
            category=CaseCategory.INSPECTION,
            subcategory=sub,
            confidence=0.90,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Licensing
    if _has(t, "License Amendment Request") or _has(t, "LAR"):
        return CaseClassification(
            category=CaseCategory.LICENSING,
            subcategory=CaseSubcategory.LICENSING_LICENSE_AMENDMENT_REQUEST,
            confidence=0.88,
            method="rules",
            reasons=("keyword=LAR",),
        )

    if (
        _has(t, "Safety Evaluation")
        or _has(t, "Safety Evaluation Report")
        or _has(t, "SER")
    ):
        return CaseClassification(
            category=CaseCategory.LICENSING,
            subcategory=CaseSubcategory.LICENSING_SER,
            confidence=0.86,
            method="rules",
            reasons=("keyword=SER/Safety Evaluation",),
        )

    if _has(t, "Request for Additional Information") or _has(t, "RAI"):
        return CaseClassification(
            category=CaseCategory.LICENSING,
            subcategory=CaseSubcategory.LICENSING_RAI,
            confidence=0.86,
            method="rules",
            reasons=("keyword=RAI",),
        )

    # Rulemaking
    if _has(t, "Federal Register") or _has(t, "Proposed Rule") or _has(t, "Final Rule"):
        sub = CaseSubcategory.RULEMAKING_FEDERAL_REGISTER_NOTICE
        if _has(t, "Proposed Rule"):
            sub = CaseSubcategory.RULEMAKING_PROPOSED_RULE
        elif _has(t, "Final Rule"):
            sub = CaseSubcategory.RULEMAKING_FINAL_RULE
        return CaseClassification(
            category=CaseCategory.RULEMAKING,
            subcategory=sub,
            confidence=0.88,
            method="rules",
            reasons=("keyword=Rulemaking",),
        )

    # Incident
    if _has(t, "Event Notification") or _has(t, "EN"):
        return CaseClassification(
            category=CaseCategory.INCIDENT,
            subcategory=CaseSubcategory.INCIDENT_EVENT_NOTIFICATION,
            confidence=0.85,
            method="rules",
            reasons=("keyword=Event Notification/EN",),
        )

    # Security (public)
    if _has(t, "Security Order") or _has(t, "Safeguards Information") or _has(t, "SGI"):
        return CaseClassification(
            category=CaseCategory.SECURITY,
            subcategory=CaseSubcategory.SECURITY_ORDER_PUBLIC,
            confidence=0.80,
            method="rules",
            reasons=("keyword=Security Order/SGI",),
        )

    return CaseClassification(
        category=CaseCategory.UNKNOWN,
        subcategory=CaseSubcategory.UNKNOWN,
        confidence=0.0,
        method="rules",
        reasons=(),
    )


# =========================================================
# Metadata extraction
# =========================================================

# Matches CFR part references like "10 CFR Part 50" or "Part 21"
_PART_RE = re.compile(r"(?:10\s*CFR\s*)?Part\s+(\d+)", re.IGNORECASE)

# Matches CFR section references like "10 CFR 50.46" or "§50.46"
_SECTION_RE = re.compile(
    r"(?:(\d+)\s*CFR\s*)?§?\s*(\d+\.\d+[A-Za-z0-9-]*)",
)


def extract_case_metadata(doc: CaseDocument) -> CaseMetadata:
    """Build enriched :class:`CaseMetadata` from a :class:`CaseDocument`.

    Runs the deterministic classifier, extracts regulation references,
    and copies docket numbers from the document.
    """
    classification = classify_case_document(
        title=doc.title,
        doc_type=doc.document_type,
    )

    # Extract regulation parts and sections from content + title
    search_text = doc.title + (" " + doc.content if doc.content else "")

    parts: set[str] = set()
    for m in _PART_RE.finditer(search_text):
        parts.add(m.group(1))

    sections: set[str] = set()
    for m in _SECTION_RE.finditer(search_text):
        sections.add(m.group(2))

    return CaseMetadata(
        case_category=classification.category,
        case_subcategory=classification.subcategory,
        case_category_method=classification.method,
        case_category_confidence=classification.confidence,
        case_category_reasons=classification.reasons,
        case_signals=(),
        regulation_parts=tuple(sorted(parts)),
        regulation_sections=tuple(sorted(sections)),
        dockets=doc.docket_numbers,
    )


# =========================================================
# Frontmatter
# =========================================================


def _build_frontmatter(
    doc: CaseDocument,
    meta: CaseMetadata,
    config: CaseNormalizationConfig,
    *,
    cross_references: list[str],
    facility_info: dict[str, str | None],
) -> list[str]:
    """Build YAML frontmatter lines (including ``---`` delimiters)."""
    lines = ["---"]

    kv: list[tuple[str, object]] = [
        ("corpus", "case"),
        ("regime", config.regime),
        ("corpus_label", config.corpus_label),
        ("accession_number", doc.accession_number),
        ("title", doc.title),
        ("document_type", doc.document_type),
        ("document_date", doc.document_date.isoformat() if doc.document_date else ""),
        ("case_category", meta.case_category.value),
        ("case_subcategory", meta.case_subcategory.value),
        ("case_category_confidence", meta.case_category_confidence),
    ]

    if facility_info.get("facility_name"):
        kv.append(("facility_name", facility_info["facility_name"]))
    if facility_info.get("reactor_type"):
        kv.append(("reactor_type", facility_info["reactor_type"]))

    for key, value in kv:
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")

    # List fields
    for key, items in [
        ("dockets", list(meta.dockets)),
        ("regulation_parts", list(meta.regulation_parts)),
        ("regulation_sections", list(meta.regulation_sections)),
        ("cross_references", cross_references),
    ]:
        items_json = ", ".join(json.dumps(item, ensure_ascii=False) for item in items)
        lines.append(f"{key}: [{items_json}]")

    if config.source_url:
        lines.append(f"source_url: {json.dumps(config.source_url, ensure_ascii=False)}")
    if config.source_revision:
        lines.append(
            f"source_revision: {json.dumps(config.source_revision, ensure_ascii=False)}"
        )

    lines.append("---")
    lines.append("")
    return lines


# =========================================================
# Public API
# =========================================================


def normalize_case_document_to_markdown(
    doc: CaseDocument,
    config: CaseNormalizationConfig,
) -> str:
    """Render a :class:`CaseDocument` to a complete markdown document string.

    Extracts metadata, classifies the document, rewrites CFR references
    to wikilinks, and builds YAML frontmatter.
    """
    meta = extract_case_metadata(doc)
    facility_info = extract_facility_info(doc.title, doc.content)

    # Rewrite CFR references in the body content
    body = doc.content or ""
    body_rewritten = rewrite_cross_references_to_wikilinks(body)

    # Extract cross-references from the rewritten body
    cross_refs = extract_cross_references(body_rewritten)

    lines: list[str] = []
    lines.extend(
        _build_frontmatter(
            doc,
            meta,
            config,
            cross_references=cross_refs,
            facility_info=facility_info,
        )
    )

    # Top-level heading
    lines.append(f"# {doc.accession_number} — {doc.title}")
    lines.append("")

    # Metadata summary section
    if meta.case_category != CaseCategory.UNKNOWN:
        lines.append(f"**Category:** {meta.case_category.value}")
        if meta.case_subcategory != CaseSubcategory.UNKNOWN:
            lines.append(f"**Subcategory:** {meta.case_subcategory.value}")
        lines.append("")

    if doc.docket_numbers:
        docket_str = ", ".join(doc.docket_numbers)
        lines.append(f"**Docket(s):** {docket_str}")
        lines.append("")

    if facility_info.get("facility_name"):
        lines.append(f"**Facility:** {facility_info['facility_name']}")
        if facility_info.get("reactor_type"):
            lines.append(f"**Reactor Type:** {facility_info['reactor_type']}")
        lines.append("")

    # Body content
    if body_rewritten.strip():
        lines.append("## Content")
        lines.append("")
        lines.append(body_rewritten.strip())
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def normalize_cases(
    docs: list[CaseDocument],
    config: CaseNormalizationConfig,
    output_dir: Path,
) -> list[Path]:
    """Normalize all case documents and write them to *output_dir*.

    Returns the list of written file paths, sorted by accession number.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for doc in sorted(docs, key=lambda d: d.accession_number):
        out_path = output_dir / f"{doc.accession_number}.md"
        out_path.write_text(
            normalize_case_document_to_markdown(doc, config),
            encoding="utf-8",
        )
        written.append(out_path)
    return written


__all__ = [
    "CaseNormalizationConfig",
    "classify_case_document",
    "extract_case_metadata",
    "extract_facility_info",
    "normalize_case_document_to_markdown",
    "normalize_cases",
]
