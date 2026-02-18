"""Domain models for NRC case document ingestion.

These models represent case documents fetched from the ADAMS API,
their classification metadata, and fetch-run summaries.  All domain
objects are frozen dataclasses — use ``dataclasses.replace()`` for
modifications.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

# =========================================================
# Enums
# =========================================================


class CaseCategory(StrEnum):
    """Top-level classification of an NRC case document."""

    UNKNOWN = "unknown"
    LICENSING = "licensing"
    INSPECTION = "inspection"
    ENFORCEMENT = "enforcement"
    ADJUDICATION = "adjudication"
    RULEMAKING = "rulemaking"
    GENERIC_COMMUNICATION = "generic_communication"
    VENDOR_PART21 = "vendor_part21"
    DECOMMISSIONING = "decommissioning"
    INCIDENT = "incident"
    SECURITY = "security"
    OPERATIONS = "operations"
    QUALITY_ASSURANCE = "quality_assurance"


class CaseSubcategory(StrEnum):
    """Fine-grained subtype within a :class:`CaseCategory`.

    Member values are namespaced by their parent category
    (e.g. ``enforcement_notice_of_violation``).  The sentinel values
    ``UNKNOWN`` and ``OTHER`` are valid under any category.
    """

    UNKNOWN = "unknown"
    OTHER = "other"

    # Enforcement
    ENFORCEMENT_NOTICE_OF_VIOLATION = "enforcement_notice_of_violation"
    ENFORCEMENT_CIVIL_PENALTY = "enforcement_civil_penalty"
    ENFORCEMENT_CONFIRMATORY_ORDER = "enforcement_confirmatory_order"
    ENFORCEMENT_DEMAND_FOR_INFORMATION = "enforcement_demand_for_information"

    # Inspection
    INSPECTION_ROUTINE_REPORT = "inspection_routine_report"
    INSPECTION_SPECIAL = "inspection_special"
    INSPECTION_SUPPLEMENTAL_95001 = "inspection_supplemental_95001"
    INSPECTION_SUPPLEMENTAL_95002 = "inspection_supplemental_95002"
    INSPECTION_SUPPLEMENTAL_95003 = "inspection_supplemental_95003"

    # Licensing
    LICENSING_LICENSE_AMENDMENT_REQUEST = "licensing_license_amendment_request"
    LICENSING_SER = "licensing_safety_evaluation_report"
    LICENSING_RAI = "licensing_request_for_additional_information"

    # Adjudication
    ADJUDICATION_ASLB_DECISION = "adjudication_aslb_decision"
    ADJUDICATION_PETITION_FOR_HEARING = "adjudication_petition_for_hearing"

    # Rulemaking
    RULEMAKING_PROPOSED_RULE = "rulemaking_proposed_rule"
    RULEMAKING_FINAL_RULE = "rulemaking_final_rule"
    RULEMAKING_FEDERAL_REGISTER_NOTICE = "rulemaking_federal_register_notice"

    # Incident
    INCIDENT_LER = "incident_licensee_event_report"
    INCIDENT_EVENT_NOTIFICATION = "incident_event_notification"

    # Generic communication
    GENERIC_COMMUNICATION_RIS = "generic_communication_regulatory_issue_summary"
    GENERIC_COMMUNICATION_GENERIC_LETTER = "generic_communication_generic_letter"
    GENERIC_COMMUNICATION_INFORMATION_NOTICE = "generic_communication_information_notice"

    # Vendor / Part 21
    VENDOR_PART21_REPORT = "vendor_part21_report"

    # Security
    SECURITY_ORDER_PUBLIC = "security_order_public"


# =========================================================
# Validation
# =========================================================


def validate_category_pair(
    category: CaseCategory,
    subcategory: CaseSubcategory,
) -> None:
    """Enforce that *subcategory* belongs to *category*.

    The sentinel values ``UNKNOWN`` and ``OTHER`` are accepted under
    any category.  All other subcategory values must start with the
    category value followed by ``_``.

    Raises:
        ValueError: If the subcategory does not match the category.
    """
    if subcategory in {CaseSubcategory.UNKNOWN, CaseSubcategory.OTHER}:
        return

    if not subcategory.value.startswith(category.value + "_"):
        raise ValueError(
            f"Subcategory '{subcategory.value}' does not match category '{category.value}'"
        )


# =========================================================
# Dataclasses
# =========================================================


@dataclass(frozen=True, slots=True)
class CaseClassification:
    """Result of classifying a case document into category/subcategory."""

    category: CaseCategory
    subcategory: CaseSubcategory
    confidence: float
    method: str  # "rules" | "llm" | "manual"
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CaseMetadata:
    """Enriched metadata derived from a case document.

    Carries classification results, extracted regulatory references,
    and provenance signals used by downstream normalisation and query
    generation.
    """

    case_category: CaseCategory = CaseCategory.UNKNOWN
    case_subcategory: CaseSubcategory = CaseSubcategory.UNKNOWN

    case_category_method: str = "rules"
    case_category_confidence: float = 0.0
    case_category_reasons: tuple[str, ...] = ()

    case_signals: tuple[str, ...] = ()

    regulation_parts: tuple[str, ...] = ()
    regulation_sections: tuple[str, ...] = ()
    dockets: tuple[str, ...] = ()

    # New citation extraction fields
    citation_keys: tuple[str, ...] = ()  # Canonical citation keys (e.g., "cfr:10:50.46")
    citation_spans: tuple[Mapping[str, object], ...] = ()  # type: ignore[type-arg]  # Serialized CitationSpan objects

    def validate(self) -> None:
        """Check internal consistency.

        Raises:
            ValueError: If *case_subcategory* does not match *case_category*.
        """
        validate_category_pair(self.case_category, self.case_subcategory)

    def to_dict(self) -> dict[str, object]:
        """Flatten to a JSON-safe dict for storage in ``Document.metadata``."""
        self.validate()
        return {
            "case_category": self.case_category.value,
            "case_subcategory": self.case_subcategory.value,
            "case_category_method": self.case_category_method,
            "case_category_confidence": self.case_category_confidence,
            "case_category_reasons": list(self.case_category_reasons),
            "case_signals": list(self.case_signals),
            "regulation_parts": list(self.regulation_parts),
            "regulation_sections": list(self.regulation_sections),
            "dockets": list(self.dockets),
            "citation_keys": list(self.citation_keys),
            "citation_spans": list(self.citation_spans),
        }


@dataclass(frozen=True, slots=True)
class CaseDocument:
    """A case document fetched from the ADAMS API.

    Mirrors the data carried by
    :class:`~rag.ports.nrc_adams_client.AdamsDocument` but lives in
    the domain layer with parsed dates and enriched metadata.
    """

    accession_number: str  # e.g. "ML12345A678"
    title: str
    document_type: str  # raw ADAMS document type string

    document_date: datetime | None = None
    content: str | None = None

    author_name: str | None = None
    author_affiliation: str | None = None
    addressee: str | None = None
    addressee_affiliation: str | None = None

    docket_numbers: tuple[str, ...] = ()  # list in AdamsDocument; convert on construction
    license_numbers: tuple[str, ...] = ()  # list in AdamsDocument; convert on construction
    keywords: tuple[str, ...] = ()  # list in AdamsDocument; convert on construction

    url: str | None = None
    estimated_pages: int | None = None

    case_metadata: CaseMetadata = field(default_factory=CaseMetadata)


@dataclass(frozen=True, slots=True)
class FetchReport:
    """Summary of a case-document fetch run.

    Modelled after :class:`~rag.domain.models.IngestReport`.
    """

    total_requested: int
    fetched: int
    skipped: int
    failed: int
    by_document_type: Mapping[str, int] = field(default_factory=dict)
    errors: Sequence[str] = field(default_factory=tuple)
