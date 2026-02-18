```python
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import List


# =========================
# Enums
# =========================

class CaseCategory(StrEnum):
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
    UNKNOWN = "unknown"
    OTHER = "other"

    # Enforcement
    ENFORCEMENT_NOTICE_OF_VIOLATION = "enforcement_notice_of_violation"
    ENFORCEMENT_CIVIL_PENALTY = "enforcement_civil_penalty"
    ENFORCEMENT_CONFIRMATORY_ORDER = "enforcement_confirmatory_order"

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

    # Rulemaking
    RULEMAKING_PROPOSED_RULE = "rulemaking_proposed_rule"
    RULEMAKING_FINAL_RULE = "rulemaking_final_rule"

    # Incident
    INCIDENT_LER = "incident_licensee_event_report"
    INCIDENT_EVENT_NOTIFICATION = "incident_event_notification"

    # Vendor / Part 21
    VENDOR_PART21_REPORT = "vendor_part21_report"


# =========================
# Validation
# =========================

def _validate_category_pair(
    category: CaseCategory,
    subcategory: CaseSubcategory,
) -> None:
    """
    Enforce that subcategory namespace matches category.
    Allows UNKNOWN and OTHER universally.
    """
    if subcategory in {CaseSubcategory.UNKNOWN, CaseSubcategory.OTHER}:
        return

    if not subcategory.value.startswith(category.value + "_"):
        raise ValueError(
            f"Subcategory '{subcategory.value}' "
            f"does not match category '{category.value}'"
        )


# =========================
# Dataclass
# =========================

@dataclass
class CaseMetadata:
    case_category: CaseCategory = CaseCategory.UNKNOWN
    case_subcategory: CaseSubcategory = CaseSubcategory.UNKNOWN

    case_category_method: str = "rules"  # "rules" | "llm" | "manual"
    case_category_confidence: float = 0.0
    case_category_reasons: List[str] = field(default_factory=list)

    case_signals: List[str] = field(default_factory=list)

    regulation_parts: List[str] = field(default_factory=list)
    regulation_sections: List[str] = field(default_factory=list)
    dockets: List[str] = field(default_factory=list)

    def validate(self) -> None:
        """
        Validate internal consistency.
        """
        _validate_category_pair(
            self.case_category,
            self.case_subcategory,
        )

    def to_dict(self) -> dict:
        """
        Flatten to JSON-safe dict for storage in Document.metadata.
        """
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
        }
```

```python
import re
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class CaseClassification:
    category: CaseCategory
    subcategory: CaseSubcategory
    confidence: float
    method: str
    reasons: tuple[str, ...]


_WORD_BOUNDARY = r"(?<![A-Z0-9]){term}(?![A-Z0-9])"

def _has(text: str, term: str) -> bool:
    # case-insensitive word-ish match
    pat = _WORD_BOUNDARY.format(term=re.escape(term))
    return re.search(pat, text, flags=re.IGNORECASE) is not None

def _any_has(text: str, terms: Iterable[str]) -> Optional[str]:
    for t in terms:
        if _has(text, t):
            return t
    return None

def classify_case_document(
    *,
    title: str,
    doc_type: Optional[str] = None,
    abstract: Optional[str] = None,
) -> CaseClassification:
    """
    Deterministic, high-precision classifier.
    Prefers doc_type signals (if present), then title/abstract keyword rules.
    Returns unknown if no high-confidence match.
    """
    t = " ".join([title or "", abstract or ""]).strip()

    reasons: list[str] = []
    doc_type_norm = (doc_type or "").strip().lower()

    # --- 1) Strong doc_type-based rules (if you have them) ---
    # NOTE: doc_type values vary by source. Start with conservative matching.
    if doc_type_norm:
        if "licensee event report" in doc_type_norm or "ler" == doc_type_norm:
            return CaseClassification(
                category=CaseCategory.incident,
                subcategory=CaseSubcategory.incident_ler,
                confidence=0.98,
                method="rules",
                reasons=(f"doc_type={doc_type_norm}",),
            )
        if "inspection report" in doc_type_norm:
            return CaseClassification(
                category=CaseCategory.inspection,
                subcategory=CaseSubcategory.inspection_routine_report,
                confidence=0.95,
                method="rules",
                reasons=(f"doc_type={doc_type_norm}",),
            )
        if "notice of violation" in doc_type_norm:
            return CaseClassification(
                category=CaseCategory.enforcement,
                subcategory=CaseSubcategory.enforcement_notice_of_violation,
                confidence=0.97,
                method="rules",
                reasons=(f"doc_type={doc_type_norm}",),
            )

    # --- 2) Title/abstract keyword rules (high-signal phrases) ---

    # Enforcement
    hit = _any_has(t, ["Notice of Violation", "NOV"])
    if hit:
        return CaseClassification(
            category=CaseCategory.enforcement,
            subcategory=CaseSubcategory.enforcement_notice_of_violation,
            confidence=0.96,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Civil Penalty"])
    if hit:
        return CaseClassification(
            category=CaseCategory.enforcement,
            subcategory=CaseSubcategory.enforcement_civil_penalty,
            confidence=0.95,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Confirmatory Order"])
    if hit:
        return CaseClassification(
            category=CaseCategory.enforcement,
            subcategory=CaseSubcategory.enforcement_confirmatory_order,
            confidence=0.95,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Demand for Information", "DFI"])
    if hit:
        return CaseClassification(
            category=CaseCategory.enforcement,
            subcategory=CaseSubcategory.enforcement_demand_for_information,
            confidence=0.95,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Adjudication
    hit = _any_has(t, ["Atomic Safety and Licensing Board", "ASLB"])
    if hit:
        return CaseClassification(
            category=CaseCategory.adjudication,
            subcategory=CaseSubcategory.adjudication_aslb_decision,
            confidence=0.92,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    hit = _any_has(t, ["Petition for Hearing", "Intervention"])
    if hit:
        return CaseClassification(
            category=CaseCategory.adjudication,
            subcategory=CaseSubcategory.adjudication_petition_for_hearing,
            confidence=0.90,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Generic communications
    if _has(t, "Regulatory Issue Summary") or _has(t, "RIS"):
        return CaseClassification(
            category=CaseCategory.generic_communication,
            subcategory=CaseSubcategory.generic_ris,
            confidence=0.94,
            method="rules",
            reasons=("keyword=RIS",),
        )

    if _has(t, "Generic Letter") or _has(t, "GL"):
        return CaseClassification(
            category=CaseCategory.generic_communication,
            subcategory=CaseSubcategory.generic_generic_letter,
            confidence=0.93,
            method="rules",
            reasons=("keyword=Generic Letter/GL",),
        )

    if _has(t, "Information Notice") or _has(t, "IN"):
        return CaseClassification(
            category=CaseCategory.generic_communication,
            subcategory=CaseSubcategory.generic_information_notice,
            confidence=0.92,
            method="rules",
            reasons=("keyword=Information Notice/IN",),
        )

    # Vendor / Part 21
    hit = _any_has(t, ["10 CFR Part 21", "Part 21"])
    if hit:
        return CaseClassification(
            category=CaseCategory.vendor_part21,
            subcategory=CaseSubcategory.vendor_part21_report,
            confidence=0.93,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Inspection
    if _has(t, "Inspection Report"):
        return CaseClassification(
            category=CaseCategory.inspection,
            subcategory=CaseSubcategory.inspection_routine_report,
            confidence=0.90,
            method="rules",
            reasons=("keyword=Inspection Report",),
        )

    hit = _any_has(t, ["95001", "95002", "95003"])
    if hit:
        sub = {
            "95001": CaseSubcategory.inspection_supplemental_95001,
            "95002": CaseSubcategory.inspection_supplemental_95002,
            "95003": CaseSubcategory.inspection_supplemental_95003,
        }[hit]
        return CaseClassification(
            category=CaseCategory.inspection,
            subcategory=sub,
            confidence=0.90,
            method="rules",
            reasons=(f"keyword={hit}",),
        )

    # Licensing
    if _has(t, "License Amendment Request") or _has(t, "LAR"):
        return CaseClassification(
            category=CaseCategory.licensing,
            subcategory=CaseSubcategory.licensing_license_amendment_request,
            confidence=0.88,
            method="rules",
            reasons=("keyword=LAR",),
        )

    if _has(t, "Safety Evaluation") or _has(t, "Safety Evaluation Report") or _has(t, "SER"):
        return CaseClassification(
            category=CaseCategory.licensing,
            subcategory=CaseSubcategory.licensing_ser,
            confidence=0.86,  # SER can appear in other contexts; keep slightly lower
            method="rules",
            reasons=("keyword=SER/Safety Evaluation",),
        )

    if _has(t, "Request for Additional Information") or _has(t, "RAI"):
        return CaseClassification(
            category=CaseCategory.licensing,
            subcategory=CaseSubcategory.licensing_rai,
            confidence=0.86,
            method="rules",
            reasons=("keyword=RAI",),
        )

    # Rulemaking
    if _has(t, "Federal Register") or _has(t, "Proposed Rule") or _has(t, "Final Rule"):
        # If you want to be more precise, split by proposed/final keywords.
        sub = CaseSubcategory.rulemaking_federal_register_notice
        if _has(t, "Proposed Rule"):
            sub = CaseSubcategory.rulemaking_proposed_rule
        elif _has(t, "Final Rule"):
            sub = CaseSubcategory.rulemaking_final_rule
        return CaseClassification(
            category=CaseCategory.rulemaking,
            subcategory=sub,
            confidence=0.88,
            method="rules",
            reasons=("keyword=Rulemaking",),
        )

    # Incident
    if _has(t, "Event Notification") or _has(t, "EN"):
        return CaseClassification(
            category=CaseCategory.incident,
            subcategory=CaseSubcategory.incident_event_notification,
            confidence=0.85,  # "EN" is ambiguous; consider requiring "Event Notification"
            method="rules",
            reasons=("keyword=Event Notification/EN",),
        )

    # Security (public)
    if _has(t, "Security Order") or _has(t, "Safeguards Information") or _has(t, "SGI"):
        return CaseClassification(
            category=CaseCategory.security,
            subcategory=CaseSubcategory.security_order_public,
            confidence=0.80,  # keep conservative
            method="rules",
            reasons=("keyword=Security Order/SGI",),
        )

    return CaseClassification(
        category=CaseCategory.unknown,
        subcategory=CaseSubcategory.unknown,
        confidence=0.0,
        method="rules",
        reasons=(),
    )

```