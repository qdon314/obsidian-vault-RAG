"""Adapter to convert AdamsDocument (port) to CaseDocument (domain)."""

from __future__ import annotations

from datetime import datetime

from rag.domain.case_documents import CaseDocument, CaseMetadata
from rag.ports.nrc_adams_client import AdamsDocument


def adams_document_to_case_document(
    adams_doc: AdamsDocument,
    *,
    case_metadata: CaseMetadata | None = None,
) -> CaseDocument:
    """Convert port-layer AdamsDocument to domain-layer CaseDocument.

    Transforms API boundary data types (ISO date strings, lists) into
    domain-appropriate types (datetime, tuples). This adapter bridges
    the port and domain layers in the hexagonal architecture.

    Parameters
    ----------
    adams_doc:
        Raw document from ADAMS API client.
    case_metadata:
        Optional precomputed metadata; if None, uses default empty metadata.

    Returns
    -------
    CaseDocument with parsed dates and tuple-ified collections.

    Examples
    --------
    >>> from rag.ports.nrc_adams_client import AdamsDocument
    >>> adams = AdamsDocument(
    ...     accession_number="ML12345A678",
    ...     title="Test Document",
    ...     document_type="Inspection Report",
    ...     document_date="2024-01-15",
    ...     docket_numbers=["DOCKET-123", "DOCKET-456"],
    ... )
    >>> case = adams_document_to_case_document(adams)
    >>> case.accession_number
    'ML12345A678'
    >>> isinstance(case.document_date, datetime)
    True
    >>> case.docket_numbers
    ('DOCKET-123', 'DOCKET-456')
    """
    # Parse ISO date string to datetime
    doc_date: datetime | None = None
    if adams_doc.document_date:
        try:
            # Handle both ISO 8601 formats with/without timezone
            date_str = adams_doc.document_date
            # Replace 'Z' with '+00:00' for fromisoformat compatibility
            if date_str.endswith("Z"):
                date_str = date_str[:-1] + "+00:00"
            doc_date = datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            # Keep None if parsing fails - log at call site if needed
            pass

    return CaseDocument(
        accession_number=adams_doc.accession_number,
        title=adams_doc.title,
        document_type=adams_doc.document_type,
        document_date=doc_date,
        content=adams_doc.content,
        author_name=adams_doc.author_name,
        author_affiliation=adams_doc.author_affiliation,
        addressee=adams_doc.addressee,
        addressee_affiliation=adams_doc.addressee_affiliation,
        docket_numbers=tuple(adams_doc.docket_numbers),
        license_numbers=tuple(adams_doc.license_numbers),
        keywords=tuple(adams_doc.keywords),
        url=adams_doc.url,
        estimated_pages=adams_doc.estimated_pages,
        case_metadata=case_metadata or CaseMetadata(),
    )


__all__ = ["adams_document_to_case_document"]
