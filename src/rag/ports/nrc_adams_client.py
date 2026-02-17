"""Port for NRC ADAMS Public Search API interactions."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from typing import Protocol

# ------------------------------------
# Data types crossing the port boundary
# ------------------------------------


@dataclass(frozen=True)
class AdamsSearchResult:
    """A single result from an ADAMS search query."""

    accession_number: str  # e.g. "ML12345A678"
    title: str
    document_type: str
    document_date: str | None = None  # ISO date string from API
    author_name: str | None = None
    author_affiliation: str | None = None
    docket_number: str | None = None
    url: str | None = None
    estimated_pages: int | None = None


@dataclass(frozen=True)
class AdamsDocument:
    """Full document fetched by accession number."""

    accession_number: str
    title: str
    document_type: str
    document_date: str | None = None
    author_name: str | None = None
    author_affiliation: str | None = None
    addressee: str | None = None
    addressee_affiliation: str | None = None
    docket_numbers: list[str] = field(default_factory=list)
    license_numbers: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    content: str | None = None  # full text; may be absent for binary docs
    url: str | None = None
    estimated_pages: int | None = None


# ------------------------------------
# Protocol
# ------------------------------------


class NrcAdamsClient(Protocol):
    """Port for NRC ADAMS Public Search API interactions."""

    def search_documents(
        self,
        query: str = "",
        *,
        filters: list[dict[str, object]] | None = None,
        any_filters: list[dict[str, object]] | None = None,
        sort_by: str = "DocumentDate",
        sort_desc: bool = True,
        metadata: Mapping[str, object] | None = None,
    ) -> Iterator[AdamsSearchResult]:
        """Search for documents with automatic pagination."""
        ...

    def get_document(
        self,
        accession_number: str,
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> AdamsDocument | None:
        """Fetch a single document by accession number. Returns None if not found."""
        ...
