"""Case document fetcher: search ADAMS, fetch, normalize, and write to disk.

Orchestrates the full fetch pipeline for NRC case documents:

1. **Discovery mode** — search ADAMS by document type / date range,
   collect accession numbers, then fetch each.
2. **Direct mode** — caller supplies explicit accession numbers.

Documents are converted to domain ``CaseDocument`` objects, normalised
to Obsidian markdown, and written into year-month subdirectories under
the configured output directory.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from rag.adapters.ingestion.case.document_adapter import adams_document_to_case_document
from rag.adapters.ingestion.case.normalizer import (
    CaseNormalizationConfig,
    normalize_case_document_to_markdown,
)
from rag.domain.case_documents import CaseDocument, FetchReport
from rag.domain.errors import AdamsApiError, AdamsAuthError
from rag.ports.nrc_adams_client import NrcAdamsClient

logger = logging.getLogger(__name__)

_LOG_INTERVAL = 10


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FetchConfig:
    """Immutable configuration for a case-document fetch run."""

    output_dir: Path
    norm_config: CaseNormalizationConfig
    document_types: tuple[str, ...] = ()
    date_from: date | None = None
    date_to: date | None = None
    limit: int | None = None
    overwrite: bool = False
    log_interval: int = _LOG_INTERVAL


# ------------------------------------------------------------------
# Pure helpers
# ------------------------------------------------------------------


def _year_month_subdir(doc: CaseDocument) -> str:
    """Return ``"YYYY-MM"`` from *doc.document_date*, or ``"undated"``."""
    if doc.document_date is not None:
        return doc.document_date.strftime("%Y-%m")
    return "undated"


def _output_path(output_dir: Path, doc: CaseDocument) -> Path:
    """Resolve the full target path for a normalised markdown file."""
    return output_dir / _year_month_subdir(doc) / f"{doc.accession_number}.md"


def _build_discovery_filters(
    document_type: str,
    date_from: date | None,
    date_to: date | None,
) -> list[dict[str, object]]:
    """Build the ADAMS ``filters`` list for a single document-type search.

    ADAMS uses a special syntax for date comparisons where the value contains
    the operator expression, e.g. "(DocumentDate ge '2024-01-01')".
    """
    filters: list[dict[str, object]] = [
        {"field": "DocumentType", "operator": "contains", "value": document_type},
    ]

    # ADAMS date filters use value-based operator syntax with parentheses
    # When both dates are specified, combine into a single ANDed expression
    if date_from is not None and date_to is not None:
        filters.append(
            {
                "field": "DocumentDate",
                "value": f"(DocumentDate ge '{date_from.isoformat()}') and (DocumentDate le '{date_to.isoformat()}')",
            }
        )
    elif date_from is not None:
        filters.append(
            {"field": "DocumentDate", "value": f"(DocumentDate ge '{date_from.isoformat()}')"}
        )
    elif date_to is not None:
        filters.append(
            {"field": "DocumentDate", "value": f"(DocumentDate le '{date_to.isoformat()}')"}
        )

    return filters


# ------------------------------------------------------------------
# Mutable accumulator (private)
# ------------------------------------------------------------------


@dataclass
class _FetchCounter:
    """Mutable state bag used within a single ``fetch()`` call."""

    total_requested: int = 0
    fetched: int = 0
    skipped: int = 0
    failed: int = 0
    by_document_type: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def to_report(self) -> FetchReport:
        return FetchReport(
            total_requested=self.total_requested,
            fetched=self.fetched,
            skipped=self.skipped,
            failed=self.failed,
            by_document_type=dict(self.by_document_type),
            errors=tuple(self.errors),
        )


# ------------------------------------------------------------------
# Adapter
# ------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CaseDocumentFetcher:
    """Fetch, normalise, and write NRC ADAMS case documents.

    Depends on the :class:`~rag.ports.nrc_adams_client.NrcAdamsClient`
    *protocol*, not the concrete HTTP adapter.
    """

    client: NrcAdamsClient
    config: FetchConfig

    def fetch(
        self,
        *,
        accessions: Sequence[str] | None = None,
    ) -> FetchReport:
        """Run a fetch-and-write pass.

        Parameters
        ----------
        accessions:
            When provided, fetches exactly these accession numbers
            (direct mode).  When ``None``, searches ADAMS using the
            document types and date range in *config* (discovery mode).

        Returns
        -------
        :class:`~rag.domain.case_documents.FetchReport` summarising
        the run.  On ``KeyboardInterrupt`` a *partial* report is
        returned covering work completed before the interrupt.
        """
        counter = _FetchCounter()
        try:
            if accessions is not None:
                self._fetch_accessions(list(accessions), counter=counter)
            else:
                self._discover_and_fetch(counter=counter)
        except KeyboardInterrupt:
            logger.warning(
                "Interrupted after processing %d documents (fetched=%d skipped=%d failed=%d).",
                counter.fetched + counter.skipped + counter.failed,
                counter.fetched,
                counter.skipped,
                counter.failed,
            )
        return counter.to_report()

    # -- discovery mode --------------------------------------------------

    def _discover_and_fetch(self, *, counter: _FetchCounter) -> None:
        seen: set[str] = set()
        for doc_type in self.config.document_types:
            if self.config.limit is not None and counter.fetched >= self.config.limit:
                break
            filters = _build_discovery_filters(doc_type, self.config.date_from, self.config.date_to)
            for result in self.client.search_documents(
                "",
                filters=filters,
                sort_by="DocumentDate",
                sort_desc=True,
            ):
                accession = result.accession_number.strip()
                if not accession or accession in seen:
                    continue
                seen.add(accession)
                counter.total_requested += 1
                self._fetch_one(accession, doc_type=doc_type, counter=counter)
                if self.config.limit is not None and counter.fetched >= self.config.limit:
                    break

    # -- direct mode -----------------------------------------------------

    def _fetch_accessions(
        self,
        accessions: list[str],
        *,
        counter: _FetchCounter,
    ) -> None:
        counter.total_requested = len(accessions)
        for accession in accessions:
            self._fetch_one(accession, doc_type=None, counter=counter)

    # -- single-document processing --------------------------------------

    def _fetch_one(
        self,
        accession: str,
        *,
        doc_type: str | None,
        counter: _FetchCounter,
    ) -> None:
        total_processed = counter.fetched + counter.skipped + counter.failed
        if total_processed > 0 and total_processed % self.config.log_interval == 0:
            logger.info(
                "Progress: fetched=%d skipped=%d failed=%d",
                counter.fetched,
                counter.skipped,
                counter.failed,
            )

        try:
            adams_doc = self.client.get_document(accession)
            if adams_doc is None:
                counter.failed += 1
                counter.errors.append(f"{accession}: not found")
                logger.warning("Not found: %s", accession)
                return

            case_doc = adams_document_to_case_document(adams_doc)
            path = _output_path(self.config.output_dir, case_doc)

            if path.exists() and not self.config.overwrite:
                counter.skipped += 1
                logger.debug("Skipping %s (already exists)", accession)
                return

            markdown = normalize_case_document_to_markdown(case_doc, self.config.norm_config)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(markdown, encoding="utf-8")

            counter.fetched += 1
            effective_type = doc_type or adams_doc.document_type or "unknown"
            counter.by_document_type[effective_type] = (
                counter.by_document_type.get(effective_type, 0) + 1
            )
            logger.debug("Wrote %s -> %s", accession, path)

        except AdamsAuthError:
            # Auth failures affect every subsequent request — abort.
            raise
        except AdamsApiError as exc:
            counter.failed += 1
            counter.errors.append(f"{accession}: {exc}")
            logger.warning("Failed %s: %s", accession, exc)
        except Exception as exc:
            counter.failed += 1
            counter.errors.append(f"{accession}: unexpected error: {exc}")
            logger.exception("Unexpected error fetching %s", accession)


__all__ = ["CaseDocumentFetcher", "FetchConfig"]
