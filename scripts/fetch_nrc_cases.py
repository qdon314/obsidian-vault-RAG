#!/usr/bin/env python3
"""Fetch and normalise NRC ADAMS case documents to the local corpus.

Operates in two modes:

* **Discovery mode** (default) — search ADAMS by document type and
  optional date range, then fetch each result.
* **Direct mode** (``--accession``) — fetch specific accession numbers.

Normalised markdown files are written into year-month subdirectories
under the output directory (e.g. ``corpus/us-nrc/cases/2024-01/ML24001A123.md``).

Usage examples::

    ./scripts/py scripts/fetch_nrc_cases.py --date-from 2024-01-01 --date-to 2024-03-31
    ./scripts/py scripts/fetch_nrc_cases.py --accession ML23318A200 --accession ML24002B456
    ./scripts/py scripts/fetch_nrc_cases.py --document-types "Inspection Report" --limit 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

from rag import settings
from rag.adapters.ingestion.case.fetcher import CaseDocumentFetcher, FetchConfig
from rag.adapters.ingestion.case.http_client import HttpNrcAdamsClient, RetryConfig
from rag.adapters.ingestion.case.normalizer import CaseNormalizationConfig
from rag.domain.case_documents import FetchReport

log = logging.getLogger("fetch-nrc-cases")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch NRC ADAMS case documents and write normalised markdown to corpus.",
    )
    parser.add_argument(
        "--date-from",
        default=None,
        metavar="YYYY-MM-DD",
        help="Fetch documents on or after this date (discovery mode only).",
    )
    parser.add_argument(
        "--date-to",
        default=None,
        metavar="YYYY-MM-DD",
        help="Fetch documents on or before this date (discovery mode only).",
    )
    parser.add_argument(
        "--document-types",
        action="append",
        dest="document_types",
        default=None,
        metavar="TYPE",
        help=(
            "Document type filter (repeat for multiple). "
            "Defaults to [case_ingestion].document_types from settings.toml."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Output directory for markdown files. "
            "Defaults to [case_ingestion].output_dir from settings.toml."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of documents to fetch in discovery mode.",
    )
    parser.add_argument(
        "--accession",
        action="append",
        default=None,
        help="Explicit accession number (repeat for multiple; activates direct mode).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing markdown files (default: skip).",
    )
    parser.add_argument(
        "--settings",
        default="settings.toml",
        help="Path to settings TOML file.",
    )
    return parser


def _parse_date(value: str | None, flag: str) -> date | None:
    if value is None:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError as err:
        raise SystemExit(f"{flag} must be YYYY-MM-DD, got: {value!r}") from err


def _print_report(report: FetchReport, output_dir: Path) -> None:
    log.info("Fetch complete.")
    log.info("  output_dir:      %s", output_dir)
    log.info("  total_requested: %d", report.total_requested)
    log.info("  fetched:         %d", report.fetched)
    log.info("  skipped:         %d", report.skipped)
    log.info("  failed:          %d", report.failed)
    if report.by_document_type:
        for doc_type, count in sorted(report.by_document_type.items()):
            log.info("    [%s]: %d", doc_type, count)
    if report.errors:
        log.warning("  Errors (%d):", len(report.errors))
        for err in report.errors:
            log.warning("    %s", err)


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()
    logging.basicConfig(
        format="[fetch-nrc-cases] %(levelname)s %(message)s",
        level=logging.INFO,
    )

    cfg = settings.load_settings(path=args.settings, require_openai=False)

    if not cfg.secrets.nrc_adams_api_key:
        raise SystemExit(
            "NRC_ADAMS_API_KEY is required. Set it in your environment or .env file."
        )

    date_from = _parse_date(args.date_from, "--date-from")
    date_to = _parse_date(args.date_to, "--date-to")

    document_types = tuple(args.document_types or cfg.case_ingestion.document_types)
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else cfg.case_ingestion.output_dir
    )

    client = HttpNrcAdamsClient(
        subscription_key=cfg.secrets.nrc_adams_api_key,
        base_url=cfg.nrc_adams.base_url,
        timeout=cfg.nrc_adams.timeout,
        page_size=cfg.nrc_adams.page_size,
        retry=RetryConfig(max_retries=cfg.nrc_adams.max_retries),
    )

    fetch_config = FetchConfig(
        output_dir=output_dir,
        norm_config=CaseNormalizationConfig(
            regime="us-nrc",
            corpus_label="nrc-cases",
            source_url=cfg.nrc_adams.base_url,
        ),
        document_types=document_types,
        date_from=date_from,
        date_to=date_to,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    fetcher = CaseDocumentFetcher(client=client, config=fetch_config)

    mode = "direct" if args.accession else "discovery"
    log.info("Starting fetch run (mode=%s, output_dir=%s)", mode, output_dir)

    accessions = (
        [a.strip() for a in args.accession if a.strip()] if args.accession else None
    )

    report = fetcher.fetch(accessions=accessions)
    _print_report(report, output_dir)

    if report.failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
