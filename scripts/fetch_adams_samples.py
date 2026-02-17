#!/usr/bin/env python3
"""Fetch a small set of ADAMS documents and save them as local reference JSON files.

Usage examples:
    ./scripts/py scripts/fetch_adams_samples.py
    ./scripts/py scripts/fetch_adams_samples.py --limit 10
    ./scripts/py scripts/fetch_adams_samples.py --accession ML25001A123 --accession ML25002B456
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from rag import settings
from rag.adapters.ingestion.case.http_client import HttpNrcAdamsClient, RetryConfig
from rag.ports.nrc_adams_client import AdamsDocument

log = logging.getLogger("fetch-adams-samples")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fetch sample NRC ADAMS documents into local JSON files."
    )
    parser.add_argument(
        "--output-dir",
        default="data/adams_samples",
        help="Directory where sample JSON files are written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=6,
        help="Maximum number of documents to fetch (ignored when --accession is used).",
    )
    parser.add_argument(
        "--per-type",
        type=int,
        default=2,
        help="Maximum docs to sample per document type during discovery.",
    )
    parser.add_argument(
        "--document-type",
        action="append",
        dest="document_types",
        default=None,
        help=(
            "Document type to sample (repeat flag for multiple). "
            "Defaults to [case_ingestion].document_types from settings.toml."
        ),
    )
    parser.add_argument(
        "--accession",
        action="append",
        default=None,
        help="Explicit accession number to fetch (repeat flag for multiple).",
    )
    parser.add_argument(
        "--settings",
        default="settings.toml",
        help="Path to settings TOML file.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing JSON files in output directory.",
    )
    return parser


def _discover_accessions(
    client: HttpNrcAdamsClient,
    *,
    document_types: tuple[str, ...],
    per_type: int,
    limit: int,
) -> list[str]:
    accessions: list[str] = []
    seen: set[str] = set()

    for document_type in document_types:
        if len(accessions) >= limit:
            break

        found_for_type = 0
        for result in client.search_documents(
            "",
            filters=[{"field": "DocumentType", "value": document_type}],
            sort_by="DocumentDate",
            sort_desc=True,
        ):
            accession = result.accession_number.strip()
            if not accession or accession in seen:
                continue
            seen.add(accession)
            accessions.append(accession)
            found_for_type += 1
            if found_for_type >= per_type or len(accessions) >= limit:
                break

    return accessions[:limit]


def _write_document(path: Path, doc: AdamsDocument) -> None:
    payload = {
        "fetched_at": datetime.now(UTC).isoformat(),
        "document": asdict(doc),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()
    logging.basicConfig(format="[adams-samples] %(message)s", level=logging.INFO)

    cfg = settings.load_settings(path=args.settings, require_openai=False)

    if not cfg.secrets.nrc_adams_api_key:
        raise SystemExit(
            "NRC_ADAMS_API_KEY is required. Set it in your environment or .env file first."
        )

    if args.limit <= 0:
        raise SystemExit("--limit must be > 0")
    if args.per_type <= 0:
        raise SystemExit("--per-type must be > 0")

    client = HttpNrcAdamsClient(
        subscription_key=cfg.secrets.nrc_adams_api_key,
        base_url=cfg.nrc_adams.base_url,
        timeout=cfg.nrc_adams.timeout,
        page_size=cfg.nrc_adams.page_size,
        retry=RetryConfig(max_retries=cfg.nrc_adams.max_retries),
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.accession:
        accessions = [a.strip() for a in args.accession if a and a.strip()]
    else:
        document_types = tuple(args.document_types or cfg.case_ingestion.document_types)
        log.info(
            "Discovering up to %d samples across document types: %s",
            args.limit,
            ", ".join(document_types),
        )
        accessions = _discover_accessions(
            client,
            document_types=document_types,
            per_type=args.per_type,
            limit=args.limit,
        )

    if not accessions:
        raise SystemExit("No accession numbers found to fetch.")

    written: list[str] = []
    skipped: list[str] = []
    missing: list[str] = []

    for accession in accessions:
        path = output_dir / f"{accession}.json"
        if path.exists() and not args.overwrite:
            skipped.append(accession)
            log.info("Skipping %s (already exists)", accession)
            continue

        log.info("Fetching %s", accession)
        doc = client.get_document(accession)
        if doc is None:
            missing.append(accession)
            log.warning("Not found: %s", accession)
            continue

        _write_document(path, doc)
        written.append(accession)

    index_payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "output_dir": str(output_dir),
        "requested_accessions": accessions,
        "written_accessions": written,
        "skipped_accessions": skipped,
        "missing_accessions": missing,
    }
    (output_dir / "index.json").write_text(
        json.dumps(index_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    log.info("Done.")
    log.info("  written: %d", len(written))
    log.info("  skipped: %d", len(skipped))
    log.info("  missing: %d", len(missing))
    log.info("  index:   %s", output_dir / "index.json")


if __name__ == "__main__":
    main()
