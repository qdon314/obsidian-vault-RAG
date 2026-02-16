#!/usr/bin/env python3
"""Normalize eCFR XML into canonical regulatory markdown files."""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from rag.adapters.ingestion.regulatory.normalizer import NormalizationConfig
from rag.app.regulatory_pipeline import normalize_part_from_xml

log = logging.getLogger("regulatory-normalize")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Normalize eCFR XML to canonical markdown.")
    parser.add_argument("--xml-source", required=True, help="Path to eCFR XML file")
    parser.add_argument(
        "--corpus-dir",
        default="corpus/us-nrc/10-CFR",
        help="Output dir for canonical markdown",
    )
    parser.add_argument("--part", type=int, required=True, help="CFR part number (e.g. 50)")
    parser.add_argument("--regime", default="US-NRC")
    parser.add_argument("--instrument", default="10-CFR")
    parser.add_argument("--instrument-version", required=True, help="e.g. 2025-01-01")
    parser.add_argument("--source-url", default="https://www.ecfr.gov/current/title-10")
    parser.add_argument("--source-revision", required=True, help="e.g. ecfr-2025-01-01")
    parser.add_argument("--effective-date", required=True, help="e.g. 2025-01-01")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()
    logging.basicConfig(format="[regulatory-normalize] %(message)s", level=logging.INFO)

    config = NormalizationConfig(
        regime=args.regime,
        instrument=args.instrument,
        instrument_version=args.instrument_version,
        source_url=args.source_url,
        source_revision=args.source_revision,
        effective_date=args.effective_date,
    )

    log.info("Normalizing part %d from %s...", args.part, args.xml_source)
    written, part_dir = normalize_part_from_xml(
        xml_source=args.xml_source,
        corpus_dir=args.corpus_dir,
        part=args.part,
        config=config,
    )
    log.info("Wrote %d files to %s", len(written), part_dir)


if __name__ == "__main__":
    main()
