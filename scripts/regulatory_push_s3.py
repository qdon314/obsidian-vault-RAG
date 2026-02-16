#!/usr/bin/env python3
"""Push normalized regulatory files to S3."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from rag.app.regulatory_pipeline import upload_directory_to_s3, wipe_directory

log = logging.getLogger("regulatory-push-s3")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Push normalized regulatory files to S3.")
    parser.add_argument("--bucket", required=True, help="Target S3 bucket")
    parser.add_argument("--prefix", default="", help="S3 key prefix")
    parser.add_argument(
        "--corpus-dir",
        default="corpus/us-nrc/10-CFR",
        help="Corpus directory containing normalized markdown files",
    )
    parser.add_argument("--part", type=int, default=50, help="CFR part number (default: 50)")
    parser.add_argument(
        "--local-dir",
        default=None,
        help="Explicit local directory to push (overrides --corpus-dir/--part)",
    )
    parser.add_argument(
        "--wipe-local",
        action="store_true",
        help="Delete local directory after successful upload",
    )
    return parser


def _resolve_local_dir(*, local_dir: str | None, corpus_dir: str, part: int) -> Path:
    if local_dir:
        return Path(local_dir)
    return Path(corpus_dir) / f"part-{part}"


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()
    logging.basicConfig(format="[regulatory-push-s3] %(message)s", level=logging.INFO)

    local_dir = _resolve_local_dir(local_dir=args.local_dir, corpus_dir=args.corpus_dir, part=args.part)
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

    prefix = args.prefix.strip("/")
    log.info("Uploading %s to s3://%s/%s", local_dir, args.bucket, prefix)
    uploaded = upload_directory_to_s3(local_dir, bucket=args.bucket, prefix=prefix)
    log.info("Uploaded %d files to s3://%s/%s", uploaded, args.bucket, prefix)

    if args.wipe_local:
        deleted = wipe_directory(local_dir)
        if deleted:
            log.info("Deleted local directory: %s", local_dir)
        else:
            log.info("Local directory already absent: %s", local_dir)


if __name__ == "__main__":
    main()
