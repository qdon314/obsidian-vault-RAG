"""Shared helpers for regulatory corpus normalization and publishing."""

from __future__ import annotations

import shutil
from pathlib import Path

from rag.adapters.ingestion.regulatory.ecfr_parser import parse_ecfr_xml
from rag.adapters.ingestion.regulatory.normalizer import NormalizationConfig, normalize_part


def normalize_part_from_xml(
    *,
    xml_source: str | Path,
    corpus_dir: str | Path,
    part: int,
    config: NormalizationConfig,
) -> tuple[list[Path], Path]:
    """Parse XML, filter sections for a part, and write canonical markdown files."""
    source_path = Path(xml_source)
    target_corpus = Path(corpus_dir)
    part_dir = target_corpus / f"part-{part}"

    xml_text = source_path.read_text(encoding="utf-8")
    sections = parse_ecfr_xml(xml_text)
    filtered = [section for section in sections if section.part_number == str(part)]
    written = normalize_part(filtered, config, part_dir)
    return written, part_dir


def upload_directory_to_s3(local_dir: str | Path, *, bucket: str, prefix: str = "") -> int:
    """Upload all files from local_dir recursively to S3."""
    import boto3  # type: ignore[import-untyped]

    directory = Path(local_dir)
    s3 = boto3.client("s3")
    clean_prefix = prefix.strip("/")
    uploaded = 0

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(directory).as_posix()
        key = f"{clean_prefix}/{relative}" if clean_prefix else relative
        s3.upload_file(str(file_path), bucket, key)
        uploaded += 1
    return uploaded


def wipe_directory(path: str | Path) -> bool:
    """Delete a directory if present. Returns True when a deletion occurred."""
    target = Path(path)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True
