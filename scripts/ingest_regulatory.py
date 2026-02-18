#!/usr/bin/env python3
"""Regulatory corpus ingestion pipeline.

1. Parse eCFR XML for a CFR part.
2. Normalize into canonical markdown files.
3. Ingest -> chunk -> enrich -> embed -> store.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from rag import settings
from rag.adapters.ingestion.filesystem import FilesystemIngestor
from rag.adapters.ingestion.loaders.obsidian_markdown_loader import ObsidianMarkdownLoader
from rag.adapters.ingestion.loaders.text_loader import TextLoader
from rag.adapters.ingestion.regulatory.enrichment import enrich_regulatory_chunks
from rag.adapters.ingestion.regulatory.normalizer import NormalizationConfig
from rag.app.container import ContainerOverrides, build_container
from rag.app.regulatory_pipeline import (
    normalize_part_from_xml,
    upload_directory_to_s3,
    wipe_directory,
)
from rag.domain.index_manifest import IndexManifest
from rag.domain.models import Chunk

log = logging.getLogger("ingest-regulatory")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest regulatory corpus from eCFR XML.")
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
    parser.add_argument("--index-name", default="regulatory")
    parser.add_argument("--artifacts-dir", default="artifacts")
    parser.add_argument("--use-dummy-embeddings", action="store_true", help="Use DummyEmbedder")
    parser.add_argument("--embed-dim", type=int, default=128, help="Dummy embedder dim")
    parser.add_argument(
        "--skip-normalize",
        action="store_true",
        help="Skip normalization (use existing markdown)",
    )
    parser.add_argument("--skip-index", action="store_true", help="Only normalize, do not index")
    parser.add_argument(
        "--push-s3-bucket",
        default=None,
        help="Upload normalized files to this S3 bucket after processing",
    )
    parser.add_argument(
        "--push-s3-prefix",
        default="",
        help="S3 key prefix for normalized files upload",
    )
    parser.add_argument(
        "--wipe-local",
        action="store_true",
        help="Delete local normalized files directory after processing",
    )
    return parser


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:04.1f}s"


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()
    logging.basicConfig(format="[regulatory] %(message)s", level=logging.INFO)

    corpus_dir = Path(args.corpus_dir)
    part_dir = corpus_dir / f"part-{args.part}"
    artifacts_dir = Path(args.artifacts_dir).resolve()
    index_dir = artifacts_dir / "indexes" / args.index_name

    config = NormalizationConfig(
        regime=args.regime,
        instrument=args.instrument,
        instrument_version=args.instrument_version,
        source_url=args.source_url,
        source_revision=args.source_revision,
        effective_date=args.effective_date,
    )

    wall_start = perf_counter()

    if not args.skip_normalize:
        log.info("Reading XML from %s...", args.xml_source)
        log.info("Normalizing to %s...", part_dir)
        written, _part_dir = normalize_part_from_xml(
            xml_source=args.xml_source,
            corpus_dir=corpus_dir,
            part=args.part,
            config=config,
        )
        log.info("Wrote %d canonical markdown files", len(written))
    else:
        log.info("Skipping normalization (--skip-normalize)")

    if args.skip_index:
        log.info("Skipping indexing (--skip-index)")
    else:
        log.info("Building index at %s...", index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        cfg = settings.load_settings()
        overrides = ContainerOverrides(
            store_backend="jsonl",
            jsonl_index_dir=index_dir,
            vault_dir=corpus_dir,
            embedder_backend="dummy" if args.use_dummy_embeddings else None,
            dummy_embed_dim=args.embed_dim if args.use_dummy_embeddings else None,
        )
        container = build_container(overrides=overrides)
        emb_backend = "dummy" if args.use_dummy_embeddings else cfg.embeddings.backend

        ingestor = FilesystemIngestor(
            text_loader=TextLoader(),
            markdown_loader=ObsidianMarkdownLoader(vault_dir=corpus_dir),
            skip_hidden=False,
        )
        docs, report = ingestor.ingest([str(corpus_dir)], metadata={"collection": args.index_name})
        log.info("Ingested %d documents", len(docs))

        chunks: list[Chunk] = []
        for doc in docs:
            chunks.extend(container.chunker.chunk(doc))
        log.info("Chunked into %d chunks", len(chunks))

        chunks = enrich_regulatory_chunks(chunks)
        log.info("Enriched chunks with citation metadata")

        vectors = container.embedder.embed_texts([chunk.text for chunk in chunks])
        container.store.upsert(chunks=chunks, vectors=vectors)
        container.store.save()
        log.info("Indexed and saved %d chunks", len(chunks))

        wall_dt = perf_counter() - wall_start
        index_manifest = IndexManifest.create(
            index_name=args.index_name,
            corpus=str(corpus_dir),
            doc_count=len(docs),
            chunk_count=len(chunks),
            chunking=container.chunker.get_config(),
            embedding={"backend": emb_backend, "model": container.embedder.model_name},
            ingest_report=asdict(report),
            store={"type": "jsonl", "path": str(index_dir)},
            build_duration_s=round(wall_dt, 2),
        )
        index_manifest.save(index_dir)

        log.info("Done in %s", _format_duration(wall_dt))
        log.info("  docs:     %d", len(docs))
        log.info("  chunks:   %d", len(chunks))
        log.info("  manifest: %s", index_dir / "manifest.json")

    if args.push_s3_bucket:
        if not part_dir.exists():
            raise FileNotFoundError(
                f"Cannot upload normalized files because directory does not exist: {part_dir}"
            )
        prefix = args.push_s3_prefix.strip("/")
        log.info(
            "Uploading normalized files from %s to s3://%s/%s",
            part_dir,
            args.push_s3_bucket,
            prefix,
        )
        uploaded = upload_directory_to_s3(
            part_dir,
            bucket=args.push_s3_bucket,
            prefix=prefix,
        )
        log.info("Uploaded %d files to s3://%s/%s", uploaded, args.push_s3_bucket, prefix)

    if args.wipe_local:
        if wipe_directory(part_dir):
            log.info("Deleted local normalized files directory: %s", part_dir)
        else:
            log.info("Local normalized files directory already absent: %s", part_dir)


if __name__ == "__main__":
    main()
