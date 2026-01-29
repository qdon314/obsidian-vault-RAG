from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from rag import settings
from rag.app.container import ContainerOverrides, build_container
from rag.app.pipeline import index_document


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a persisted RAG index from a corpus.")
    ap.add_argument("--corpus", required=True, help="Directory (or file/glob) to ingest.")
    ap.add_argument("--index-name", required=True, help="Name of the index under artifacts/indexes/")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory (default: artifacts)")
    ap.add_argument("--recursive", action="store_true", default=True, help="Recurse into subdirectories (default: true)")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursion")

    # Fixed chunker overrides (only apply when chunking.backend="fixed" in settings)
    ap.add_argument("--chunk-size", type=int, default=None, help="Override chunk size (default: from settings)")
    ap.add_argument("--overlap", type=int, default=None, help="Override chunk overlap (default: from settings)")

    # Obsidian structural chunker overrides (only apply when chunking.backend="obsidian_structural")
    ap.add_argument("--target-chars", type=int, default=None, help="Override target chars per chunk (default: from settings)")
    ap.add_argument("--hard-max-chars", type=int, default=None, help="Override hard max chars per chunk (default: from settings)")
    ap.add_argument("--overlap-blocks", type=int, default=None, help="Override overlap blocks (default: from settings)")
    ap.add_argument("--no-heading-preamble", dest="include_heading_preamble", action="store_false", default=None, help="Disable heading preamble in chunks")

    # Embedder override (optional)
    ap.add_argument("--use-dummy-embeddings", action="store_true", help="Use DummyEmbedder instead of OpenAI")
    ap.add_argument("--embed-dim", type=int, default=128, help="Dummy embedder dim (ignored for OpenAI)")

    ap.add_argument("--cache-embeddings", action="store_true", default=True, help="Cache embeddings in SQLite (default: true)")
    ap.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")

    # Ingest selection
    ap.add_argument("--extensions", default=None, help="Comma-separated allowed extensions (default: from settings)")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit number of docs ingested (0 = no limit)")

    return ap


def _utcnow_iso() -> str:
    return datetime.now(UTC).isoformat()


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()

    # Load settings early so we can use defaults from config
    cfg = settings.load_settings()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    index_dir = artifacts_dir / "indexes" / args.index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    vault_root = Path(args.corpus).expanduser().resolve()

    # -----------------------------
    # 1) Build container (wiring)
    #    - force store_backend="jsonl" for index builds
    #    - pass index_dir to JsonlVectorStore
    #    - override chunking/embedder as requested
    # -----------------------------
    overrides = ContainerOverrides(
        store_backend="jsonl",
        jsonl_index_dir=index_dir,
        # Fixed chunker overrides
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        # Obsidian structural chunker overrides
        target_chars=args.target_chars,
        hard_max_chars=args.hard_max_chars,
        overlap_blocks=args.overlap_blocks,
        include_heading_preamble=args.include_heading_preamble,
        vault_dir=vault_root,  # indexing corpus defines vault root for ingestion
        embedder_backend="dummy" if args.use_dummy_embeddings else None,
        dummy_embed_dim=args.embed_dim if args.use_dummy_embeddings else None,
        cache_embeddings=args.cache_embeddings,
    )
    container = build_container(overrides=overrides)

    # IMPORTANT: In index-build mode, we do NOT call store.load().
    # We want a fresh in-memory store that will be saved at the end.
    # (JsonlVectorStore's internal lists start empty by default.)

    # -----------------------------
    # 2) Ingest (via container.ingestor)
    # -----------------------------
    metadata = {"collection": args.index_name}

    # If you want to override extensions/recursive per run, do it here.
    # (This keeps Settings as defaults, CLI as overrides, without complicating the container.)
    if args.extensions:
        allowed_exts = {e.strip() for e in args.extensions.split(",") if e.strip()}
        # If your FilesystemIngestor supports overriding allowed_extensions/recursive at call time,
        # prefer that. If not, we just filter after ingest.
    else:
        allowed_exts = None

    docs, report = container.ingestor.ingest([str(vault_root)], metadata=metadata)

    if allowed_exts is not None:
        docs = [d for d in docs if str(Path(d.uri).suffix).lower() in allowed_exts]  # assumes Document.uri exists

    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]

    # -----------------------------
    # 3) Index via container (chunk → embed → upsert)
    # -----------------------------
    total_chunks = 0
    for doc in docs:
        total_chunks += index_document(
            doc,
            chunker=container.chunker,
            embedder=container.embedder,
            store=container.store,
        )

    # Persist the built index
    container.store.save()

    # -----------------------------
    # 4) Manifest
    # -----------------------------
    manifest = {
        "index_name": args.index_name,
        "created_at": _utcnow_iso(),
        "corpus": str(vault_root),
        "recursive": args.recursive,
        "doc_count": len(docs),
        "chunk_count": total_chunks,
        "chunking": container.chunker.get_config(),
        "embedding": {
            "backend": cfg.embeddings.backend if not args.use_dummy_embeddings else "dummy",
            "model": container.embedder.model_name,
            "cached": bool(args.cache_embeddings),
        },
        "ingest_report": asdict(report),
        "store": {
            "type": "jsonl",
            "path": str(index_dir),
            "file": "chunks.jsonl",
        },
    }

    (index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Index built: {args.index_name}")
    print(f"  docs:     {len(docs)}")
    print(f"  chunks:   {total_chunks}")
    print(f"  store:    {index_dir / 'chunks.jsonl'}")
    print(f"  manifest: {index_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
