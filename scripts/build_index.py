from __future__ import annotations
from os import getenv

import argparse
import json

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from rag.adapters.chunking.fixed import FixedChunker
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
# from rag.adapters.embedding.openai_embedder import OpenAIEmbedder

from rag.adapters.embedding.openai_embedder import OpenAIEmbedder
from rag.adapters.embedding.sqlite_cache import CachedEmbedder
from rag.adapters.ingestion.filesystem import FilesystemIngestor
from rag.adapters.ingestion.loaders.obsidian_markdown_loader import ObsidianMarkdownLoader
from rag.adapters.vectorstores.jsonl_store import JsonlVectorStore
from rag.app.pipeline import index_document


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a persisted RAG index from a corpus.")
    ap.add_argument("--corpus", required=True, help="Directory (or file/glob) to ingest.")
    ap.add_argument("--index-name", required=True, help="Name of the index under artifacts/indexes/")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory (default: artifacts)")
    ap.add_argument("--recursive", action="store_true", default=True, help="Recurse into subdirectories (default: true)")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false", help="Disable recursion")

    ap.add_argument("--chunk-size", type=int, default=1200)
    ap.add_argument("--overlap", type=int, default=150)

    ap.add_argument("--use-openai-embeddings", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY)")
    ap.add_argument("--embed-dim", type=int, default=128, help="Dummy embedder dim (ignored for OpenAI)")

    ap.add_argument("--cache-embeddings", action="store_true", default=True, help="Cache embeddings in SQLite (default: true)")
    ap.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")

    ap.add_argument("--extensions", default=".md", help="Comma-separated allowed extensions (default: .md)")
    ap.add_argument("--max-docs", type=int, default=0, help="Limit number of docs ingested (0 = no limit)")
    return ap


def main() -> None:
    args = build_argparser().parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    index_dir = artifacts_dir / "indexes" / args.index_name
    index_dir.mkdir(parents=True, exist_ok=True)

    allowed_exts = {e.strip() for e in args.extensions.split(",") if e.strip()}
    vault_root = Path(args.corpus).expanduser().resolve()

    # Ingestor: Obsidian-aware markdown loader (strips wikilinks outside code, expands embeds)
    md_loader = ObsidianMarkdownLoader(vault_root=vault_root, expand_embeds=True, max_embed_depth=4)
    ingestor = FilesystemIngestor(
        allowed_extensions=allowed_exts,
        recursive=args.recursive,
        markdown_loader=md_loader if ".md" in allowed_exts else None,
    )

    docs, report = ingestor.ingest([str(vault_root)], metadata={"collection": args.index_name})
    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]

    # Chunker
    chunker = FixedChunker(chunk_size=args.chunk_size, overlap=args.overlap)

    # Embedder
    if args.use_openai_embeddings:
        api_key = getenv("OPENAI_API_KEY", "")
        embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
    else:
        embedder = DummyEmbedder(dim=args.embed_dim)

    # Optional embedding cache
    if args.cache_embeddings:
        cache_db = artifacts_dir / "cache" / "embeddings" / f"{embedder.model_name}.sqlite3"
        embedder = CachedEmbedder(embedder=embedder, db_path=cache_db)

    # Store (persisted)
    store = JsonlVectorStore(path=index_dir)
    # Start fresh each time for reproducibility:
    store._chunks.clear()   # intentional: rebuild index from scratch
    store._vectors.clear()

    # Index everything
    total_chunks = 0
    for doc in docs:
        total_chunks += index_document(doc, chunker=chunker, embedder=embedder, store=store)

    store.save()

    manifest = {
        "index_name": args.index_name,
        "created_at": datetime.utcnow().isoformat(),
        "corpus": str(vault_root),
        "allowed_extensions": sorted(allowed_exts),
        "recursive": args.recursive,
        "doc_count": len(docs),
        "chunk_count": total_chunks,
        "chunking": {
            "strategy": getattr(chunker, "strategy_name", "unknown"),
            "chunk_size": args.chunk_size,
            "overlap": args.overlap,
        },
        "embedding": {
            "model": embedder.model_name,
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
    print(f"  docs:   {len(docs)}")
    print(f"  chunks: {total_chunks}")
    print(f"  store:  {index_dir / 'chunks.jsonl'}")
    print(f"  manifest: {index_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
