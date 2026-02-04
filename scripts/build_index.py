from __future__ import annotations

import argparse
import logging
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv

from rag import settings
from rag.app.container import ContainerOverrides, build_container
from rag.app.pipeline import index_documents
from rag.domain.index_manifest import IndexManifest
from rag.domain.models import Chunk, Document
from rag.ports import Embedder, VectorStore

log = logging.getLogger("index")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build a persisted RAG index from a corpus.")
    ap.add_argument("--corpus", required=True, help="Directory (or file/glob) to ingest.")
    ap.add_argument(
        "--index-name", required=True, help="Name of the index under artifacts/indexes/"
    )
    ap.add_argument(
        "--artifacts-dir", default="artifacts", help="Artifacts root directory (default: artifacts)"
    )
    ap.add_argument(
        "--recursive",
        action="store_true",
        default=True,
        help="Recurse into subdirectories (default: true)",
    )
    ap.add_argument(
        "--no-recursive", dest="recursive", action="store_false", help="Disable recursion"
    )

    # Fixed chunker overrides (only apply when chunking.backend="fixed" in settings)
    ap.add_argument(
        "--chunk-size", type=int, default=None, help="Override chunk size (default: from settings)"
    )
    ap.add_argument(
        "--overlap", type=int, default=None, help="Override chunk overlap (default: from settings)"
    )

    # Obsidian structural chunker overrides (only apply when chunking.backend="obsidian_structural")
    ap.add_argument(
        "--target-chars",
        type=int,
        default=None,
        help="Override target chars per chunk (default: from settings)",
    )
    ap.add_argument(
        "--hard-max-chars",
        type=int,
        default=None,
        help="Override hard max chars per chunk (default: from settings)",
    )
    ap.add_argument(
        "--overlap-blocks",
        type=int,
        default=None,
        help="Override overlap blocks (default: from settings)",
    )
    ap.add_argument(
        "--no-heading-preamble",
        dest="include_heading_preamble",
        action="store_false",
        default=None,
        help="Disable heading preamble in chunks",
    )

    # Embedder override (optional)
    ap.add_argument(
        "--use-dummy-embeddings", action="store_true", help="Use DummyEmbedder instead of OpenAI"
    )
    ap.add_argument(
        "--embed-dim", type=int, default=128, help="Dummy embedder dim (ignored for OpenAI)"
    )

    ap.add_argument(
        "--cache-embeddings",
        action="store_true",
        default=True,
        help="Cache embeddings in SQLite (default: true)",
    )
    ap.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")

    # Ingest selection
    ap.add_argument(
        "--extensions",
        default=None,
        help="Comma-separated allowed extensions (default: from settings)",
    )
    ap.add_argument(
        "--max-docs", type=int, default=0, help="Limit number of docs ingested (0 = no limit)"
    )

    # Performance tuning
    ap.add_argument(
        "--embed-batch-size",
        type=int,
        default=512,
        help="Max chunks per embedding API call (default: 512)",
    )
    ap.add_argument(
        "--no-parallel", action="store_true", help="Disable parallel chunking/embedding pipeline"
    )

    return ap


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m{secs:04.1f}s"


# ---------------------------------------------------------------------------
# Parallel pipeline: overlap T5 chunking (CPU) with embedding API calls (IO)
# ---------------------------------------------------------------------------


def _embed_and_upsert(
    chunks: list[Chunk],
    embedder: Embedder,
    store: VectorStore,
) -> int:
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.upsert(chunks=chunks, vectors=vectors)
    return len(chunks)


def _run_parallel_index(
    docs: list[Document],
    *,
    chunker: object,
    embedder: Embedder,
    store: VectorStore,
    embed_batch_size: int,
) -> int:
    """Index documents with T5 chunking overlapping embedding API calls."""
    chunk_buffer: list[Chunk] = []
    futures: list[Future[int]] = []
    n_docs = len(docs)

    with ThreadPoolExecutor(max_workers=1) as pool:
        for i, doc in enumerate(docs):
            t0 = perf_counter()
            chunks = chunker.chunk(doc)  # type: ignore[union-attr]
            chunk_dt = perf_counter() - t0

            doc_name = Path(doc.uri).name
            log.info(
                "  [%*d/%d] %s — %d chunks (%.1fs chunk)",
                len(str(n_docs)),
                i + 1,
                n_docs,
                doc_name,
                len(chunks),
                chunk_dt,
            )

            chunk_buffer.extend(chunks)

            while len(chunk_buffer) >= embed_batch_size:
                batch = chunk_buffer[:embed_batch_size]
                chunk_buffer = chunk_buffer[embed_batch_size:]
                futures.append(pool.submit(_embed_and_upsert, batch, embedder, store))

        # flush remainder
        if chunk_buffer:
            futures.append(pool.submit(_embed_and_upsert, chunk_buffer, embedder, store))

    # collect results (raises if any future failed)
    total = 0
    for f in futures:
        total += f.result()
    return total


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()

    logging.basicConfig(
        format="[index] %(message)s",
        level=logging.INFO,
    )

    wall_start = perf_counter()

    # Load settings early so we can use defaults from config
    log.info("Loading settings and building container...")
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

    emb_backend = "dummy" if args.use_dummy_embeddings else cfg.embeddings.backend
    cache_status = "on" if args.cache_embeddings else "off"
    log.info(
        "Container ready (chunker=%s, embedder=%s/%s, cache=%s)",
        cfg.chunking.backend,
        emb_backend,
        container.embedder.model_name,
        cache_status,
    )

    # IMPORTANT: In index-build mode, we do NOT call store.load().
    # We want a fresh in-memory store that will be saved at the end.
    # (JsonlVectorStore's internal lists start empty by default.)

    # -----------------------------
    # 2) Ingest (via container.ingestor)
    # -----------------------------
    metadata = {"collection": args.index_name}

    log.info("Ingesting from %s...", vault_root)
    t0 = perf_counter()

    # If you want to override extensions/recursive per run, do it here.
    # (This keeps Settings as defaults, CLI as overrides, without complicating the container.)
    if args.extensions:
        allowed_exts = {e.strip() for e in args.extensions.split(",") if e.strip()}
    else:
        allowed_exts = None

    docs, report = container.ingestor.ingest([str(vault_root)], metadata=metadata)

    if allowed_exts is not None:
        docs = [d for d in docs if str(Path(d.uri).suffix).lower() in allowed_exts]

    if args.max_docs and args.max_docs > 0:
        docs = docs[: args.max_docs]

    ingest_dt = perf_counter() - t0
    log.info(
        "Ingested %d docs (scanned=%d, loaded=%d, skipped_empty=%d) in %s",
        len(docs),
        report.scanned,
        report.loaded,
        report.skipped_empty,
        _format_duration(ingest_dt),
    )

    # -----------------------------
    # 3) Index via container (chunk → embed → upsert)
    # -----------------------------
    log.info("Indexing %d documents...", len(docs))
    t0 = perf_counter()

    if args.no_parallel:
        # Sequential path: uses cross-document batching but no threading
        n_docs = len(docs)
        width = len(str(n_docs))

        doc_counter = 0

        def _on_doc_chunked(doc: Document, n_chunks: int, elapsed: float) -> None:
            nonlocal doc_counter
            doc_counter += 1
            doc_name = Path(doc.uri).name
            log.info(
                "  [%*d/%d] %s — %d chunks (%.1fs chunk)",
                width,
                doc_counter,
                n_docs,
                doc_name,
                n_chunks,
                elapsed,
            )

        def _on_batch_embedded(batch_size: int, elapsed: float) -> None:
            log.info("  Embedded batch of %d chunks (%.1fs)", batch_size, elapsed)

        total_chunks = index_documents(
            docs,
            chunker=container.chunker,
            embedder=container.embedder,
            store=container.store,
            embed_batch_size=args.embed_batch_size,
            on_doc_chunked=_on_doc_chunked,
            on_batch_embedded=_on_batch_embedded,
        )
    else:
        # Parallel path: T5 chunking overlaps with embedding API calls
        total_chunks = _run_parallel_index(
            docs,
            chunker=container.chunker,
            embedder=container.embedder,
            store=container.store,
            embed_batch_size=args.embed_batch_size,
        )

    index_dt = perf_counter() - t0
    log.info(
        "Indexing complete: %s chunks from %d docs in %s",
        f"{total_chunks:,}",
        len(docs),
        _format_duration(index_dt),
    )

    # Persist the built index
    log.info("Saving to %s...", index_dir / "chunks.jsonl")
    t0 = perf_counter()
    container.store.save()
    save_dt = perf_counter() - t0
    log.info("Saved in %s", _format_duration(save_dt))

    # -----------------------------
    # 4) Manifest
    # -----------------------------
    wall_dt = perf_counter() - wall_start

    manifest = IndexManifest.create(
        index_name=args.index_name,
        corpus=str(vault_root),
        doc_count=len(docs),
        chunk_count=total_chunks,
        chunking=container.chunker.get_config(),
        embedding={
            "backend": emb_backend,
            "model": container.embedder.model_name,
            "cached": bool(args.cache_embeddings),
        },
        ingest_report=asdict(report),
        store={
            "type": "jsonl",
            "path": str(index_dir),
            "file": "chunks.jsonl",
        },
        build_duration_s=round(wall_dt, 2),
    )
    manifest.save(index_dir)

    log.info("Done in %s", _format_duration(wall_dt))
    log.info("  docs:     %d", len(docs))
    log.info("  chunks:   %s", f"{total_chunks:,}")
    log.info("  store:    %s", index_dir / "chunks.jsonl")
    log.info("  manifest: %s", index_dir / "manifest.json")


if __name__ == "__main__":
    main()
