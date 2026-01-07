from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv

from rag.adapters.embedding.sqlite_cache import CachedEmbedder
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.app.container import ContainerOverrides, build_container
from rag.app.query_runner import run_query


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Query a built RAG index.")
    ap.add_argument("--index", required=True, help="Index name under artifacts/indexes/")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    ap.add_argument("--q", required=True, help="Query text")

    ap.add_argument("--top-k", type=int, default=None, help="Override retrieval top_k (default: from settings)")
    ap.add_argument("--token-budget", type=int, default=None, help="Override token budget (default: pipeline default)")

    ap.add_argument("--use-dummy-embeddings", action="store_true", help="Use DummyEmbedder instead of OpenAI")
    ap.add_argument("--embed-dim", type=int, default=128, help="Dummy embed dim (only used with --use-dummy-embeddings)")

    ap.add_argument("--cache-embeddings", action="store_true", default=True, help="Cache embeddings in SQLite (default: true)")
    ap.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")

    return ap


def main() -> None:
    args = build_argparser().parse_args()
    load_dotenv()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    index_dir = artifacts_dir / "indexes" / args.index

    # 1) Container wiring (force JSONL store pointing at index_dir)
    overrides = ContainerOverrides(
        store_backend="jsonl",
        jsonl_index_dir=index_dir,
        embedder_backend="dummy" if args.use_dummy_embeddings else None,
        dummy_embed_dim=args.embed_dim if args.use_dummy_embeddings else None,
    )

    # Grab cfg so we can use cfg.retrieval.top_k default if user didn't pass --top-k
    from rag import settings
    cfg = settings.load_settings()

    base = build_container(cfg=cfg, overrides=overrides)

    # 2) Query lifecycle: load persisted index
    base.store.load()

    # 3) Optional embedding cache wrapper (and rebuild retriever to match)
    container = base
    if args.cache_embeddings:
        cache_db = artifacts_dir / "cache" / "embeddings" / f"{container.embedder.model_name}.sqlite3"
        cache_db.parent.mkdir(parents=True, exist_ok=True)

        cached_embedder = CachedEmbedder(embedder=container.embedder, db_path=cache_db)
        retriever = VectorRetriever(embedder=cached_embedder, store=container.store)
        container = replace(container, embedder=cached_embedder, retriever=retriever)

    # 4) Run pipeline (use config default for top_k; pipeline default for token_budget)
    top_k = args.top_k if args.top_k is not None else cfg.retrieval.top_k
    token_budget = args.token_budget if args.token_budget is not None else 1800

    run_query(
        args.q,
        retriever=container.retriever,
        context_builder=container.context_builder,
        generator=container.generator,
        logger=container.logger,
        top_k=top_k,
        token_budget=token_budget,
    )


if __name__ == "__main__":
    main()
