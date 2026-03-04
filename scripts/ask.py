from __future__ import annotations

import argparse
import logging
from pathlib import Path

from dotenv import load_dotenv

from rag import settings
from rag.app.container import ContainerOverrides, build_container
from rag.app.manifest_validation import validate_index
from rag.app.query_runner import run_query
from rag.domain.index_manifest import IndexManifest


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Query a built RAG index.")
    ap.add_argument("--index", required=True, help="Index name under artifacts/indexes/")
    ap.add_argument("--artifacts-dir", default="artifacts", help="Artifacts root directory")
    ap.add_argument("--q", required=True, help="Query text")

    ap.add_argument(
        "--top-k", type=int, default=None, help="Override retrieval top_k (default: from settings)"
    )
    ap.add_argument(
        "--token-budget",
        type=int,
        default=None,
        help="Override token budget (default: pipeline default)",
    )

    ap.add_argument(
        "--use-dummy-embeddings", action="store_true", help="Use DummyEmbedder instead of OpenAI"
    )
    ap.add_argument(
        "--embed-dim",
        type=int,
        default=128,
        help="Dummy embed dim (only used with --use-dummy-embeddings)",
    )

    ap.add_argument(
        "--cache-embeddings",
        action="store_true",
        default=True,
        help="Cache embeddings in SQLite (default: true)",
    )
    ap.add_argument("--no-cache-embeddings", dest="cache_embeddings", action="store_false")

    ap.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip manifest compatibility check (use with caution)",
    )

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
        cache_embeddings=args.cache_embeddings,
    )

    # Grab cfg so we can use cfg.retrieval.top_k default if user didn't pass --top-k
    cfg = settings.load_settings()

    container = build_container(cfg=cfg, overrides=overrides)

    # 2) Query lifecycle: load persisted index
    container.store.load()

    # 3) Validate index compatibility
    manifest_path = index_dir / "manifest.json"
    if manifest_path.exists() and not args.skip_validation:
        manifest = IndexManifest.load(index_dir)
        validate_index(manifest, container.embedder)
    elif not manifest_path.exists():
        logging.getLogger(__name__).warning(
            "No manifest.json in %s — skipping compatibility check", index_dir
        )

    # 4) Run pipeline (use config default for top_k; pipeline default for token_budget)
    top_k = args.top_k if args.top_k is not None else cfg.retrieval.top_k
    token_budget = args.token_budget if args.token_budget is not None else 1800

    result = run_query(
        args.q,
        retriever=container.retriever,
        reranker=container.reranker,
        keep_k=cfg.rerank.keep_k,
        context_builder=container.context_builder,
        generator=container.generator,
        logger=container.logger,
        compressor=container.compressor,
        top_k=top_k,
        token_budget=token_budget,
    )

    print(f"\n{result.answer.text}")


if __name__ == "__main__":
    main()
