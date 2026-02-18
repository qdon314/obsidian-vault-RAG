"""CLI: Run a single query against remote backends.

Designed to run as an ECS task (query-eval with command override).

Usage:
    ./scripts/py scripts/run_remote_query.py --query "What is 10 CFR 50.46?"
"""

from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

from rag import settings
from rag.app.container import build_container
from rag.app.query_runner import run_query

log = logging.getLogger("remote-query")


def main() -> None:
    ap = argparse.ArgumentParser(description="Query remote RAG backends.")
    ap.add_argument(
        "--query",
        default=os.environ.get("QUERY", ""),
        help="Query text (or set QUERY env var).",
    )
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=None)
    args = ap.parse_args()

    if not args.query:
        log.error("No query provided. Use --query or set QUERY env var.")
        raise SystemExit(1)

    load_dotenv()
    logging.basicConfig(format="[remote-query] %(message)s", level=logging.INFO)

    cfg = settings.load_settings()
    container = build_container(cfg=cfg)
    container.store.load()

    top_k = args.top_k or cfg.retrieval.top_k
    token_budget = args.token_budget or cfg.context.token_budget

    result = run_query(
        args.query,
        retriever=container.retriever,
        reranker=container.reranker,
        keep_k=cfg.rerank.keep_k,
        context_builder=container.context_builder,
        generator=container.generator,
        logger=container.logger,
        top_k=top_k,
        token_budget=token_budget,
    )

    print(f"\n{result.answer.text}")


if __name__ == "__main__":
    main()
