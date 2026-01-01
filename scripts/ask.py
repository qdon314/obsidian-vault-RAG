from __future__ import annotations

import argparse
from pathlib import Path
from os import getenv

from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
# from rag.adapters.embedding.openai_embedder import OpenAIEmbedder
from rag.adapters.embedding.openai_embedder import OpenAIEmbedder
from rag.adapters.embedding.sqlite_cache import CachedEmbedder
from rag.adapters.generation.openai_chat import OpenAIChatGenerator
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.adapters.vectorstores.jsonl_store import JsonlVectorStore
from rag.app.pipeline import rag_answer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Index name under artifacts/indexes/")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--q", required=True, help="Query text")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--token-budget", type=int, default=1800)
    ap.add_argument("--use-openai-embeddings", action="store_true", help="Use OpenAI embeddings (requires OPENAI_API_KEY)")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).resolve()
    index_dir = artifacts_dir / "indexes" / args.index

    store = JsonlVectorStore(path=index_dir)
    store.load()

    if args.use_openai_embeddings:
        api_key = getenv("OPENAI_API_KEY", "")
        embedder = OpenAIEmbedder(api_key=api_key, model="text-embedding-3-small")
    else:
        embedder = DummyEmbedder(dim=args.embed_dim)
        
    cache_db = artifacts_dir / "cache" / "embeddings" / f"{embedder.model_name}.sqlite3"
    embedder = CachedEmbedder(embedder=embedder, db_path=cache_db)

    retriever = VectorRetriever(embedder=embedder, store=store)
    context_builder = SimpleContextBuilder(max_chunks=10, dedupe=True)

    generator = OpenAIChatGenerator(api_key=getenv("OPENAI_API_KEY", ""), model=getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))

    ans = rag_answer(
        args.q,
        retriever=retriever,
        context_builder=context_builder,
        generator=generator,
        top_k=args.top_k,
        token_budget=args.token_budget,
    )

    print("\n=== ANSWER ===\n")
    print(ans.text)
    print("\n=== CITATIONS ===")
    for i, c in enumerate(ans.citations, start=1):
        print(f"[{i}] {c.uri} chunk={c.chunk_id}")


if __name__ == "__main__":
    main()
