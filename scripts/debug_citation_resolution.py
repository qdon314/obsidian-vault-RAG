"""Diagnostic: check citation resolution between eval queries and chunk store.

Usage: ./scripts/py scripts/debug_citation_resolution.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from rag.domain.citations import normalize_citation_key


def main() -> None:
    # --- Load eval queries ---
    queries_path = Path("eval/datasets/case_generated_queries.jsonl")
    queries = []
    with queries_path.open() as f:
        for line in f:
            if line.strip():
                queries.append(json.loads(line))

    # Collect all citations from queries
    query_citations: set[str] = set()
    for q in queries:
        for c in q.get("relevant_citations", []):
            query_citations.add(c)
        for c in q.get("critical_citations", []):
            query_citations.add(c)
        for c in q.get("supporting_citations", []):
            query_citations.add(c)
        for c in q.get("relevant_doc_citations", []):
            query_citations.add(c)
        for c in q.get("context_doc_citations", []):
            query_citations.add(c)

    print(f"=== Eval queries: {len(queries)} ===")
    print(f"=== Unique citations in queries: {len(query_citations)} ===")
    print()

    # Show raw and normalized forms
    print("--- Query citations (raw -> normalized) ---")
    for raw in sorted(query_citations):
        norm = normalize_citation_key(raw)
        marker = " *CHANGED*" if raw != norm else ""
        print(f"  {raw!r:40s} -> {norm!r}{marker}")
    print()

    # --- Load chunk store and build indexes ---
    from rag.app.container import build_container
    from rag.settings import load_settings

    settings = load_settings(require_openai=False)
    container = build_container(cfg=settings)
    container.store.load()

    try:
        chunks = container.store.all_chunks()
    except NotImplementedError:
        chunk_store = container.chunk_store
        if chunk_store is not None:
            all_ids = chunk_store.list_all_chunk_ids()
            chunks = list(chunk_store.get_chunks(all_ids).values())
        else:
            print("ERROR: No chunk store available")
            return

    print(f"=== Total chunks loaded: {len(chunks)} ===")

    # Count by corpus
    corpus_counts: dict[str, int] = defaultdict(int)
    for ch in chunks:
        corpus_counts[str(ch.metadata.get("corpus", "unknown"))] += 1
    print(f"=== Chunks by corpus: {dict(corpus_counts)} ===")
    print()

    # Build indexes (same as harness)
    citation_to_ids: dict[str, set[str]] = defaultdict(set)
    citation_key_to_ids: dict[str, set[str]] = defaultdict(set)

    for chunk in chunks:
        metadata = chunk.metadata
        citation = metadata.get("citation")
        if isinstance(citation, str) and citation.strip():
            citation_to_ids[normalize_citation_key(citation)].add(chunk.chunk_id)

        citation_key = metadata.get("citation_key")
        if isinstance(citation_key, str) and citation_key.strip():
            citation_key_to_ids[normalize_citation_key(citation_key)].add(chunk.chunk_id)

    print(f"=== citation_to_ids entries: {len(citation_to_ids)} ===")
    print(f"=== citation_key_to_ids entries: {len(citation_key_to_ids)} ===")
    print()

    # Show a sample of what's in the indexes
    print("--- Sample citation_to_ids keys (first 15) ---")
    for key in sorted(citation_to_ids)[:15]:
        print(f"  {key!r} -> {len(citation_to_ids[key])} chunks")
    print()

    print("--- Sample citation_key_to_ids keys (first 15) ---")
    for key in sorted(citation_key_to_ids)[:15]:
        print(f"  {key!r} -> {len(citation_key_to_ids[key])} chunks")
    print()

    # --- Resolution check ---
    print("--- Resolution results ---")
    resolved = 0
    unresolved = 0
    for raw in sorted(query_citations):
        norm = normalize_citation_key(raw)
        ids = citation_to_ids.get(norm) or citation_key_to_ids.get(norm)
        if ids:
            resolved += 1
            print(f"  RESOLVED: {raw!r} (normalized: {norm!r}) -> {len(ids)} chunks")
        else:
            unresolved += 1
            print(f"  UNRESOLVED: {raw!r} (normalized: {norm!r})")

    print()
    print(f"=== Resolved: {resolved}/{resolved + unresolved} ===")
    print(f"=== Unresolved: {unresolved}/{resolved + unresolved} ===")

    # --- Check a few regulatory chunks' raw metadata ---
    print()
    print("--- Sample regulatory chunk metadata (first 5 with citation_key) ---")
    count = 0
    for ch in chunks:
        if ch.metadata.get("corpus") == "regulatory" and ch.metadata.get("citation_key"):
            print(f"  chunk_id={ch.chunk_id[:20]}...")
            print(f"    citation_key={ch.metadata.get('citation_key')!r}")
            print(f"    citation={ch.metadata.get('citation')!r}")
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
