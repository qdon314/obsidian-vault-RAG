from __future__ import annotations

from rag.app.container import build_container
from rag.app.pipeline import index_document, rag_answer
from rag.domain.models import Document


def main() -> None:
    c = build_container()

    doc = Document(
        doc_id="doc_demo_001",
        text=(
            "RAG systems often fail due to retrieval errors. "
            "Chunking strategy affects recall and precision. "
            "Reranking can improve relevance at the cost of latency. "
            "Evaluation metrics like Recall@K and MRR help quantify retrieval quality.\n"
        ) * 25,
        source="demo",
        uri="memory://demo",
        metadata={"title": "Demo Doc", "uri": "memory://demo"},
    )

    n = index_document(doc, chunker=c.chunker, embedder=c.embedder, store=c.store)
    print(f"Indexed chunks: {n} (store count: {c.store.count()})")

    ans = rag_answer(
        "How do you evaluate retrieval quality?",
        retriever=c.retriever,
        context_builder=c.context_builder,
        generator=c.generator,
        top_k=8,
        token_budget=1500,
    )

    print("\n=== ANSWER ===\n")
    print(ans.text)

    print("\n=== CITATIONS ===")
    for i, cit in enumerate(ans.citations, start=1):
        print(f"[{i}] {cit.doc_id} {cit.uri} chunk={cit.chunk_id}")


if __name__ == "__main__":
    main()
