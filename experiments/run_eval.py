from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag.app.container import build_container
from rag.app.pipeline import index_document, retrieve_candidates
from rag.domain.models import Document

from experiments.metrics import RetrievalResult, summarize


EVAL_QUERIES_PATH = Path("experiments/eval_queries.jsonl")
RESULTS_DIR = Path("experiments/results")


@dataclass(frozen=True, slots=True)
class EvalQuery:
    qid: str
    query: str
    relevant_chunk_ids: set[str]


def load_eval_queries(path: Path) -> list[EvalQuery]:
    items: list[EvalQuery] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(
            EvalQuery(
                qid=str(obj["qid"]),
                query=str(obj["query"]),
                relevant_chunk_ids=set(obj.get("relevant_chunk_ids", [])),
            )
        )
    return items


def main() -> None:
    if not EVAL_QUERIES_PATH.exists():
        raise FileNotFoundError(f"Missing {EVAL_QUERIES_PATH}. Create it first.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    c = build_container()

    # TODO: Replace this demo indexing with real ingestion/index build.
    doc = Document(
        doc_id="doc_demo_001",
        text=("RAG systems often fail due to retrieval errors. " * 200),
        source="demo",
        uri="memory://demo",
        metadata={"title": "Demo Doc", "uri": "memory://demo"},
    )
    index_document(doc, chunker=c.chunker, embedder=c.embedder, store=c.store)

    eval_queries = load_eval_queries(EVAL_QUERIES_PATH)

    results: list[RetrievalResult] = []
    per_query_rows: list[dict] = []

    for q in eval_queries:
        cands = retrieve_candidates(q.query, retriever=c.retriever, top_k=10)
        retrieved_ids = [cand.chunk.chunk_id for cand in cands]
        results.append(RetrievalResult(qid=q.qid, retrieved_chunk_ids=retrieved_ids, relevant_chunk_ids=q.relevant_chunk_ids))

        per_query_rows.append(
            {
                "qid": q.qid,
                "query": q.query,
                "relevant_count": len(q.relevant_chunk_ids),
                "retrieved_top10": retrieved_ids,
            }
        )

    summary = summarize(results, ks=(5, 10))
    print("=== Retrieval Eval Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    out_json = RESULTS_DIR / "latest_eval.json"
    out_json.write_text(json.dumps({"summary": summary, "per_query": per_query_rows}, indent=2), encoding="utf-8")
    print(f"\nWrote: {out_json}")


if __name__ == "__main__":
    main()
