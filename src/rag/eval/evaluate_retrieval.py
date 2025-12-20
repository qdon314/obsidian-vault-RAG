import json
from pathlib import Path
from statistics import mean

from src.rag.index import build_or_load_index
from config import CHROMA_PATH

def load_jsonl(path: str):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield json.loads(line)

def main(eval_path: str, k: int = 5):
    index = build_or_load_index(docs=None, chroma_path=CHROMA_PATH)
    retriever = index.as_retriever(similarity_top_k=30)

    hits = []
    mrrs = []
    dupes = []

    for row in load_jsonl(eval_path):
        row_query = row["query"]
        expected_files = set(row.get("expected_files", []))

        nodes = retriever.retrieve(row_query, )
        files = [(n.node.metadata or {}).get("file_name") for n in nodes]
        files_k = files[:k]

        # Recall@k
        hit = any(file in expected_files for file in files_k if file)
        hits.append(1 if hit else 0)

        # MRR
        rr = 0.0
        for i, f in enumerate(files_k):
            if f in expected_files:
                rr = 1.0 / (i + 1)
                break
        mrrs.append(rr)

        # Duplicate ratio @k
        seen = set()
        d = 0
        for f in files_k:
            if f in seen:
                d += 1
            seen.add(f)
        dupes.append(d)

    print(f"n={len(hits)}  Recall@{k}={mean(hits):.3f}  MRR@{k}={mean(mrrs):.3f}  avg_dupes@{k}={mean(dupes):.2f}")

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("eval_path")
    p.add_argument("--k", type=int, default=5)
    args = p.parse_args()
    main(args.eval_path, k=args.k)
