from __future__ import annotations

import re
from collections import Counter

from rag.domain.models import Candidate

_WORD = re.compile(r"[A-Za-z0-9_]+")

def _tokens(s: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(s)]

class HeuristicReranker:
    """
    Cheap reranker:
    - starts from vector similarity score
    - adds lexical overlap boost between query and chunk text
    - optionally diversifies by doc_id (avoid many from same doc)
    """
    def __init__(self, *, overlap_weight: float = 0.15, diversify: bool = True, max_per_doc: int = 3):
        self.overlap_weight = overlap_weight
        self.diversify = diversify
        self.max_per_doc = max_per_doc

    @property
    def name(self) -> str:
        return "heuristic_v1"

    def rerank(self, query: str, candidates: list[Candidate], *, metadata=None) -> list[Candidate]:
        num_query_toks = Counter(_tokens(query))

        scored = []
        for cnd in candidates:
            # assumptions: Candidate has .chunk.text and .score and .chunk.metadata maybe.
            text = getattr(getattr(cnd, "chunk", None), "text", "") or ""
            num_cnd_toks = Counter(_tokens(text))

            # overlap = fraction of query tokens present (weighted)
            common = sum((num_query_toks & num_cnd_toks).values())
            denom = max(1, sum(num_query_toks.values()))
            overlap = common / denom

            base = float(getattr(cnd, "score", 0.0))
            new_score = base + self.overlap_weight * overlap
            scored.append((new_score, cnd))

        scored.sort(key=lambda x: x[0], reverse=True)

        if not self.diversify:
            return [c for _, c in scored]

        # simple doc diversification
        out: list[Candidate] = []
        per_doc: dict[str, int] = {}
        for _, cnd in scored:
            doc_id = ""
            md = getattr(getattr(cnd, "chunk", None), "metadata", None) or {}
            doc_id = str(md.get("doc_id", md.get("path", "")))

            n = per_doc.get(doc_id, 0)
            if doc_id and n >= self.max_per_doc:
                continue
            per_doc[doc_id] = n + 1
            out.append(cnd)

        return out
