# Spec 02: Hybrid Search with Reciprocal Rank Fusion

## Title
Add BM25 Keyword Retrieval with RRF Fusion

## Context / Problem

Pure vector search misses exact matches for rare terms, acronyms, and keyword-heavy queries. This reduces recall for certain query types — a gap that becomes critical with regulatory corpora where precise term matching matters.

## Goals
- Add a `BM25Retriever` adapter implementing the `Retriever` port
- Add a `HybridRetriever` that fuses vector and keyword results via Reciprocal Rank Fusion
- Zero new dependencies (BM25 is ~50 lines of Python)
- Maintain the hexagonal port/adapter pattern

## Non-Goals
- Replacing vector search
- Distributed keyword index
- Fuzzy matching beyond basic tokenization

## Proposed Solution

### New File: `src/rag/adapters/retrieval/bm25_retriever.py`

BM25 retriever operating over in-memory chunks. No external dependencies.

```python
@dataclass
class BM25Retriever:
    """BM25 keyword retriever implementing the Retriever port."""
    _chunks: list[Chunk] = field(default_factory=list)
    _tokenized: list[list[str]] = field(default_factory=list)
    _idf: dict[str, float] = field(default_factory=dict)
    _avg_dl: float = 0.0
    k1: float = 1.5
    b: float = 0.75

    def index(self, chunks: list[Chunk]) -> None:
        """Build BM25 index from chunks."""
        self._chunks = chunks
        self._tokenized = [self._tokenize(c.text) for c in chunks]
        self._avg_dl = sum(len(t) for t in self._tokenized) / max(len(self._tokenized), 1)
        self._idf = self._compute_idf()

    def retrieve(self, query: str, *, top_k: int, where: Where = None) -> list[Candidate]:
        """Score chunks against query using BM25."""
        query_tokens = self._tokenize(query)
        scores = [self._score(query_tokens, doc_tokens) for doc_tokens in self._tokenized]
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            Candidate(chunk=self._chunks[i], retrieval_score=score, rerank_score=None)
            for i, score in ranked if score > 0
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase split. Intentionally simple — swap for stemmer if needed."""
        return text.lower().split()

    def _compute_idf(self) -> dict[str, float]: ...
    def _score(self, query_tokens, doc_tokens) -> float: ...
```

### New File: `src/rag/adapters/retrieval/hybrid_retriever.py`

```python
@dataclass(frozen=True, slots=True)
class HybridRetriever:
    """Fuses two retrievers via Reciprocal Rank Fusion."""
    primary: Retriever        # Vector retriever
    secondary: Retriever      # Keyword retriever
    primary_weight: float = 0.7
    secondary_weight: float = 0.3
    rrf_k: int = 60

    def retrieve(self, query: str, *, top_k: int, where: Where = None) -> list[Candidate]:
        primary_results = self.primary.retrieve(query, top_k=top_k * 2, where=where)
        secondary_results = self.secondary.retrieve(query, top_k=top_k * 2, where=where)
        return self._rrf_fuse(primary_results, secondary_results, top_k)

    def _rrf_fuse(
        self, a: list[Candidate], b: list[Candidate], top_k: int
    ) -> list[Candidate]:
        """RRF: score = sum(weight / (k + rank)) for each list containing the item."""
        scores: dict[str, float] = {}
        lookup: dict[str, Candidate] = {}

        for rank, cand in enumerate(a, start=1):
            cid = cand.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.primary_weight / (self.rrf_k + rank)
            lookup[cid] = cand

        for rank, cand in enumerate(b, start=1):
            cid = cand.chunk.chunk_id
            scores[cid] = scores.get(cid, 0) + self.secondary_weight / (self.rrf_k + rank)
            lookup.setdefault(cid, cand)

        ranked_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
        return [
            dataclasses.replace(lookup[cid], retrieval_score=scores[cid])
            for cid in ranked_ids
        ]
```

### Configuration

```toml
[retrieval]
backend = "vector"          # "vector" | "hybrid"
top_k = 8

[retrieval.hybrid]
primary_weight = 0.7
secondary_weight = 0.3
rrf_k = 60
bm25_k1 = 1.5
bm25_b = 0.75
```

### Container Integration

```python
if cfg.retrieval.backend == "hybrid":
    bm25 = BM25Retriever(k1=cfg.retrieval.hybrid.bm25_k1, b=cfg.retrieval.hybrid.bm25_b)
    bm25.index(chunks)
    retriever = HybridRetriever(
        primary=VectorRetriever(embedder=embedder, store=store),
        secondary=bm25,
        primary_weight=cfg.retrieval.hybrid.primary_weight,
        secondary_weight=cfg.retrieval.hybrid.secondary_weight,
    )
```

## Why Not Whoosh / External Libraries

- **Whoosh**: Unmaintained since 2015.
- **SQLite FTS5**: Adds complexity for marginal benefit at this scale.
- **Qdrant text search**: Ties hybrid retrieval to one backend, breaking portability.
- **rank_bm25**: Viable, but BM25 is ~50 lines and avoids a dependency.

## Acceptance Criteria

- [ ] `BM25Retriever` satisfies the `Retriever` protocol
- [ ] `HybridRetriever` produces fused results via RRF
- [ ] Items appearing in both result lists score higher than items in only one
- [ ] `backend = "vector"` remains default (no behavior change)
- [ ] Latency overhead < 50ms for typical queries
- [ ] No new pip dependencies

## Test Plan

```python
def test_bm25_finds_exact_terms():
    """BM25 retriever finds chunks containing exact query terms."""

def test_rrf_ranks_overlap_highest():
    """Items appearing in both lists get highest fused scores."""

def test_hybrid_recall_ge_vector_on_keyword_queries():
    """Hybrid recall >= vector-only recall on keyword/acronym queries."""

def test_vector_only_default():
    """Default config uses vector-only retriever, not hybrid."""
```

## Risks

| Risk | Mitigation |
|---|---|
| Simple tokenizer misses stemming benefits | Can swap `_tokenize()` for NLTK/snowball stemmer later; port stays the same |
| BM25 quality insufficient for regulatory text | Tune k1/b; can replace implementation behind same Retriever port |
| Memory overhead of BM25 index | Negligible for <100K chunks |
