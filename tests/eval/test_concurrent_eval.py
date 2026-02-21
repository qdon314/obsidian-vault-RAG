"""Tests for concurrent eval execution in the harness."""

from __future__ import annotations

from unittest.mock import MagicMock

from rag.adapters.context_building.simple_context_builder import SimpleContextBuilder
from rag.adapters.embedding.dummy_embedder import DummyEmbedder
from rag.adapters.reranking.rerank_noop import NoOpReranker
from rag.adapters.retrieval.vector_retriever import VectorRetriever
from rag.adapters.vectorstores.in_memory_store import InMemoryVectorStore
from rag.app.container import Container
from rag.eval.harness import run_full_eval
from rag.eval.schema import EvalQuery
from tests.conftest import make_chunk


def _build_test_container(*, n_chunks: int = 10) -> Container:
    """Build a container with dummy adapters and N indexed chunks."""
    embedder = DummyEmbedder(dim=128)
    store = InMemoryVectorStore()
    chunks = [
        make_chunk(
            chunk_id=f"doc-1:chunk:{i}",
            doc_id="doc-1",
            text=f"Chunk {i} about nuclear regulation {i}.",
            chunk_index=i,
            metadata={"citation": f"10 CFR 50.{i}"},
        )
        for i in range(n_chunks)
    ]
    vectors = embedder.embed_texts([c.text for c in chunks])
    store.upsert(chunks=chunks, vectors=vectors)

    retriever = VectorRetriever(embedder=embedder, store=store)
    generator = MagicMock()
    logger = MagicMock()
    reranker = NoOpReranker()
    context_builder = SimpleContextBuilder(max_chunks=5, dedupe=True)
    chunker = MagicMock()
    ingestor = MagicMock()

    return Container(
        chunker=chunker,
        context_builder=context_builder,
        embedder=embedder,
        generator=generator,
        ingestor=ingestor,
        store=store,
        retriever=retriever,
        logger=logger,
        reranker=reranker,
    )


def _make_queries(n: int = 20) -> list[EvalQuery]:
    """Generate N simple eval queries."""
    return [
        EvalQuery.from_dict({
            "qid": f"q-{i:03d}",
            "query": f"What are the requirements of 10 CFR 50.{i % 10}?",
            "relevant_citations": [f"10 CFR 50.{i % 10}"],
        })
        for i in range(n)
    ]


class TestConcurrentEval:
    """Concurrent eval produces same results as sequential."""

    def test_concurrent_retrieval_matches_sequential(self) -> None:
        """Results from max_workers=1 and max_workers=4 should be identical."""
        container = _build_test_container()
        queries = _make_queries(20)

        sequential = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
            max_workers=1,
        )
        concurrent = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
            max_workers=4,
        )

        assert len(sequential.results) == len(concurrent.results)

        seq_by_qid = {r.qid: r for r in sequential.results}
        con_by_qid = {r.qid: r for r in concurrent.results}
        for qid in seq_by_qid:
            assert seq_by_qid[qid].retrieval_result.retrieved_chunk_ids == \
                con_by_qid[qid].retrieval_result.retrieved_chunk_ids

    def test_max_workers_defaults_to_1(self) -> None:
        """run_full_eval should accept no max_workers arg (backward compat)."""
        container = _build_test_container()
        queries = _make_queries(5)

        result = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
        )
        assert len(result.results) == 5

    def test_result_order_matches_query_order(self) -> None:
        """Results should be in the same order as input queries."""
        container = _build_test_container()
        queries = _make_queries(15)

        result = run_full_eval(
            eval_queries=queries,
            container=container,
            queries_path=None,
            run_generation=False,
            max_workers=4,
        )
        result_qids = [r.qid for r in result.results]
        query_qids = [q.qid for q in queries]
        assert result_qids == query_qids
