from __future__ import annotations

from pathlib import Path

from rag.eval.harness import load_eval_queries


def test_regulatory_dataset_has_required_coverage() -> None:
    queries = load_eval_queries(Path("eval/datasets/regulatory_adversarial.jsonl"))

    assert len(queries) >= 25

    category_counts = {
        "citation-precision": 0,
        "abstention": 0,
        "cross-reference": 0,
        "hallucination-resistance": 0,
    }

    for query in queries:
        tags = set(query.tags)
        assert "regulatory" in tags
        for category in category_counts:
            if category in tags:
                category_counts[category] += 1

    for count in category_counts.values():
        assert count >= 5
