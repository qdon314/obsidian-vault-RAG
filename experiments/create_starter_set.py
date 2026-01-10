"""
Create a starter evaluation set by analyzing existing vault content and query logs.

This script:
1. Analyzes your indexed chunks to understand content
2. Optionally uses existing query logs as inspiration
3. Creates a small curated starter set
4. Generates a template for manual expansion

Usage:
    python experiments/create_starter_set.py
"""

from __future__ import annotations

import json
import logging
import random
from datetime import UTC, datetime
from pathlib import Path

from rag.eval.schema import Difficulty, EvalQuery, QueryType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_chunks_sample(chunks_path: Path, sample_size: int = 20) -> list[dict]:
    """Load a sample of chunks from the index."""
    chunks = []
    if not chunks_path.exists():
        logger.warning(f"Chunks file not found: {chunks_path}")
        return chunks

    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if chunks:
        return random.sample(chunks, min(sample_size, len(chunks)))
    return chunks


def load_query_logs(logs_path: Path, limit: int = 10) -> list[dict]:
    """Load recent queries from logs for inspiration."""
    queries = []
    if not logs_path.exists():
        logger.warning(f"Query logs not found: {logs_path}")
        return queries

    with open(logs_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                queries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return queries[-limit:] if queries else []


def create_starter_queries(chunks: list[dict], existing_queries: list[dict]) -> list[EvalQuery]:
    """Create a starter set of evaluation queries."""
    starter_queries = []
    qid_counter = 1

    # Strategy 1: Use actual user queries if available
    if existing_queries:
        logger.info(f"Found {len(existing_queries)} existing queries in logs")
        for _, log_entry in enumerate(existing_queries[:5]):  # Take up to 5
            query_text = log_entry.get("query", "")
            if not query_text:
                continue

            # Try to get the retrieved chunks
            retrieved = log_entry.get("retrieved", [])
            relevant_chunks = set()

            if retrieved:
                # Take top 2-3 chunks as ground truth
                for cand in retrieved[:3]:
                    if isinstance(cand, dict):
                        chunk_id = cand.get("chunk", {}).get("chunk_id")
                        if chunk_id:
                            relevant_chunks.add(chunk_id)

            if relevant_chunks:
                starter_queries.append(
                    EvalQuery(
                        qid=f"starter_{qid_counter:03d}",
                        query=query_text,
                        relevant_chunk_ids=relevant_chunks,
                        expected_answer="TODO: Add expected answer for answer quality evaluation",
                        query_type=QueryType.FACTUAL,
                        difficulty=Difficulty.EASY,
                        notes="Derived from actual user query in logs",
                        tags=["starter", "from-logs"],
                        created_by="starter-script",
                        created_at=datetime.now(UTC).isoformat(),
                    )
                )
                qid_counter += 1

    # Strategy 2: Create template queries for manual curation
    template_queries = [
        {
            "query": "[TODO] What are the key points about [TOPIC]?",
            "type": QueryType.FACTUAL,
            "difficulty": Difficulty.EASY,
            "notes": "Replace [TOPIC] with a main concept from your vault",
        },
        {
            "query": "[TODO] How do I [ACCOMPLISH TASK]?",
            "type": QueryType.PROCEDURAL,
            "difficulty": Difficulty.MEDIUM,
            "notes": "Add a procedural question from your documentation",
        },
        {
            "query": "[TODO] What's the difference between [A] and [B]?",
            "type": QueryType.COMPARISON,
            "difficulty": Difficulty.MEDIUM,
            "notes": "Compare two concepts in your notes",
        },
        {
            "query": "[TODO] Why does [PHENOMENON] occur?",
            "type": QueryType.CAUSAL,
            "difficulty": Difficulty.HARD,
            "notes": "Add a causal reasoning question",
        },
        {
            "query": "[TODO] Question about topic NOT in vault",
            "type": QueryType.FACTUAL,
            "difficulty": Difficulty.EASY,
            "notes": "Negative example - should return no results",
        },
    ]

    # Add template queries
    for template in template_queries:
        is_negative = "NOT in vault" in template["query"]

        starter_queries.append(
            EvalQuery(
                qid=f"starter_{qid_counter:03d}",
                query=template["query"],
                relevant_chunk_ids=set() if is_negative else {"[TODO: Add chunk ID here]"},
                expected_answer="[TODO: Add expected answer]" if not is_negative else None,
                query_type=template["type"],
                difficulty=template["difficulty"],
                requires_synthesis=False,
                notes=template["notes"],
                tags=["starter", "template"],
                created_by="starter-script",
                created_at=datetime.now(UTC).isoformat(),
                is_unanswerable=is_negative,
                unanswerable_reason="not_in_corpus" if is_negative else None,
            )
        )
        qid_counter += 1

    # Strategy 3: Sample from chunks and create query suggestions
    if chunks:
        logger.info(f"Analyzing {len(chunks)} sample chunks for query ideas")
        for _, chunk_data in enumerate(chunks[:5]):  # Take up to 5 chunks
            chunk = chunk_data.get("chunk", {})
            chunk_id = chunk.get("chunk_id", "")
            chunk_text = chunk.get("text", "")

            if not chunk_id or len(chunk_text) < 100:
                continue

            # Create a template query based on chunk content
            # Extract first sentence or heading if available
            heading = chunk.get("section_heading", "")
            preview = chunk_text[:100] + "..." if len(chunk_text) > 100 else chunk_text

            starter_queries.append(
                EvalQuery(
                    qid=f"starter_{qid_counter:03d}",
                    query=f"[TODO: Write question about: {heading or 'this content'}]",
                    relevant_chunk_ids={chunk_id},
                    expected_answer="[TODO: Add expected answer based on chunk content]",
                    query_type=QueryType.FACTUAL,
                    difficulty=Difficulty.EASY,
                    notes=f"Based on chunk: {chunk_id}\nContent preview: {preview}",
                    tags=["starter", "from-chunks"],
                    created_by="starter-script",
                    created_at=datetime.now(UTC).isoformat(),
                    metadata={"source_chunk_id": chunk_id},
                )
            )
            qid_counter += 1

    return starter_queries


def main() -> None:
    # Paths
    chunks_path = Path("artifacts/indexes/obsidian_index/chunks.jsonl")
    logs_path = Path("artifacts/logs/queries.jsonl")
    output_path = Path("experiments/starter_queries.jsonl")

    logger.info("Creating starter evaluation set...")

    # Load data
    chunks = load_chunks_sample(chunks_path, sample_size=20)
    logger.info(f"Loaded {len(chunks)} sample chunks")

    existing_queries = load_query_logs(logs_path, limit=10)
    logger.info(f"Loaded {len(existing_queries)} recent queries from logs")

    # Create starter queries
    starter_queries = create_starter_queries(chunks, existing_queries)
    logger.info(f"Created {len(starter_queries)} starter queries")

    # Save to file as proper JSONL (one JSON object per line, no comments)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for query in starter_queries:
            f.write(json.dumps(query.to_dict()) + "\n")

    logger.info(f"Saved starter queries to {output_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("STARTER EVALUATION SET CREATED")
    print("=" * 70)
    print(f"Output: {output_path}")
    print(f"Total queries: {len(starter_queries)}")
    print("\nBreakdown:")
    print(f"  From logs: {len([q for q in starter_queries if 'from-logs' in q.tags])}")
    print(f"  Templates: {len([q for q in starter_queries if 'template' in q.tags])}")
    print(f"  From chunks: {len([q for q in starter_queries if 'from-chunks' in q.tags])}")
    print("\nNext steps:")
    print(f"  1. Edit {output_path} and fill in [TODO] placeholders")
    print(f"  2. Validate: python experiments/curate_queries.py --validate {output_path}")
    print(f"  3. Run eval: python experiments/eval_harness.py --queries {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
