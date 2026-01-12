#!/usr/bin/env python3
"""
Generate synthetic evaluation queries from your Obsidian vault.

This script:
1. Loads documents from your vault
2. Samples chunks from the indexed corpus
3. Uses an LLM to generate diverse, realistic queries
4. Creates an evaluation dataset with ground truth annotations

Usage:
    python experiments.generate_queries --num-queries 50 --output experiments/generated_queries.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from openai import OpenAI

from rag.domain.models import Chunk
from rag.eval.schema import Difficulty, EvalQuery, QueryType
from rag.settings import load_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


QUERY_GENERATION_PROMPT = """You are an expert at generating realistic, diverse evaluation queries for a Retrieval-Augmented Generation (RAG) system.

Given a text passage, generate {num_queries} high-quality questions that:
1. Can be answered using ONLY the information in this passage
2. Are natural and realistic (how a real user would ask)
3. Vary in complexity and style
4. Cover different aspects of the content

For each question, also provide:
- query_type: one of [factual, comparison, aggregation, procedural, definition, causal, temporal, multi_hop]
- difficulty: one of [easy, medium, hard]
- requires_synthesis: true if the answer requires combining multiple sentences/ideas from the passage

TEXT PASSAGE:
\"\"\"
{passage}
\"\"\"

Respond with a JSON array of objects, each with these fields:
- query: the question text
- query_type: the type of query
- difficulty: easy/medium/hard
- requires_synthesis: true/false
- notes: brief note on why this is a good evaluation question

Example response format:
[
  {{
    "query": "What is the main concept discussed in this passage?",
    "query_type": "factual",
    "difficulty": "easy",
    "requires_synthesis": false,
    "notes": "Tests basic comprehension"
  }},
  {{
    "query": "How do concepts X and Y relate to each other?",
    "query_type": "comparison",
    "difficulty": "medium",
    "requires_synthesis": true,
    "notes": "Tests ability to connect ideas"
  }}
]

Generate {num_queries} diverse questions:"""


NEGATIVE_QUERY_PROMPT = """Generate {num_queries} realistic questions that a user might ask about topics that are NOT covered in the following knowledge base excerpt.

These should be:
1. Plausible questions a user might ask
2. Related to but distinct from the topics in the excerpt
3. Clearly unanswerable given only this content

KNOWLEDGE BASE EXCERPT:
\"\"\"
{passage}
\"\"\"

Respond with a JSON array of objects:
[
  {{
    "query": "What is X?",
    "notes": "Related topic not covered"
  }}
]

Generate {num_queries} unanswerable questions:"""


def generate_queries_for_chunk(
    chunk: Chunk,
    client: OpenAI,
    num_queries: int = 3,
    model: str = "gpt-4o-mini",
    negative: bool = False,
) -> list[dict[str, Any]]:
    """
    Use LLM to generate queries for a given chunk.

    Args:
        chunk: The chunk to generate queries from
        client: OpenAI client
        num_queries: Number of queries to generate
        model: OpenAI model to use
        negative: If True, generate negative (unanswerable) queries

    Returns:
        List of query dictionaries
    """
    prompt = (
        NEGATIVE_QUERY_PROMPT if negative else QUERY_GENERATION_PROMPT
    ).format(passage=chunk.text[:2000], num_queries=num_queries)  # Limit passage length

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates evaluation queries."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,  # Higher temperature for diversity
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content
        if not content:
            logger.warning(f"Empty response for chunk {chunk.chunk_id}")
            return []

        # Try to parse JSON array or object with array
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                # Model might wrap in {"queries": [...]}
                parsed = parsed.get("queries", parsed.get("questions", []))
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}\nContent: {content[:200]}")
            return []

    except Exception as e:
        logger.error(f"Error generating queries for chunk {chunk.chunk_id}: {e}")
        return []


def load_chunks_from_index(store_path: Path) -> list[Chunk]:
    """
    Load chunks from the vector store.

    Args:
        store_path: Path to the JSONL vector store

    Returns:
        List of chunks
    """

    chunks = []
    if not store_path.exists():
        logger.warning(f"Store path {store_path} does not exist")
        return chunks

    with open(store_path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                chunks.append(Chunk.from_dict(data["chunk"]))
            except Exception as e:
                logger.error(f"Error loading chunk: {e}")
                continue

    return chunks


def sample_chunks_diverse(
    chunks: list[Chunk],
    num_samples: int,
    min_length: int = 200,
    max_length: int = 2000,
) -> list[Chunk]:
    """
    Sample chunks that are diverse and of reasonable length.

    Args:
        chunks: All available chunks
        num_samples: Number of chunks to sample
        min_length: Minimum chunk length in characters
        max_length: Maximum chunk length in characters

    Returns:
        Sampled chunks
    """
    # Filter by length
    candidates = [
        c for c in chunks
        if min_length <= len(c.text) <= max_length
    ]

    if not candidates:
        logger.warning("No chunks meet length criteria, using all chunks")
        candidates = chunks

    # Group by document to ensure diversity
    by_doc: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in candidates:
        by_doc[chunk.doc_id].append(chunk)

    # Sample from different documents when possible
    sampled = []
    docs_list = list(by_doc.keys())
    random.shuffle(docs_list)

    for doc_id in docs_list:
        if len(sampled) >= num_samples:
            break
        doc_chunks = by_doc[doc_id]
        # Take one chunk per document for diversity
        sampled.append(random.choice(doc_chunks))

    # If we need more, sample randomly from remaining
    if len(sampled) < num_samples:
        remaining = [c for c in candidates if c not in sampled]
        additional = random.sample(remaining, min(num_samples - len(sampled), len(remaining)))
        sampled.extend(additional)

    return sampled[:num_samples]


def generate_eval_dataset(
    chunks: list[Chunk],
    num_queries: int,
    client: OpenAI,
    queries_per_chunk: int = 3,
    negative_ratio: float = 0.1,
    model: str = "gpt-4o-mini",
) -> list[EvalQuery]:
    """
    Generate a complete evaluation dataset.

    Args:
        chunks: Chunks to generate queries from
        num_queries: Total number of queries to generate
        client: OpenAI client
        queries_per_chunk: How many queries to generate per chunk
        negative_ratio: Fraction of queries that should be unanswerable
        model: OpenAI model to use

    Returns:
        List of EvalQuery objects
    """
    num_negative = int(num_queries * negative_ratio)
    num_positive = num_queries - num_negative

    num_chunks_positive = max(1, num_positive // queries_per_chunk)
    num_chunks_negative = max(1, num_negative // queries_per_chunk) if num_negative > 0 else 0

    logger.info(f"Sampling {num_chunks_positive} chunks for positive queries")
    positive_chunks = sample_chunks_diverse(chunks, num_chunks_positive)

    eval_queries = []
    qid_counter = 1

    # Generate positive queries
    for chunk in positive_chunks:
        logger.info(f"Generating {queries_per_chunk} queries for chunk {chunk.chunk_id}")
        generated = generate_queries_for_chunk(
            chunk,
            client,
            num_queries=queries_per_chunk,
            model=model,
            negative=False,
        )

        for item in generated:
            eval_queries.append(
                EvalQuery(
                    qid=f"synthetic_{qid_counter:04d}",
                    query=item.get("query", ""),
                    relevant_chunk_ids={chunk.chunk_id},
                    query_type=QueryType(item.get("query_type", "factual")),
                    difficulty=Difficulty(item.get("difficulty", "easy")),
                    requires_synthesis=item.get("requires_synthesis", False),
                    notes=item.get("notes"),
                    created_by="synthetic",
                    created_at=datetime.now(UTC).isoformat(),
                    tags=["synthetic"],
                )
            )
            qid_counter += 1

    # Generate negative queries
    if num_chunks_negative > 0:
        logger.info(f"Sampling {num_chunks_negative} chunks for negative queries")
        negative_chunks = sample_chunks_diverse(chunks, num_chunks_negative)

        for chunk in negative_chunks:
            logger.info(f"Generating negative queries for chunk {chunk.chunk_id}")
            generated = generate_queries_for_chunk(
                chunk,
                client,
                num_queries=queries_per_chunk,
                model=model,
                negative=True,
            )

            for item in generated:
                eval_queries.append(
                    EvalQuery(
                        qid=f"synthetic_{qid_counter:04d}",
                        query=item.get("query", ""),
                        relevant_chunk_ids=set(),
                        query_type=QueryType.FACTUAL,
                        difficulty=Difficulty.EASY,
                        is_unanswerable=True,
                        unanswerable_reason="not_in_corpus",
                        notes=item.get("notes"),
                        created_by="synthetic",
                        created_at=datetime.now(UTC).isoformat(),
                        tags=["synthetic", "negative"],
                    )
                )
                qid_counter += 1

    return eval_queries


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic evaluation queries")
    parser.add_argument(
        "--num-queries",
        type=int,
        default=50,
        help="Total number of queries to generate",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=Path("artifacts/indexes/obsidian_index/chunks.jsonl"),
        help="Path to JSONL vector store",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/generated_queries.jsonl"),
        help="Output path for generated queries",
    )
    parser.add_argument(
        "--queries-per-chunk",
        type=int,
        default=3,
        help="Number of queries to generate per chunk",
    )
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=0.1,
        help="Fraction of queries that should be unanswerable (0.0-1.0)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use for generation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    random.seed(args.seed)

    # Load chunks
    logger.info(f"Loading chunks from {args.store_path}")
    chunks = load_chunks_from_index(args.store_path)
    logger.info(f"Loaded {len(chunks)} chunks")

    if not chunks:
        logger.error("No chunks loaded. Have you built the index?")
        return

    api_key = load_settings().secrets.openai_api_key
    if not api_key:
        logger.error("OpenAI API key not found in settings.")
        return
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    # Generate queries
    logger.info(f"Generating {args.num_queries} queries...")
    eval_queries = generate_eval_dataset(
        chunks=chunks,
        num_queries=args.num_queries,
        client=client,
        queries_per_chunk=args.queries_per_chunk,
        negative_ratio=args.negative_ratio,
        model=args.model,
    )

    # Write output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for query in eval_queries:
            f.write(json.dumps(query.to_dict()) + "\n")

    logger.info(f"Wrote {len(eval_queries)} queries to {args.output}")
    logger.info(
        f"Stats: {len([q for q in eval_queries if not q.is_unanswerable])} answerable, "
        f"{len([q for q in eval_queries if q.is_unanswerable])} unanswerable"
    )


if __name__ == "__main__":
    main()
