#!/usr/bin/env python3
"""
Helper script for manually curating evaluation queries.

This script helps you:
1. Browse your vault content
2. Create evaluation queries with ground truth
3. Validate and save them in the proper format
4. (Optional) Merge multiple query files

Usage:
    python -m experiments.curate_queries --interactive
    python -m experiments.curate_queries --from-template experiments/query_template.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from rag.eval.schema import Difficulty, EvalQuery, QueryType

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_template_file(output_path: Path, num_examples: int = 10) -> None:
    """
    Create a template JSONL file for manual curation.

    Args:
        output_path: Where to save the template
        num_examples: Number of template entries to create
    """
    templates = []

    # Create diverse templates
    template_configs = [
        {
            "query": "EDIT: What is the main concept of X?",
            "query_type": QueryType.FACTUAL,
            "difficulty": Difficulty.EASY,
            "notes": "Simple factual question - replace X and add chunk IDs",
        },
        {
            "query": "EDIT: How do I accomplish task Y?",
            "query_type": QueryType.PROCEDURAL,
            "difficulty": Difficulty.MEDIUM,
            "notes": "Procedural question - explain relevant chunk locations",
        },
        {
            "query": "EDIT: What's the difference between A and B?",
            "query_type": QueryType.COMPARISON,
            "difficulty": Difficulty.MEDIUM,
            "notes": "Comparison question - may require multiple chunks",
        },
        {
            "query": "EDIT: Why does phenomenon Z occur?",
            "query_type": QueryType.CAUSAL,
            "difficulty": Difficulty.HARD,
            "notes": "Causal reasoning question",
        },
        {
            "query": "EDIT: Question about topic not in your vault",
            "query_type": QueryType.FACTUAL,
            "difficulty": Difficulty.EASY,
            "notes": "Negative example - should return no results",
        },
    ]

    for i in range(num_examples):
        config = template_configs[i % len(template_configs)]
        query = EvalQuery(
            qid=f"manual_{i + 1:03d}",
            query=config["query"],
            relevant_chunk_ids={"EDIT: doc_id:chunking_strategy:chunk_idx:start-end"},
            expected_answer="EDIT: Write the expected answer here (optional but recommended)",
            query_type=config["query_type"],
            difficulty=config["difficulty"],
            requires_synthesis=False,
            notes=config["notes"],
            created_by="manual",
            created_at=datetime.now(UTC).isoformat(),
            tags=["manual", "TODO"],
            is_unanswerable=(i % 5 == 4),  # Every 5th is unanswerable
        )

        if query.is_unanswerable:
            query = EvalQuery(
                **{
                    **query.to_dict(),
                    "relevant_chunk_ids": set(),
                    "unanswerable_reason": "not_in_corpus",
                }
            )

        templates.append(query)

    # Write to file
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("# EVALUATION QUERY TEMPLATE\n")
        f.write("# Edit each line below to create your evaluation queries\n")
        f.write("# Remove lines starting with # before running validation\n")
        f.write("#\n")
        f.write("# Instructions:\n")
        f.write("# 1. Replace 'EDIT: ...' with your actual query\n")
        f.write("# 2. Update relevant_chunk_ids with actual chunk IDs from your index\n")
        f.write("# 3. Add expected_answer if you want to evaluate answer quality\n")
        f.write("# 4. Adjust query_type, difficulty, and other metadata\n")
        f.write(
            "# 5. For negative examples, set is_unanswerable: true and clear relevant_chunk_ids\n"
        )
        f.write("#\n")
        f.write("# To find chunk IDs, you can:\n")
        f.write("# - Look at artifacts/data/chunks_with_embeddings.jsonl\n")
        f.write("# - Run a query and examine the retrieved chunk IDs\n")
        f.write("#\n\n")

        for query in templates:
            f.write(json.dumps(query.to_dict(), indent=2) + "\n")

    logger.info(f"Created template with {len(templates)} examples at {output_path}")
    logger.info(
        f"Edit the file and run: python experiments/curate_queries.py --validate {output_path}"
    )


def validate_queries(input_path: Path) -> list[EvalQuery]:
    """
    Validate and parse manually curated queries.

    Args:
        input_path: Path to JSONL file with curated queries

    Returns:
        List of validated EvalQuery objects
    """
    queries = []
    errors = []

    with open(input_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            try:
                data = json.loads(line)
                query = EvalQuery.from_dict(data)

                # Validate required fields
                if not query.query or query.query.startswith("EDIT:"):
                    errors.append(f"Line {line_num}: Query not edited: {query.query}")
                    continue

                if not query.is_unanswerable and not query.relevant_chunk_ids:
                    errors.append(
                        f"Line {line_num}: Answerable query has no relevant chunks: {query.qid}"
                    )
                    continue

                if query.is_unanswerable and query.relevant_chunk_ids:
                    errors.append(
                        f"Line {line_num}: Unanswerable query has relevant chunks: {query.qid}"
                    )
                    continue

                # Check for placeholder chunk IDs
                if any("EDIT:" in cid for cid in query.relevant_chunk_ids):
                    errors.append(f"Line {line_num}: Chunk IDs not edited: {query.qid}")
                    continue

                queries.append(query)

            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON: {e}")
            except Exception as e:
                errors.append(f"Line {line_num}: Validation error: {e}")

    if errors:
        logger.error(f"Found {len(errors)} validation errors:")
        for error in errors:
            logger.error(f"  {error}")
        raise ValueError(f"Validation failed with {len(errors)} errors")

    logger.info(f"Validated {len(queries)} queries successfully")
    return queries


def merge_query_files(input_paths: list[Path], output_path: Path) -> None:
    """
    Merge multiple query JSONL files into one.

    Args:
        input_paths: List of input JSONL files
        output_path: Output path for merged file
    """
    all_queries = []
    seen_qids = set()

    for path in input_paths:
        logger.info(f"Loading queries from {path}")
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                data = json.loads(line)
                query = EvalQuery.from_dict(data)

                # Check for duplicate QIDs
                if query.qid in seen_qids:
                    logger.warning(f"Duplicate QID found: {query.qid}, skipping")
                    continue

                all_queries.append(query)
                seen_qids.add(query.qid)

    # Write merged file
    with open(output_path, "w", encoding="utf-8") as f:
        for query in all_queries:
            f.write(json.dumps(query.to_dict()) + "\n")

    logger.info(f"Merged {len(all_queries)} queries into {output_path}")


def show_stats(queries: list[EvalQuery]) -> None:
    """Print statistics about query dataset."""
    print("\n" + "=" * 60)
    print("QUERY DATASET STATISTICS")
    print("=" * 60)
    print(f"Total queries: {len(queries)}")
    print("\nBy type:")
    for qtype in QueryType:
        count = len([q for q in queries if q.query_type == qtype])
        if count > 0:
            print(f"  {qtype.value}: {count}")

    print("\nBy difficulty:")
    for diff in Difficulty:
        count = len([q for q in queries if q.difficulty == diff])
        if count > 0:
            print(f"  {diff.value}: {count}")

    answerable = len([q for q in queries if not q.is_unanswerable])
    unanswerable = len([q for q in queries if q.is_unanswerable])
    print(f"\nAnswerable: {answerable}")
    print(f"Unanswerable: {unanswerable}")

    with_expected = len([q for q in queries if q.expected_answer])
    print(f"With expected answer: {with_expected}")

    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate evaluation queries manually")
    parser.add_argument(
        "--create-template",
        type=Path,
        help="Create a template file for manual editing",
    )
    parser.add_argument(
        "--num-templates",
        type=int,
        default=20,
        help="Number of template entries to create",
    )
    parser.add_argument(
        "--validate",
        type=Path,
        help="Validate a manually curated query file",
    )
    parser.add_argument(
        "--merge",
        nargs="+",
        type=Path,
        help="Merge multiple query files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/curated_queries.jsonl"),
        help="Output path for validated/merged queries",
    )
    parser.add_argument(
        "--stats",
        type=Path,
        help="Show statistics for a query file",
    )

    args = parser.parse_args()

    if args.create_template:
        create_template_file(args.create_template, args.num_templates)

    elif args.validate:
        queries = validate_queries(args.validate)
        show_stats(queries)

        # Save validated queries
        with open(args.output, "w", encoding="utf-8") as f:
            for query in queries:
                f.write(json.dumps(query.to_dict()) + "\n")
        logger.info(f"Saved {len(queries)} validated queries to {args.output}")

    elif args.merge:
        merge_query_files(args.merge, args.output)

    elif args.stats:
        with open(args.stats, encoding="utf-8") as f:
            queries = []
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    queries.append(EvalQuery.from_dict(json.loads(line)))
        show_stats(queries)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
