#!/usr/bin/env python3
"""Generate evaluation queries from NRC case documents.

Reads case markdown files from the corpus directory, applies TermMapper +
CaseQueryGenerator, and writes eval-compatible JSONL output.

Usage examples::

    ./scripts/py scripts/generate_case_queries.py
    ./scripts/py scripts/generate_case_queries.py --dry-run
    ./scripts/py scripts/generate_case_queries.py --strategies 1,3 --output queries.jsonl
    ./scripts/py scripts/generate_case_queries.py --max-total 200
    ./scripts/py scripts/generate_case_queries.py --corpus-dir corpus/us-nrc/cases/2024-01/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag.adapters.query_generation.case_query_generator import CaseQueryGenerator
from rag.adapters.query_generation.term_mapper import TermMapper

# Strategy tag used for post-filtering
_STRATEGY_TAGS = {
    1: "citation-direct",
    2: "term-mapping",
    3: "scenario-based",
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate evaluation queries from NRC case documents.",
    )
    parser.add_argument(
        "--corpus-dir",
        default="corpus/us-nrc/cases",
        help="Path to case markdown directory (default: corpus/us-nrc/cases).",
    )
    parser.add_argument(
        "--term-map",
        default="config/case_regulatory_terms.json",
        help="Path to term map JSON (default: config/case_regulatory_terms.json).",
    )
    parser.add_argument(
        "--output",
        default="eval/datasets/case_generated_queries.jsonl",
        help="Output JSONL path (default: eval/datasets/case_generated_queries.jsonl).",
    )
    parser.add_argument(
        "--strategies",
        default="1,2,3",
        help="Comma-separated strategy numbers to include (default: 1,2,3).",
    )
    parser.add_argument(
        "--max-per-case",
        type=int,
        default=50,
        help="Max queries per case file (default: 50).",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=None,
        help="Max total queries to output (default: unlimited). Applied after strategy filtering.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print statistics without writing the output file.",
    )
    return parser


def _parse_strategies(raw: str) -> set[int]:
    """Parse comma-separated strategy numbers into a set."""
    try:
        nums = {int(s.strip()) for s in raw.split(",")}
    except ValueError:
        raise SystemExit(f"--strategies must be comma-separated integers, got: {raw!r}") from None
    invalid = nums - {1, 2, 3}
    if invalid:
        raise SystemExit(f"Invalid strategy numbers: {invalid}. Valid: 1, 2, 3") from None
    return nums


def _filter_by_strategy(queries: list[dict], strategies: set[int]) -> list[dict]:
    """Post-filter queries to only include selected strategies."""
    if strategies == {1, 2, 3}:
        return queries
    allowed_tags = {_STRATEGY_TAGS[s] for s in strategies}
    return [q for q in queries if any(t in allowed_tags for t in q.get("tags", []))]


def _cap_equal_by_strategy(queries: list[dict], max_total: int) -> list[dict]:
    """Cap total queries while distributing evenly across strategy buckets.

    Each bucket gets ``max_total // num_buckets`` slots.  If a bucket has
    fewer queries than its share, the surplus is redistributed to the
    remaining buckets in a second pass.
    """
    # Group by first matching strategy tag
    buckets: dict[str, list[dict]] = {}
    for q in queries:
        for tag_name in _STRATEGY_TAGS.values():
            if tag_name in q.get("tags", []):
                buckets.setdefault(tag_name, []).append(q)
                break

    if not buckets:
        return queries[:max_total]

    # Distribute budget equally, then redistribute surplus from small buckets
    remaining = max_total
    per_bucket = remaining // len(buckets)
    result: list[dict] = []
    overflow_buckets: list[list[dict]] = []

    for items in buckets.values():
        if len(items) <= per_bucket:
            result.extend(items)
            remaining -= len(items)
        else:
            overflow_buckets.append(items)

    # Redistribute remaining budget across overflow buckets
    if overflow_buckets:
        per_overflow = remaining // len(overflow_buckets)
        for items in overflow_buckets:
            result.extend(items[:per_overflow])

    return result


def main() -> None:
    args = build_argparser().parse_args()

    corpus_dir = Path(args.corpus_dir).expanduser().resolve()
    term_map_path = Path(args.term_map).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    strategies = _parse_strategies(args.strategies)

    if not corpus_dir.is_dir():
        raise SystemExit(f"Corpus directory not found: {corpus_dir}")
    if not term_map_path.is_file():
        raise SystemExit(f"Term map file not found: {term_map_path}")

    mapper = TermMapper.from_json(term_map_path)
    gen = CaseQueryGenerator(term_mapper=mapper, max_queries_per_case=args.max_per_case)

    case_files = sorted(corpus_dir.rglob("*.md"))
    if not case_files:
        raise SystemExit(f"No .md files found in {corpus_dir}")

    print(f"Processing {len(case_files)} case files...", file=sys.stderr)

    all_queries: list[dict] = []
    files_with_queries = 0
    for case_file in case_files:
        queries = gen.generate(case_file)
        if queries:
            files_with_queries += 1
        all_queries.extend(queries)

    # Post-filter by strategy
    filtered = _filter_by_strategy(all_queries, strategies)

    # Apply total cap with equal distribution across strategies
    if args.max_total is not None and len(filtered) > args.max_total:
        filtered = _cap_equal_by_strategy(filtered, args.max_total)

    # Count by strategy tag
    counts: dict[str, int] = {}
    for q in filtered:
        for _tag_num, tag_name in _STRATEGY_TAGS.items():
            if tag_name in q.get("tags", []):
                counts[tag_name] = counts.get(tag_name, 0) + 1
                break

    # Print summary
    print("\n--- Query Generation Summary ---", file=sys.stderr)
    print(f"{'Strategy':<20} {'Count':>6}", file=sys.stderr)
    print("-" * 28, file=sys.stderr)
    for tag_name in ("citation-direct", "term-mapping", "scenario-based"):
        if tag_name in counts:
            print(f"{tag_name:<20} {counts[tag_name]:>6}", file=sys.stderr)
    print("-" * 28, file=sys.stderr)
    print(
        f"{'Total':<20} {len(filtered):>6} queries from {files_with_queries} case files",
        file=sys.stderr,
    )

    if args.dry_run:
        print("\n(dry-run: no file written)", file=sys.stderr)
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for q in filtered:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
    print(f"\nWritten to {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
