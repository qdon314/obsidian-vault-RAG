#!/usr/bin/env python3
"""
Run RAG evaluation against an evaluation query set.

This script loads evaluation queries from a JSONL file, runs them through the
RAG pipeline, computes retrieval metrics (recall, precision, MRR, NDCG, MAP),
and optionally evaluates answer quality using an LLM-as-judge.

Modes:
    - Retrieval-only (default): Tests retrieval quality without generation.
    - Full pipeline (--run-generation): Runs retrieve → rerank → generate and
      measures end-to-end latency. Optionally scores answers with --use-llm-judge.

Outputs:
    - Console summary of metrics
    - results_{run_name}.jsonl: Per-query detailed results
    - metrics_{run_name}.json: Aggregated metrics and run metadata

Examples:
    # Retrieval-only evaluation
    python -m eval.scripts.run_eval --queries experiments/eval_queries.jsonl

    # Full pipeline with LLM judge
    python -m eval.scripts.run_eval --queries experiments/eval_queries.jsonl \\
        --run-generation --use-llm-judge --judge-model gpt-4o-mini

    # Custom retrieval settings
    python -m eval.scripts.run_eval --queries experiments/eval_queries.jsonl \\
        --top-k 20 --keep-k 5 --token-budget 2000
"""
from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from rag.app.container import ContainerOverrides, build_container
from rag.eval.harness import (  # your runner function returning EvalRun
    load_eval_queries,
    run_full_eval,
    save_run,
)
from rag.settings import load_settings

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG evaluation")
    parser.add_argument("--queries", type=Path, default=Path("eval/datasets/generated_queries.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("eval/runs"))
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--keep-k", type=int, default=None)
    parser.add_argument("--token-budget", type=int, default=1500)

    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")

    # when generation is on: choose whether to score retrieved or reranked ids
    parser.add_argument(
        "--score-ids",
        type=str,
        choices=("retrieved", "reranked"),
        default="reranked",
        help="Which chunk-id list to score for retrieval metrics when generation is enabled.",
    )

    parser.add_argument("--no-save", action="store_true", help="Do not write artifacts to disk.")

    args = parser.parse_args()
    load_dotenv()

    # Create timestamped run directory
    timestamp = datetime.now(UTC).strftime("%Y_%m_%dT%H-%M")
    run_dir = args.output / f"run_{timestamp}"

    # Load eval queries
    if not args.queries.exists():
        raise FileNotFoundError(f"Missing eval queries: {args.queries}")

    logger.info("Loading queries: %s", args.queries)
    eval_queries = load_eval_queries(args.queries)
    logger.info("Loaded %d queries", len(eval_queries))

    overrides = ContainerOverrides(
        store_backend="jsonl",
        jsonl_index_dir=Path("artifacts/indexes/obsidian"),
        logs_directory=Path(run_dir),
        cache_embeddings=False
    )
    
    # Build container (your app wiring)
    logger.info("Building container")
    container = build_container(overrides=overrides)

    if args.use_llm_judge:
        logger.info("Using LLM judge with model: %s", args.judge_model)
        api_key = load_settings().secrets.openai_api_key
        if not api_key:
            raise ValueError("OpenAI API key not found in settings.")
        judge_client = OpenAI(api_key=api_key)
    else:
        judge_client = None

    # Run eval
    run = run_full_eval(
        eval_queries=eval_queries,
        container=container,
        queries_path=str(args.queries),
        top_k=args.top_k,
        keep_k=args.keep_k,
        token_budget=args.token_budget,
        run_generation=args.run_generation,
        use_llm_judge=args.use_llm_judge,
        judge_client=judge_client,
        judge_model=args.judge_model if args.use_llm_judge else None,
        score_ids=args.score_ids,
    )

    # Persist artifacts
    if not args.no_save:
        run = save_run(run, output_dir=run_dir, run_name=args.run_name)

    # Print summary
    # overall = run.aggregates.overall.to_dict()  # flat dict, or use fields directly

    print("\n" + "=" * 72)
    print("EVAL RUN")
    print("=" * 72)
    print(f"run_id:        {run.meta.run_id}")
    print(f"queries:       {run.meta.queries_path}")
    print(f"generation:    {run.meta.run_generation}")
    print(f"top_k/keep_k:  {run.meta.top_k}/{run.meta.keep_k}")
    print(f"token_budget:  {run.meta.token_budget}")
    if run.meta.use_llm_judge:
        print(f"judge_model:   {run.meta.judge_model}")
    print("-" * 72)
    print(f"num_queries:   {run.aggregates.overall.num_queries}")
    print(f"avg_retrieved: {run.aggregates.overall.avg_retrieved:.2f}")
    print(f"mrr:           {run.aggregates.overall.mrr:.4f}")
    print(f"map:           {run.aggregates.overall.map:.4f}")

    for k in sorted(run.aggregates.overall.recall_at_k):
        print(f"recall@{k}:     {run.aggregates.overall.recall_at_k[k]:.4f}")
        print(f"ndcg@{k}:       {run.aggregates.overall.ndcg_at_k[k]:.4f}")

    if run.aggregates.answer_quality:
        print("-" * 72)
        for k, v in run.aggregates.answer_quality.items():
            print(f"{k}: {v:.4f}")

    if run.aggregates.latency_ms:
        print("-" * 72)
        for k, v in run.aggregates.latency_ms.items():
            print(f"latency_{k}_ms: {v:.1f}")

    if run.artifacts:
        print("-" * 72)
        for k, v in run.artifacts.items():
            print(f"{k}: {v}")

    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
