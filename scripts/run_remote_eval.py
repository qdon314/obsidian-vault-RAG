"""CLI: Run evaluation against remote backends (Qdrant + S3 chunk store).

Downloads eval queries from S3, runs the harness, uploads results to S3.
Designed to run as an ECS task (query-eval).

Usage:
    ./scripts/py scripts/run_remote_eval.py \
        --query-set default \
        --run-name my-eval-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import boto3  # type: ignore[import-untyped]
from dotenv import load_dotenv

from rag.app.container import build_container
from rag.eval.harness import load_eval_queries, run_full_eval, save_run
from rag.settings import load_settings

log = logging.getLogger("remote-eval")


def _download_s3_prefix(bucket: str, prefix: str, local_dir: Path) -> list[Path]:
    """Download all objects under an S3 prefix to a local directory."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    downloaded: list[Path] = []

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            rel = key[len(prefix):].lstrip("/")
            if not rel:
                continue
            local_path = local_dir / rel
            local_path.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(bucket, key, str(local_path))
            downloaded.append(local_path)
            log.info("Downloaded s3://%s/%s -> %s", bucket, key, local_path)

    return downloaded


def _upload_directory(local_dir: Path, bucket: str, prefix: str) -> None:
    """Upload all files in a local directory to S3."""
    s3 = boto3.client("s3")
    for path in local_dir.rglob("*"):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(local_dir)}"
            s3.upload_file(str(path), bucket, key)
            log.info("Uploaded %s -> s3://%s/%s", path.name, bucket, key)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run eval against remote backends.")
    ap.add_argument("--query-set", default="default", help="Name of query set in S3.")
    ap.add_argument("--run-name", default=None, help="Optional run name label.")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--keep-k", type=int, default=None)
    ap.add_argument("--token-budget", type=int, default=1500)
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--use-llm-judge", action="store_true")
    ap.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    ap.add_argument("--score-ids", choices=("retrieved", "reranked"), default="reranked")
    ap.add_argument("--manifest", type=str, default=None, help="Manifest URI (local or s3://).")
    args = ap.parse_args()

    load_dotenv()
    logging.basicConfig(format="[remote-eval] %(message)s", level=logging.INFO)

    cfg = load_settings()

    # Determine S3 bucket and prefixes
    bucket = cfg.distributed_ingestion.corpus_s3_bucket or cfg.chunk_storage.s3_bucket
    if not bucket:
        log.error("No S3 bucket configured (need distributed_ingestion.corpus_s3_bucket or chunk_storage.s3_bucket)")
        raise SystemExit(1)

    eval_prefix = os.environ.get("RAG_EVAL_S3_PREFIX", "eval")

    # ── Download eval queries from S3 ──────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        queries_dir = Path(tmpdir) / "queries"
        queries_dir.mkdir()

        s3_queries_prefix = f"{eval_prefix}/queries/{args.query_set}"
        log.info("Downloading eval queries from s3://%s/%s", bucket, s3_queries_prefix)
        downloaded = _download_s3_prefix(bucket, s3_queries_prefix, queries_dir)

        if not downloaded:
            log.error("No eval queries found at s3://%s/%s", bucket, s3_queries_prefix)
            raise SystemExit(1)

        # Find the queries JSONL file
        jsonl_files = [f for f in downloaded if f.suffix == ".jsonl"]
        if not jsonl_files:
            log.error("No .jsonl file found in downloaded queries")
            raise SystemExit(1)

        queries_path = jsonl_files[0]
        log.info("Using queries file: %s", queries_path)
        eval_queries = load_eval_queries(queries_path)
        log.info("Loaded %d eval queries", len(eval_queries))

        # ── Build container with remote backends ───────────────────
        container = build_container(cfg=cfg)

        # ── LLM judge setup ────────────────────────────────────────
        judge_client = None
        if args.use_llm_judge:
            from openai import OpenAI
            api_key = cfg.secrets.openai_api_key
            if not api_key:
                raise ValueError("OpenAI API key required for LLM judge")
            judge_client = OpenAI(api_key=api_key)

        # ── Load manifest if provided ──────────────────────────────
        manifest = None
        if args.manifest:
            from rag.domain.index_manifest import IndexManifest
            manifest = IndexManifest.load_uri(args.manifest)

        # ── Run eval ───────────────────────────────────────────────
        log.info("Running evaluation...")
        run = run_full_eval(
            eval_queries=eval_queries,
            container=container,
            queries_path=str(queries_path),
            manifest=manifest,
            top_k=args.top_k,
            keep_k=args.keep_k,
            token_budget=args.token_budget,
            run_generation=args.run_generation,
            use_llm_judge=args.use_llm_judge,
            judge_client=judge_client,
            judge_model=args.judge_model if args.use_llm_judge else None,
            score_ids=args.score_ids,
            run_name=args.run_name,
        )

        # ── Save locally then upload to S3 ─────────────────────────
        timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
        run_label = args.run_name or timestamp
        local_run_dir = Path(tmpdir) / "run_output"
        run = save_run(run, output_dir=local_run_dir)

        s3_run_prefix = f"{eval_prefix}/runs/{run_label}"
        log.info("Uploading results to s3://%s/%s", bucket, s3_run_prefix)
        _upload_directory(local_run_dir, bucket, s3_run_prefix)

        # Also save config snapshot
        config_snapshot = {
            "query_set": args.query_set,
            "top_k": args.top_k,
            "keep_k": args.keep_k,
            "token_budget": args.token_budget,
            "run_generation": args.run_generation,
            "use_llm_judge": args.use_llm_judge,
            "judge_model": args.judge_model if args.use_llm_judge else None,
            "score_ids": args.score_ids,
            "timestamp": timestamp,
        }
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=f"{s3_run_prefix}/config.json",
            Body=json.dumps(config_snapshot, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    # ── Print summary ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("REMOTE EVAL RUN")
    print("=" * 72)
    print(f"run_id:        {run.meta.run_id}")
    print(f"query_set:     {args.query_set}")
    print(f"num_queries:   {run.aggregates.overall.num_queries}")
    print(f"mrr:           {run.aggregates.overall.mrr:.4f}")
    print(f"map:           {run.aggregates.overall.map:.4f}")
    for k in sorted(run.aggregates.overall.recall_at_k):
        print(f"recall@{k}:     {run.aggregates.overall.recall_at_k[k]:.4f}")
    print(f"results:       s3://{bucket}/{s3_run_prefix}/")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
