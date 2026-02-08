#!/usr/bin/env python3
"""CLI for computing eval verdict artifacts and optional release-gate exit code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.eval.models import EvalAggregates, EvalResult, EvalRun, EvalRunMeta
from rag.eval.verdict import Decision, compute_verdict, render_verdict_json, render_verdict_markdown
from rag.eval.verdict_thresholds import load_verdict_thresholds


def resolve_run_dir(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Run directory not found: {candidate}")


def load_eval_run(run_dir: Path) -> EvalRun:
    run_path = resolve_run_dir(run_dir)
    metrics_path = run_path / "metrics.json"
    results_path = run_path / "results.jsonl"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics.json in {run_path}")
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.jsonl in {run_path}")

    # Reconstruct EvalRun from persisted harness artifacts.
    metrics_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    meta = EvalRunMeta.from_dict(metrics_data.get("meta", {}))
    aggregates = EvalAggregates.from_flat_dict(metrics_data)

    results: list[EvalResult] = []
    with results_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            results.append(EvalResult.from_results_dict(json.loads(line)))

    return EvalRun(meta=meta, results=tuple(results), aggregates=aggregates, artifacts=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate eval verdict and release gate decision")
    parser.add_argument(
        "--current", 
        type=Path, 
        required=False,
        default=Path("eval/runs/latest"), 
        help="Current eval run directory")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline eval run directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval/verdicts"),
        help="Output directory for verdict.md and verdict.json",
    )
    parser.add_argument(
        "--settings",
        type=Path,
        default=Path("settings.toml"),
        help="Settings TOML path containing [eval.verdict] thresholds",
    )
    parser.add_argument(
        "--fail-on-block",
        action="store_true",
        help="Exit with code 1 when verdict decision is BLOCK",
    )
    args = parser.parse_args()

    # Thresholds are loaded from [eval.verdict], with safe defaults if absent.
    thresholds = load_verdict_thresholds(args.settings)
    current_run = load_eval_run(args.current)
    baseline_run = load_eval_run(args.baseline) if args.baseline else None

    verdict = compute_verdict(current=current_run, baseline=baseline_run, thresholds=thresholds)
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    verdict_md = output_dir / "verdict.md"
    verdict_json = output_dir / "verdict.json"
    # Emit both human-friendly and machine-readable artifacts for CI and local review.
    verdict_md.write_text(render_verdict_markdown(verdict), encoding="utf-8")
    verdict_json.write_text(render_verdict_json(verdict), encoding="utf-8")

    print(f"Decision: {verdict.decision.value.upper()}")
    print(f"Summary: {verdict.summary}")
    print(f"Markdown: {verdict_md}")
    print(f"JSON: {verdict_json}")

    if args.fail_on_block and verdict.decision is Decision.BLOCK:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
