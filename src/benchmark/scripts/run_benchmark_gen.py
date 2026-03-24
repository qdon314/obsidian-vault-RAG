"""CLI entry point for the benchmark generation pipeline.

Usage:
    ./scripts/py -m benchmark.scripts.run_benchmark_gen \\
        --run-id "run_20260324" \\
        --output-dir benchmark_runs/ \\
        --model gpt-4o \\
        [--resume-from stage_3] \\
        [--query-classes citation_lookup,unanswerable] \\
        [--skip-hard-negatives] \\
        [--export-path eval/datasets/benchmark_v1.jsonl] \\
        [--valid-as-of 2026-03-24]

This script is the composition root for M4 adapters. It wires together:
- LLMClient: OpenAI-based adapter (from OPENAI_API_KEY env var)
- QueryGenerators: {CITATION_LOOKUP: TemplateQueryGenerator,
                    UNANSWERABLE: UnanswerableGenerator}
- QueryValidator: LLMValidator (wrapping DeterministicValidator)
- Retriever: optional, from RAG container (requires Qdrant running)
- Exporter: EvalQueryExporter writing to --export-path
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the NRC benchmark generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Unique identifier for this pipeline run (e.g. run_20260324)",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_runs/",
        help="Directory to write checkpoint files (default: benchmark_runs/)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI model name for LLM stages (default: gpt-4o)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Stage to resume from: stage_0, stage_1a, stage_1b, stage_2, "
            "stage_3, stage_5a, stage_5b, export"
        ),
    )
    parser.add_argument(
        "--query-classes",
        default="citation_lookup",
        help=(
            "Comma-separated list of query classes to generate "
            "(default: citation_lookup)"
        ),
    )
    parser.add_argument(
        "--skip-hard-negatives",
        action="store_true",
        help="Skip Stage 5b hard negative mining (no Qdrant required)",
    )
    parser.add_argument(
        "--export-path",
        default=None,
        help="Path to write EvalQuery JSONL export (e.g. eval/datasets/benchmark_v1.jsonl)",
    )
    parser.add_argument(
        "--valid-as-of",
        default="",
        help="ISO date for benchmark record validity (e.g. 2026-03-24)",
    )
    return parser


def main() -> None:
    """Parse CLI args, wire adapters, and run the pipeline."""
    # Lazy imports here so --help works without all deps installed.
    from benchmark.adapters.generation.template_generator import TemplateQueryGenerator
    from benchmark.adapters.generation.unanswerable_generator import UnanswerableGenerator
    from benchmark.adapters.validation.deterministic_validator import DeterministicValidator
    from benchmark.adapters.validation.llm_validator import LLMValidator
    from benchmark.domain.enums import QueryClass
    from benchmark.domain.models import StageConfig
    from benchmark.pipeline.runner import PipelineConfig, PipelineRunner

    parser = _build_arg_parser()
    args = parser.parse_args()

    # -- Validate OPENAI_API_KEY -------------------------------------------
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error(
            "OPENAI_API_KEY not set. Set it in your environment or .env file."
        )
        sys.exit(1)

    # -- Parse query classes -----------------------------------------------
    class_map = {qc.value: qc for qc in QueryClass}
    query_classes: list[QueryClass] = []
    for raw in args.query_classes.split(","):
        cls = class_map.get(raw.strip())
        if cls is None:
            logger.error("Unknown query class %r. Valid: %s", raw, list(class_map))
            sys.exit(1)
        query_classes.append(cls)

    # -- Wire LLM client ---------------------------------------------------
    try:
        from benchmark.adapters.llm.openai_client import OpenAILLMClient  # type: ignore[import]
        llm_client = OpenAILLMClient(api_key=api_key)
    except ImportError:
        logger.error(
            "OpenAI LLM client not available. Install with: "
            "./scripts/pip install -e '.[openai]'"
        )
        sys.exit(1)

    stage_config = StageConfig(model=args.model)

    # -- Wire query generators --------------------------------------------
    query_generators: dict[QueryClass, object] = {}
    if QueryClass.CITATION_LOOKUP in query_classes:
        query_generators[QueryClass.CITATION_LOOKUP] = TemplateQueryGenerator(
            llm_client, stage_config
        )
    if QueryClass.UNANSWERABLE in query_classes:
        query_generators[QueryClass.UNANSWERABLE] = UnanswerableGenerator(
            llm_client, stage_config
        )

    # -- Wire validator ---------------------------------------------------
    deterministic = DeterministicValidator()
    validator = LLMValidator(
        llm_client, stage_config, deterministic=deterministic
    )

    # -- Wire retriever (optional) ----------------------------------------
    retriever = None
    retriever_config: dict[str, object] = {}
    if not args.skip_hard_negatives:
        try:
            from rag.app.container import build_container  # type: ignore[import]
            container = build_container()
            retriever = container.retriever
            retriever_config = {"model": args.model, "top_k": 20}
            logger.info("Retriever wired for Stage 5b hard negative mining")
        except Exception as exc:
            logger.warning(
                "Could not initialize retriever (%s). "
                "Skipping Stage 5b. Pass --skip-hard-negatives to suppress.",
                exc,
            )

    # -- Wire exporter (optional) ----------------------------------------
    exporter = None
    if args.export_path:
        from benchmark.adapters.export.eval_query_exporter import EvalQueryExporter
        exporter = EvalQueryExporter(Path(args.export_path))
        logger.info("EvalQuery exporter writing to %s", args.export_path)

    # -- Build unit extractor / evidence builder / classifier --------------
    # These are not changed by M4; they require the eCFR corpus to be indexed.
    try:
        from benchmark.adapters.evidence.evidence_builder import (
            DefaultEvidenceBuilder,  # type: ignore[import]
        )
        from benchmark.adapters.extraction.ecfr_extractor import (
            ECFRUnitExtractor,  # type: ignore[import]
        )
        from benchmark.adapters.extraction.llm_classifier import (
            LLMUnitClassifier,  # type: ignore[import]
        )

        from benchmark.stages.stage_0_source_view import build_corpus_spans  # type: ignore[import]
        unit_extractor = ECFRUnitExtractor()
        evidence_builder = DefaultEvidenceBuilder()
        llm_classifier = LLMUnitClassifier(llm_client, stage_config).classify
        corpus_spans_builder = build_corpus_spans
    except ImportError as exc:
        logger.error(
            "Could not import pipeline adapters: %s. "
            "Ensure the package is installed correctly.",
            exc,
        )
        sys.exit(1)

    # -- Assemble and run -------------------------------------------------
    config = PipelineConfig(
        run_id=args.run_id,
        output_dir=args.output_dir,
        resume_from=args.resume_from,
    )

    runner = PipelineRunner(
        config,
        corpus_spans_builder=corpus_spans_builder,
        unit_extractor=unit_extractor,
        llm_classifier=llm_classifier,
        evidence_builder=evidence_builder,
        query_generators=query_generators,  # type: ignore[arg-type]
        query_validator=validator,
        retriever=retriever,
        retriever_config=retriever_config,  # type: ignore[arg-type]
        exporter=exporter,
        query_classes=tuple(query_classes),
        valid_as_of=args.valid_as_of,
    )

    result = runner.run()

    logger.info(
        "Pipeline complete: run_id=%s candidates=%d validated=%d "
        "flagged=%d hard_negatives=%d exported=%d",
        result.run_id,
        result.total_candidates,
        result.total_validated,
        result.total_flagged,
        result.total_hard_negatives,
        result.total_exported,
    )
    logger.info("Output: %s", result.output_dir)


if __name__ == "__main__":
    main()
