"""CLI entry point for the benchmark generation pipeline.

Usage:
    ./scripts/py -m benchmark.scripts.run_benchmark_gen \\
        --run-id "run_20260324" \\
        --output-dir benchmark_runs/ \\
        --model gpt-4o \\
        [--resume-from candidate_generation] \\
        [--query-classes citation_lookup,unanswerable] \\
        [--skip-hard-negatives] \\
        [--export-path eval/datasets/benchmark_v1.jsonl] \\
        [--valid-as-of 2026-03-24] \\
        [--synthesize-gold-answers] \\
        [--contamination-model gpt-4o-2025-01-01]

This script is the composition root for M4+M5 adapters. It wires together:
- LLMClient: OpenAI-based adapter (from OPENAI_API_KEY env var)
- QueryGenerators: {CITATION_LOOKUP: TemplateQueryGenerator,
                    UNANSWERABLE: UnanswerableGenerator}
- QueryValidator: LLMValidator (wrapping DeterministicValidator)
- Retriever: optional, from RAG container (requires Qdrant running)
- Exporter: EvalQueryExporter writing to --export-path
- GoldAnswerSynthesizer: LLMGoldAnswerSynthesizer (when --synthesize-gold-answers)
- ContaminationProbe: contamination_probe (when --contamination-model is set)
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
            "Stage to resume from: source_spans, unit_extraction, unit_classification, "
            "evidence_tiers, candidate_generation, query_validation, "
            "hard_negative_mining, gold_answer_synthesis, contamination_probe, export"
        ),
    )
    parser.add_argument(
        "--query-classes",
        default="citation_lookup",
        help=("Comma-separated list of query classes to generate (default: citation_lookup)"),
    )
    parser.add_argument(
        "--skip-hard-negatives",
        action="store_true",
        help="Skip hard negative mining (no Qdrant required)",
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
    # M5 flags.
    parser.add_argument(
        "--synthesize-gold-answers",
        action="store_true",
        help="Enable gold answer synthesis (requires LLM)",
    )
    parser.add_argument(
        "--contamination-model",
        default=None,
        metavar="MODEL_ID",
        help=(
            "Model ID to use for contamination probe "
            "(e.g. gpt-4o-2025-01-01). "
            "Requires --synthesize-gold-answers or a run with gold_answer_synthesis complete."
        ),
    )
    parser.add_argument(
        "--warn-only-flags",
        default="",
        metavar="FLAGS",
        help=(
            "Comma-separated validation flags that are recorded but do not block a query "
            "(e.g. missing_snapshot_id,too_many_evidence_spans). "
            "Useful when resuming older runs that predate certain checks."
        ),
    )
    parser.add_argument(
        "--rpm-limit",
        type=float,
        default=0.0,
        metavar="N",
        help=(
            "Requests-per-minute limit for LLM calls. "
            "Adds a minimum delay between calls to avoid rate limit errors "
            "(e.g. 60 → 1s delay, 120 → 0.5s delay). Default 0 = no enforced delay."
        ),
    )
    parser.add_argument(
        "--ecfr-xml",
        default=None,
        metavar="PATH",
        help=(
            "Path to eCFR XML file for Stage 0 source span extraction. "
            "Required unless resuming from unit_extraction or later."
        ),
    )
    parser.add_argument(
        "--doc-id",
        default="ecfr",
        help="Document ID to assign to source spans (default: ecfr)",
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
        logger.error("OPENAI_API_KEY not set. Set it in your environment or .env file.")
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

        min_delay_s = (60.0 / args.rpm_limit) if args.rpm_limit > 0 else 0.0
        llm_client = OpenAILLMClient(api_key=api_key, min_delay_s=min_delay_s)
    except ImportError:
        logger.error(
            "OpenAI LLM client not available. Install with: ./scripts/pip install -e '.[openai]'"
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
        query_generators[QueryClass.UNANSWERABLE] = UnanswerableGenerator(llm_client, stage_config)

    # -- Wire validator ---------------------------------------------------
    warn_only: frozenset[str] = frozenset(
        f.strip() for f in args.warn_only_flags.split(",") if f.strip()
    )
    deterministic = DeterministicValidator(warn_only_flags=warn_only or None)
    validator = LLMValidator(
        llm_client, stage_config, deterministic=deterministic, warn_only_flags=warn_only or None
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
            logger.info("Retriever wired for hard_negative_mining")
        except Exception as exc:
            logger.warning(
                "Could not initialize retriever (%s). "
                "Skipping hard_negative_mining. Pass --skip-hard-negatives to suppress.",
                exc,
            )

    # -- Wire exporter (optional) ----------------------------------------
    exporter = None
    if args.export_path:
        from benchmark.adapters.export.eval_query_exporter import EvalQueryExporter

        exporter = EvalQueryExporter(Path(args.export_path))
        logger.info("EvalQuery exporter writing to %s", args.export_path)

    # -- Wire M5 gold answer synthesizer (optional) -----------------------
    gold_answer_synthesizer = None
    if args.synthesize_gold_answers:
        from benchmark.adapters.generation.llm_gold_answer_synthesizer import (  # type: ignore[import]
            LLMGoldAnswerSynthesizer,
        )

        gold_answer_synthesizer = LLMGoldAnswerSynthesizer(llm_client, stage_config)
        logger.info("Gold answer synthesizer wired for gold_answer_synthesis")

    # -- Validate M5 contamination probe args ------------------------------
    contamination_model_id = args.contamination_model
    if contamination_model_id and not args.synthesize_gold_answers:
        # Warn if no synthesizer but allow: operator may be resuming from
        # a run that already has Stage 6 complete.
        logger.warning(
            "contamination_probe requires gold answers. Pass --synthesize-gold-answers "
            "or resume from a run that has gold_answer_synthesis complete."
        )

    # -- Build unit extractor / evidence builder / classifier --------------
    # These are not changed by M4; they require the eCFR corpus to be indexed.
    try:
        from benchmark.adapters.evidence.evidence_builder import (  # type: ignore[import-untyped]
            DefaultEvidenceBuilder,
        )
        from benchmark.adapters.extraction.ecfr_extractor import (  # type: ignore[import-untyped]
            ECFRUnitExtractor,
        )
        from benchmark.adapters.extraction.llm_classifier import (  # type: ignore[import-untyped]
            LLMUnitClassifier,
        )
        from benchmark.stages.source_spans import build_source_spans

        unit_extractor = ECFRUnitExtractor()
        evidence_builder = DefaultEvidenceBuilder()
        llm_classifier = LLMUnitClassifier(llm_client, stage_config).classify

        _ecfr_xml = args.ecfr_xml
        _doc_id = args.doc_id
        _effective_date = args.valid_as_of

        def _make_corpus_spans() -> list:
            if _ecfr_xml is None:
                raise RuntimeError(
                    "source_spans stage requires --ecfr-xml <path>. "
                    "Pass --resume-from unit_extraction (or later) to skip this stage."
                )
            from benchmark.domain.snapshot import compute_snapshot_id
            from rag.adapters.ingestion.regulatory.ecfr_parser import (
                parse_ecfr_xml as _parse_xml,
            )
            from rag.app.container import _get_store_chunks, build_container

            _sections = _parse_xml(Path(_ecfr_xml).read_text())
            _chunks: list = []
            _snapshot_id = ""
            try:
                _cont = build_container()
                _chunks = list(_get_store_chunks(_cont.store))
                _snapshot_id = compute_snapshot_id(_chunks)
            except Exception as _exc:
                logger.warning(
                    "Could not load chunk index (%s). Span overlap resolution will be empty.",
                    _exc,
                )
            return build_source_spans(
                sections=_sections,
                doc_id=_doc_id,
                chunk_index=_chunks,
                corpus_snapshot_id=_snapshot_id,
                effective_date=_effective_date,
            )

        corpus_spans_builder = _make_corpus_spans
    except ImportError as exc:
        logger.error(
            "Could not import pipeline adapters: %s. Ensure the package is installed correctly.",
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
        corpus_spans_builder=corpus_spans_builder,  # type: ignore[arg-type]
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
        # M5 additions.
        gold_answer_synthesizer=gold_answer_synthesizer,  # type: ignore[arg-type]
        llm_client=llm_client,
        contamination_model_id=contamination_model_id,
    )

    result = runner.run()

    logger.info(
        "Pipeline complete: run_id=%s candidates=%d validated=%d "
        "flagged=%d hard_negatives=%d exported=%d contaminated=%d",
        result.run_id,
        result.total_candidates,
        result.total_validated,
        result.total_flagged,
        result.total_hard_negatives,
        result.total_exported,
        result.total_contaminated,
    )
    logger.info("Output: %s", result.output_dir)


if __name__ == "__main__":
    main()
