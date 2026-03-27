"""EvalQuery-compatible JSONL exporter for benchmark datasets.

Maps ``BenchmarkDataset`` → one ``EvalQuery`` JSON object per line, using
the field-mapping table from the design doc.  This is the bridge that lets
the existing eval harness (``rag/eval/``) consume benchmark output.
"""

from __future__ import annotations

import logging
from pathlib import Path

from benchmark.domain.enums import QueryClass
from benchmark.domain.models import (
    BenchmarkDataset,
    BenchmarkRecord,
    EvidenceEntry,
)
from rag.eval.schema import EvalQuery, QueryType

logger = logging.getLogger(__name__)

# -- Enum translation ---------------------------------------------------------

_QUERY_CLASS_TO_QUERY_TYPE: dict[QueryClass, QueryType] = {
    QueryClass.CITATION_LOOKUP: QueryType.FACTUAL,
    QueryClass.NARROW_FACTUAL: QueryType.FACTUAL,
    QueryClass.RULE_EXPLANATION: QueryType.PROCEDURAL,
    QueryClass.CROSS_REFERENCE: QueryType.MULTI_HOP,
    QueryClass.SCENARIO_APPLICATION: QueryType.SCENARIO,
    QueryClass.UNANSWERABLE: QueryType.FACTUAL,  # type is orthogonal to answerability
    QueryClass.ROBUSTNESS_VARIANT: QueryType.FACTUAL,
}


class EvalQueryExporter:
    """Export BenchmarkDataset as EvalQuery-compatible JSONL.

    Each line of the output file is a JSON object matching the ``EvalQuery``
    schema.  The file is suitable for use as a ``--queries`` input to
    ``eval/scripts/run_eval.py``.
    """

    def __init__(self, output_path: Path) -> None:
        self._output_path = output_path

    def export(self, dataset: BenchmarkDataset) -> None:
        """Write one EvalQuery JSON object per line to output_path."""
        self._output_path.parent.mkdir(parents=True, exist_ok=True)

        written = 0
        with self._output_path.open("w", encoding="utf-8") as fh:
            for record in dataset.records:
                eval_query = _to_eval_query(record)
                fh.write(eval_query.to_json() + "\n")
                written += 1

        logger.info(
            "EvalQueryExporter: wrote %d records to %s",
            written,
            self._output_path,
        )


# -- Conversion ---------------------------------------------------------------


def _to_eval_query(record: BenchmarkRecord) -> EvalQuery:
    """Convert a BenchmarkRecord to an EvalQuery."""
    critical_ids = _collect_chunk_ids(record.critical_evidence)
    supporting_ids = _collect_chunk_ids(record.supporting_evidence)
    contextual_ids = _collect_chunk_ids(record.contextual_evidence)

    # For unanswerable records, evidence is intentionally empty.
    relevant_chunk_ids = critical_ids | supporting_ids | contextual_ids

    query_type = _QUERY_CLASS_TO_QUERY_TYPE.get(record.query_class, QueryType.FACTUAL)

    return EvalQuery(
        qid=record.qid,
        query=record.query,
        relevant_chunk_ids=relevant_chunk_ids,
        relevant_citations=set(record.source_citations),
        critical_chunk_ids=critical_ids,
        supporting_chunk_ids=supporting_ids,
        context_chunk_ids=contextual_ids,
        query_type=query_type,
        is_unanswerable=record.is_unanswerable,
        unanswerable_reason=record.unanswerable_reason,
        expected_answer=record.gold_answer,
        expected_answer_alternatives=list(record.acceptable_answer_variants),
        metadata={
            "corpus_snapshot_id": record.corpus_snapshot_id,
            "valid_as_of": record.valid_as_of,
            **record.metadata,
        },
    )


def _collect_chunk_ids(entries: tuple[EvidenceEntry, ...]) -> set[str]:
    """Flatten chunk_ids from a tuple of EvidenceEntry objects."""
    ids: set[str] = set()
    for entry in entries:
        ids.update(entry.chunk_ids)
    return ids
