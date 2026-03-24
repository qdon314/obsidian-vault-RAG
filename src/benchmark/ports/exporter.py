"""Port protocol for benchmark dataset export."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from benchmark.domain.models import BenchmarkDataset


@runtime_checkable
class BenchmarkExporter(Protocol):
    """Export a benchmark dataset to an external format.

    The primary implementation (``EvalQueryExporter``) emits
    ``EvalQuery``-compatible JSONL for the eval harness.  Other
    implementations may write the full benchmark JSONL or push
    to a database.
    """

    def export(self, dataset: BenchmarkDataset) -> None: ...
