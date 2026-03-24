"""Port conformance test for BenchmarkExporter protocol."""

from __future__ import annotations

from benchmark.domain.models import BenchmarkDataset
from benchmark.ports.exporter import BenchmarkExporter


class _StubExporter:
    """Minimal stub that structurally satisfies BenchmarkExporter."""

    def __init__(self) -> None:
        self.last_dataset: BenchmarkDataset | None = None

    def export(self, dataset: BenchmarkDataset) -> None:
        self.last_dataset = dataset


class TestBenchmarkExporterProtocol:
    def test_stub_is_instance(self) -> None:
        exporter = _StubExporter()
        assert isinstance(exporter, BenchmarkExporter)

    def test_stub_receives_dataset(self) -> None:
        exporter = _StubExporter()
        ds = BenchmarkDataset(
            schema_version="1.0",
            records=(),
            corpus_snapshot_id="abc123",
            created_at="2026-03-24T12:00:00Z",
        )
        exporter.export(ds)
        assert exporter.last_dataset is ds
