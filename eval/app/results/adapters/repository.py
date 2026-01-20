"""
In-memory repository for managing evaluation runs.

Provides caching and management of runs from multiple sources
(auto-discovered and manually added paths).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from eval.app.results.adapters.filesystem_loader import FilesystemRunLoader
from eval.app.results.domain.models import LoadedRun, RunSummary

logger = logging.getLogger(__name__)


@dataclass
class InMemoryRunRepository:
    """Repository that manages runs from filesystem with caching.

    Implements the RunRepository protocol. Combines auto-discovery
    from the standard runs directory with support for manually
    added external paths.
    """

    loader: FilesystemRunLoader
    additional_paths: list[Path] = field(default_factory=list)
    _summaries_cache: list[RunSummary] | None = field(default=None)

    def list_runs(self, limit: int | None = None) -> list[RunSummary]:
        """List all available runs from all sources."""
        if self._summaries_cache is not None:
            summaries = self._summaries_cache
        else:
            summaries = self._discover_all_runs()
            self._summaries_cache = summaries

        if limit is not None:
            return summaries[:limit]
        return summaries

    def get_run(self, run_id: str) -> LoadedRun:
        """Get a run by ID, checking all sources."""
        # First try the main loader
        try:
            return self.loader.load_run(run_id)
        except FileNotFoundError:
            pass

        # Try additional paths
        for path in self.additional_paths:
            if path.name == run_id:
                return self._load_from_path(path)

        raise FileNotFoundError(f"Run not found: {run_id}")

    def add_run_path(self, path: Path) -> RunSummary:
        """Add an external run directory."""
        path = path.resolve()

        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        metrics_file = path / "metrics.json"
        if not metrics_file.exists():
            raise FileNotFoundError(f"No metrics.json in {path}")

        # Add to additional paths if not already there
        if path not in self.additional_paths:
            self.additional_paths.append(path)

        # Invalidate cache
        self._summaries_cache = None

        # Load and return summary
        temp_loader = FilesystemRunLoader(runs_dir=path.parent)
        return temp_loader._load_summary_from_dir(path)

    def refresh(self) -> None:
        """Refresh all caches."""
        self._summaries_cache = None
        self.loader.clear_cache()
        logger.info("Repository cache cleared")

    def _discover_all_runs(self) -> list[RunSummary]:
        """Discover runs from all sources."""
        # Get runs from main directory
        summaries = self.loader.discover_runs()
        seen_ids = {s.run_id for s in summaries}

        # Add runs from additional paths
        for path in self.additional_paths:
            if path.name in seen_ids:
                continue

            try:
                temp_loader = FilesystemRunLoader(runs_dir=path.parent)
                summary = temp_loader._load_summary_from_dir(path)
                summaries.append(summary)
                seen_ids.add(summary.run_id)
            except Exception as e:
                logger.warning(f"Failed to load run from {path}: {e}")
                continue

        # Sort by timestamp descending
        summaries.sort(key=lambda s: s.timestamp, reverse=True)
        return summaries

    def _load_from_path(self, path: Path) -> LoadedRun:
        """Load a run from a specific path."""
        temp_loader = FilesystemRunLoader(runs_dir=path.parent)
        return temp_loader._load_run_from_dir(path)
