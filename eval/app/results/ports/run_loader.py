"""
Protocol interfaces for loading and managing evaluation runs.

These protocols define the abstract interfaces that adapters must implement.
Following PEP 544 structural subtyping pattern from the main RAG system.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from eval.app.results.domain.models import LoadedRun, RunSummary


class RunLoader(Protocol):
    """Protocol for loading evaluation runs from storage.

    Implementations may load from filesystem, database, or remote APIs.
    """

    def discover_runs(self) -> list[RunSummary]:
        """Discover all available runs in the configured location.

        Returns:
            List of RunSummary objects for each discovered run,
            sorted by timestamp descending (most recent first).
        """
        ...

    def load_run(self, run_id: str) -> LoadedRun:
        """Load complete run data by ID.

        Args:
            run_id: The unique identifier for the run.

        Returns:
            Fully loaded run with all results and metrics.

        Raises:
            FileNotFoundError: If run directory doesn't exist.
            ValueError: If run data is malformed.
        """
        ...

    def load_summary(self, run_id: str) -> RunSummary:
        """Load only the summary/metadata for a run.

        This is faster than load_run when only metadata is needed.

        Args:
            run_id: The unique identifier for the run.

        Returns:
            RunSummary with metadata but no detailed results.
        """
        ...


class RunRepository(Protocol):
    """Protocol for managing a collection of evaluation runs.

    Extends basic loading with management operations like
    adding custom paths and caching.
    """

    def list_runs(self, limit: int | None = None) -> list[RunSummary]:
        """List all available runs, optionally limited.

        Args:
            limit: Maximum number of runs to return. None for all.

        Returns:
            List of RunSummary objects sorted by timestamp descending.
        """
        ...

    def get_run(self, run_id: str) -> LoadedRun:
        """Get a run by ID, loading if necessary.

        May return cached data if available.

        Args:
            run_id: The unique identifier for the run.

        Returns:
            Fully loaded run.
        """
        ...

    def add_run_path(self, path: Path) -> RunSummary:
        """Add an external run directory to the repository.

        Useful for loading runs from non-standard locations.

        Args:
            path: Path to a run directory containing metrics.json.

        Returns:
            RunSummary for the added run.

        Raises:
            FileNotFoundError: If path doesn't exist or lacks metrics.json.
        """
        ...

    def refresh(self) -> None:
        """Refresh the list of available runs.

        Clears caches and re-discovers runs from all sources.
        """
        ...
