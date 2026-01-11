from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rag.eval.schema import EvalQuery

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class JsonlEvalStore:
    """
    JSONL-based persistence for evaluation queries.

    Each line is a JSON object representing one EvalQuery.
    """

    def load_queries(self, path: Path) -> list[EvalQuery]:
        """Load all queries from the JSONL file."""
        queries: list[EvalQuery] = []

        if not path.exists():
            logger.info(f"Query file not found, starting fresh: {path}")
            return queries

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    queries.append(EvalQuery.from_dict(data))
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.warning(f"Error loading query at line {line_num}: {e}")
                    continue

        logger.info(f"Loaded {len(queries)} queries from {path}")
        return queries

    def save_queries(self, queries: list[EvalQuery], path: Path) -> None:
        """Save all queries to the JSONL file (overwrites existing)."""
        path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write using temp file
        tmp_path = path.with_suffix(".jsonl.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            for query in queries:
                f.write(json.dumps(query.to_dict()) + "\n")

        tmp_path.replace(path)
        logger.info(f"Saved {len(queries)} queries to {path}")

    def append_query(self, query: EvalQuery, path: Path) -> None:
        """Append a single query to the file."""
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(query.to_dict()) + "\n")

        logger.debug(f"Appended query {query.qid} to {path}")

    def count_queries(self, path: Path) -> int:
        """Count queries in the file without loading all."""
        if not path.exists():
            return 0

        count = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count

    def get_existing_qids(self, path: Path) -> set[str]:
        """Get all existing query IDs in the file."""
        queries = self.load_queries(path)
        return {q.qid for q in queries}
