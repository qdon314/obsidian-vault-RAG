"""Orchestrator: poll loop for monitoring distributed ingestion progress.

The poll loop queries the Postgres job store for task counts until all tasks
have reached a terminal state (SUCCEEDED or FAILED) or a timeout is reached.
"""
from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rag.domain.ingestion import TaskStatus

if TYPE_CHECKING:
    from rag.ports import IngestJobStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PollResult:
    """Outcome of polling for task completion."""

    succeeded: int
    failed: int
    pending: int
    running: int
    retryable: int
    timed_out: bool
    expected_total: int

    @property
    def total(self) -> int:
        return self.succeeded + self.failed + self.pending + self.running + self.retryable

    @property
    def all_succeeded(self) -> bool:
        return (
            not self.timed_out
            and self.succeeded == self.expected_total
            and self.failed == 0
            and self.pending == 0
            and self.running == 0
            and self.retryable == 0
        )


def poll_until_complete(
    *,
    job_id: uuid.UUID,
    job_store: IngestJobStore,
    total_tasks: int,
    poll_interval_s: float = 30.0,
    timeout_s: float = 7200.0,
) -> PollResult:
    """Poll task counts until all tasks reach a terminal state or timeout.

    Terminal states: SUCCEEDED, FAILED (past max retries).
    Non-terminal states: PENDING, RUNNING, RETRYABLE.
    """
    deadline = time.monotonic() + timeout_s

    while True:
        counts = job_store.get_task_counts(job_id)
        succeeded = counts.get(TaskStatus.SUCCEEDED, 0)
        failed = counts.get(TaskStatus.FAILED, 0)
        pending = counts.get(TaskStatus.PENDING, 0)
        running = counts.get(TaskStatus.RUNNING, 0)
        retryable = counts.get(TaskStatus.RETRYABLE, 0)

        logger.info(
            "Poll: %d/%d succeeded, %d failed, %d pending, %d running, %d retryable",
            succeeded,
            total_tasks,
            failed,
            pending,
            running,
            retryable,
        )

        terminal = succeeded + failed
        # Done only when no tasks are in-flight and every expected task is terminal.
        in_flight = pending + running + retryable
        if in_flight == 0 and terminal == total_tasks:
            return PollResult(
                succeeded=succeeded,
                failed=failed,
                pending=0,
                running=0,
                retryable=0,
                timed_out=False,
                expected_total=total_tasks,
            )
        if in_flight == 0 and terminal != total_tasks:
            logger.warning(
                "No in-flight tasks but terminal count mismatch: terminal=%d expected=%d",
                terminal,
                total_tasks,
            )

        if time.monotonic() >= deadline:
            logger.warning("Timeout reached with %d tasks still in-flight", in_flight)
            return PollResult(
                succeeded=succeeded,
                failed=failed,
                pending=pending,
                running=running,
                retryable=retryable,
                timed_out=True,
                expected_total=total_tasks,
            )

        time.sleep(poll_interval_s)
