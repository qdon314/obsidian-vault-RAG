"""Tests for the orchestrator poll loop."""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

from rag.app.ingestion.orchestrator import poll_until_complete
from rag.domain.ingestion import TaskStatus


def _mock_job_store(counts_sequence: list[dict[TaskStatus, int]]) -> MagicMock:
    """Return a mock job store that returns successive task count dicts."""
    store = MagicMock()
    store.get_task_counts = MagicMock(side_effect=counts_sequence)
    return store


def test_poll_completes_when_all_succeeded():
    job_id = uuid.uuid4()
    store = _mock_job_store([
        {TaskStatus.PENDING: 5, TaskStatus.RUNNING: 5},
        {TaskStatus.SUCCEEDED: 8, TaskStatus.RUNNING: 2},
        {TaskStatus.SUCCEEDED: 10},
    ])

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=10,
    )

    assert result.all_succeeded is True
    assert result.succeeded == 10
    assert result.failed == 0
    assert result.expected_total == 10


def test_poll_detects_failures():
    job_id = uuid.uuid4()
    store = _mock_job_store([
        {TaskStatus.SUCCEEDED: 8, TaskStatus.FAILED: 2},
    ])

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=10,
    )

    assert result.all_succeeded is False
    assert result.succeeded == 8
    assert result.failed == 2
    assert result.expected_total == 10


def test_poll_times_out():
    job_id = uuid.uuid4()
    # Always returns tasks in progress — will never complete
    store = _mock_job_store(
        [{TaskStatus.RUNNING: 10}] * 100,
    )

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=0,  # immediate timeout
    )

    assert result.timed_out is True
    assert result.expected_total == 10


def test_poll_times_out_when_terminal_count_mismatch():
    job_id = uuid.uuid4()
    # No tasks in flight, but not all expected tasks are represented as terminal.
    store = _mock_job_store([{TaskStatus.SUCCEEDED: 9}] * 5)

    result = poll_until_complete(
        job_id=job_id,
        job_store=store,
        total_tasks=10,
        poll_interval_s=0,
        timeout_s=0,
    )

    assert result.timed_out is True
    assert result.succeeded == 9
    assert result.failed == 0
    assert result.all_succeeded is False
