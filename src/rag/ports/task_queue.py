"""Port for distributing ingestion tasks to workers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class TaskQueue(Protocol):
    """Distributes ingestion task messages to workers.

    Each message is a small JSON dict: ``{"job_id": "...", "corpus_id": "...", "doc_id": "..."}``.
    """

    def send(self, message: dict[str, str]) -> None:
        """Enqueue a single task message."""
        ...

    def send_batch(self, messages: Sequence[dict[str, str]]) -> None:
        """Enqueue a batch of task messages (max 10 per SQS batch)."""
        ...

    def receive(self, *, max_messages: int = 1, wait_seconds: int = 20) -> list[dict[str, Any]]:
        """Long-poll for messages.

        Returns a list of dicts, each containing:
        - ``"body"``: the parsed message body (dict)
        - ``"receipt_handle"``: opaque handle for ack/nack
        """
        ...

    def ack(self, receipt_handle: str) -> None:
        """Delete a message after successful processing."""
        ...

    def nack(self, receipt_handle: str, *, visibility_timeout: int = 0) -> None:
        """Return a message to the queue for retry."""
        ...
