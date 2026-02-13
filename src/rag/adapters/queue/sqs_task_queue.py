"""SQS-backed task queue for distributing ingestion work."""
from __future__ import annotations

import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SQSTaskQueue:
    """Distributes ingestion task messages via AWS SQS."""

    queue_url: str
    _sqs: Any = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        import boto3  # type: ignore[import-untyped]
        object.__setattr__(self, "_sqs", boto3.client("sqs"))

    @classmethod
    def _for_test(cls, *, queue_url: str, sqs_client: Any) -> SQSTaskQueue:
        """Create an instance with an injected SQS client (for testing)."""
        obj = object.__new__(cls)
        object.__setattr__(obj, "queue_url", queue_url)
        object.__setattr__(obj, "_sqs", sqs_client)
        return obj

    def send(self, message: dict[str, str]) -> None:
        self._sqs.send_message(
            QueueUrl=self.queue_url,
            MessageBody=json.dumps(message),
        )

    def send_batch(self, messages: Sequence[dict[str, str]]) -> None:
        entries = [
            {"Id": str(uuid.uuid4()), "MessageBody": json.dumps(m)}
            for m in messages
        ]
        # SQS batch limit is 10
        for i in range(0, len(entries), 10):
            batch = entries[i : i + 10]
            resp = self._sqs.send_message_batch(
                QueueUrl=self.queue_url, Entries=batch
            )
            failed = resp.get("Failed", [])
            if failed:
                raise RuntimeError(f"Failed to enqueue {len(failed)} messages: {failed}")

    def receive(self, *, max_messages: int = 1, wait_seconds: int = 20) -> list[dict[str, Any]]:
        resp = self._sqs.receive_message(
            QueueUrl=self.queue_url,
            MaxNumberOfMessages=max_messages,
            WaitTimeSeconds=wait_seconds,
        )
        raw_messages = resp.get("Messages", [])
        return [
            {
                "body": json.loads(m["Body"]),
                "receipt_handle": m["ReceiptHandle"],
            }
            for m in raw_messages
        ]

    def ack(self, receipt_handle: str) -> None:
        self._sqs.delete_message(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
        )

    def nack(self, receipt_handle: str, *, visibility_timeout: int = 0) -> None:
        self._sqs.change_message_visibility(
            QueueUrl=self.queue_url,
            ReceiptHandle=receipt_handle,
            VisibilityTimeout=visibility_timeout,
        )
