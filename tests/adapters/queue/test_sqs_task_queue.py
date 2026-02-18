"""Tests for SQSTaskQueue — uses unittest.mock for boto3."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from rag.adapters.queue.sqs_task_queue import SQSTaskQueue


@pytest.fixture
def mock_sqs() -> MagicMock:
    return MagicMock()


@pytest.fixture
def queue(mock_sqs: MagicMock) -> SQSTaskQueue:
    return SQSTaskQueue._for_test(
        queue_url="https://sqs.us-east-1.amazonaws.com/123/test-queue", sqs_client=mock_sqs
    )


class TestSend:
    def test_send_single(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        queue.send({"job_id": "j1", "corpus_id": "c1", "doc_id": "d1"})
        mock_sqs.send_message.assert_called_once()
        kwargs = mock_sqs.send_message.call_args.kwargs
        body = json.loads(kwargs["MessageBody"])
        assert body["doc_id"] == "d1"

    def test_send_batch(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        mock_sqs.send_message_batch.return_value = {"Failed": []}
        msgs = [{"job_id": "j", "corpus_id": "c", "doc_id": f"d{i}"} for i in range(3)]
        queue.send_batch(msgs)
        mock_sqs.send_message_batch.assert_called_once()
        entries = mock_sqs.send_message_batch.call_args.kwargs["Entries"]
        assert len(entries) == 3


class TestReceive:
    def test_receive_returns_parsed_messages(
        self, queue: SQSTaskQueue, mock_sqs: MagicMock
    ) -> None:
        mock_sqs.receive_message.return_value = {
            "Messages": [
                {
                    "Body": json.dumps({"job_id": "j1", "doc_id": "d1"}),
                    "ReceiptHandle": "handle-1",
                }
            ]
        }
        msgs = queue.receive(max_messages=1)
        assert len(msgs) == 1
        assert msgs[0]["body"]["doc_id"] == "d1"
        assert msgs[0]["receipt_handle"] == "handle-1"

    def test_receive_empty_queue(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        mock_sqs.receive_message.return_value = {}
        msgs = queue.receive()
        assert msgs == []


class TestAckNack:
    def test_ack_deletes_message(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        queue.ack("handle-1")
        mock_sqs.delete_message.assert_called_once_with(
            QueueUrl=queue.queue_url, ReceiptHandle="handle-1"
        )

    def test_nack_changes_visibility(self, queue: SQSTaskQueue, mock_sqs: MagicMock) -> None:
        queue.nack("handle-1", visibility_timeout=60)
        mock_sqs.change_message_visibility.assert_called_once()
