"""Tests for distributed ingestion settings."""
from __future__ import annotations

from rag.settings import DistributedIngestion, load_settings


class TestDistributedIngestionDefaults:
    def test_defaults_are_disabled(self) -> None:
        cfg = DistributedIngestion()
        assert cfg.enabled is False
        assert cfg.postgres_dsn is None
        assert cfg.sqs_queue_url is None
        assert cfg.corpus_s3_bucket is None


class TestLoadSettingsIncludesDistributed:
    def test_distributed_section_present(self, tmp_path) -> None:
        toml_path = tmp_path / "settings.toml"
        toml_path.write_text("""
[paths]
vault_dir = "/tmp/vault"

[distributed_ingestion]
enabled = true
postgres_dsn = "postgresql://user:pass@host:5432/rag"
sqs_queue_url = "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
corpus_s3_bucket = "rag-prod-artifacts"
corpus_s3_prefix = "corpus"
""")
        cfg = load_settings(toml_path, require_openai=False)
        assert cfg.distributed_ingestion.enabled is True
        assert cfg.distributed_ingestion.postgres_dsn == "postgresql://user:pass@host:5432/rag"
        assert cfg.distributed_ingestion.sqs_queue_url == "https://sqs.us-east-1.amazonaws.com/123/rag-tasks"
        assert cfg.distributed_ingestion.corpus_s3_bucket == "rag-prod-artifacts"
        assert cfg.distributed_ingestion.corpus_s3_prefix == "corpus"
