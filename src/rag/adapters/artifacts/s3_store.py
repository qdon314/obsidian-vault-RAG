"""S3-backed artifact store.

Syncs index artifacts (JSONL files, manifests, caches) between
a local working directory and an S3 bucket.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class S3ArtifactStore:
    """Stores and retrieves index artifacts from S3.

    Parameters
    ----------
    bucket:
        S3 bucket name.
    client:
        A boto3 S3 client. If not provided, one is created via boto3.client("s3").
    """

    bucket: str
    client: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.client is None:
            import boto3  # type: ignore[import-untyped]

            object.__setattr__(self, "client", boto3.client("s3"))

    def push(self, local_dir: Path, remote_key: str) -> None:
        """Upload all files in local_dir to s3://bucket/remote_key/.

        Raises FileNotFoundError if manifest.json is missing from local_dir.
        """
        manifest_path = local_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"manifest.json not found in {local_dir}. "
                "Index artifacts must include a manifest for provenance tracking."
            )
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                relative = file_path.relative_to(local_dir)
                s3_key = f"{remote_key}/{relative}"
                self.client.upload_file(
                    Filename=str(file_path),
                    Bucket=self.bucket,
                    Key=s3_key,
                )

    def pull(self, remote_key: str, local_dir: Path) -> Path:
        """Download all objects under s3://bucket/remote_key/ to local_dir."""
        response = self.client.list_objects_v2(
            Bucket=self.bucket,
            Prefix=remote_key,
        )

        for obj in response.get("Contents", []):
            s3_key = obj["Key"]
            relative = s3_key[len(remote_key) :].lstrip("/")
            if not relative:
                continue
            local_path = local_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.client.download_file(
                Bucket=self.bucket,
                Key=s3_key,
                Filename=str(local_path),
            )

        return local_dir
