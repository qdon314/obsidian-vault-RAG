from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    vault_dir: Path
    index_dir: Path


@dataclass(frozen=True)
class Chunking:
    method: str
    chunk_size: int
    chunk_overlap: int


@dataclass(frozen=True)
class Embeddings:
    provider: str
    model: str


@dataclass(frozen=True)
class LLM:
    provider: str
    model: str


@dataclass(frozen=True)
class Retrieval:
    top_k: int


@dataclass(frozen=True)
class Rerank:
    enabled: bool
    keep_k: int


@dataclass(frozen=True)
class Settings:
    paths: Paths
    chunking: Chunking
    embeddings: Embeddings
    llm: LLM
    retrieval: Retrieval
    rerank: Rerank


def load_settings(path: str | Path = "settings.toml") -> Settings:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    def expand(p: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(p))).resolve()

    try:
        return Settings(
            paths=Paths(
                vault_dir=expand(raw["paths"]["vault_dir"]),
                index_dir=expand(raw["paths"]["index_dir"]),
            ),
            chunking=Chunking(
                method=raw["chunking"]["method"],
                chunk_size=int(raw["chunking"]["chunk_size"]),
                chunk_overlap=int(raw["chunking"]["chunk_overlap"]),
            ),
            embeddings=Embeddings(
                provider=raw["embeddings"]["provider"],
                model=raw["embeddings"]["model"],
            ),
            llm=LLM(
                provider=raw["llm"]["provider"],
                model=raw["llm"]["model"],
            ),
            retrieval=Retrieval(
                top_k=int(raw["retrieval"]["top_k"]),
            ),
            rerank=Rerank(
                enabled=bool(raw["rerank"]["enabled"]),
                keep_k=int(raw["rerank"]["keep_k"]),
            ),
        )
    except KeyError as e:
        raise KeyError(f"Missing config key: {e}") from e
