from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rag.config.env_override import apply_env_overrides

# ----------------------------
# Secrets
# ----------------------------


@dataclass(frozen=True, slots=True)
class Secrets:
    openai_api_key: str | None

    @staticmethod
    def from_env(*, require_openai: bool) -> Secrets:
        key = os.getenv("OPENAI_API_KEY")
        if require_openai and not key:
            raise RuntimeError("OPENAI_API_KEY is required but not set in environment")
        return Secrets(openai_api_key=key)


# ----------------------------
# Config sections
# ----------------------------


@dataclass(frozen=True, slots=True)
class Paths:
    vault_dir: Path
    artifacts_dir: Path
    index_dir: Path  # default index dir (may be overridden by CLI)
    queries_file: Path


@dataclass(frozen=True, slots=True)
class Ingestion:
    recursive: bool = True
    skip_hidden: bool = True
    allowed_extensions: tuple[str, ...] = (".md", ".txt")
    expand_embeds: bool = True
    max_embed_depth: int = 4


@dataclass(frozen=True, slots=True)
class Chunking:
    backend: Literal["fixed", "obsidian_structural", "obsidian_proposition"] = "fixed"
    # fixed chunker params
    chunk_size: int = 800
    overlap: int = 120
    # obsidian_structural chunker params
    target_chars: int = 4000
    hard_max_chars: int = 5200
    overlap_blocks: int = 1
    include_heading_preamble: bool = True
    # obsidian_proposition chunker params
    proposition_batch_size: int = 8


@dataclass(frozen=True, slots=True)
class Context:
    max_chunks: int = 5
    dedupe: bool = True
    include_scores: bool = False
    min_score: float | None = None
    token_budget: int = 1500


@dataclass(frozen=True, slots=True)
class Embeddings:
    backend: Literal["openai", "dummy"] = "openai"
    model: str = "text-embedding-3-large"
    dummy_dim: int = 128
    cache_embeddings: bool = False
    cache_db_path: Path | None = None  # path to SQLite cache DB
    timeout: float = 30.0
    max_retries: int = 3


@dataclass(frozen=True, slots=True)
class VectorStore:
    backend: Literal["memory", "jsonl", "qdrant"] = "memory"
    jsonl_dir: Path | None = None  # only required when backend="jsonl"
    # Qdrant-specific settings
    qdrant_collection: str = "chunks"
    qdrant_url: str | None = None  # remote Qdrant server URL
    qdrant_path: Path | None = None  # local disk path for Qdrant
    qdrant_api_key: str | None = None  # API key for Qdrant Cloud


@dataclass(frozen=True, slots=True)
class LLM:
    backend: Literal["openai"] = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: float = 60.0
    max_retries: int = 3


@dataclass(frozen=True, slots=True)
class Retrieval:
    top_k: int = 8


@dataclass(frozen=True, slots=True)
class Rerank:
    enabled: bool = True
    backend: Literal["heuristic", "noop"] = "heuristic"
    keep_k: int = 4


@dataclass(frozen=True, slots=True)
class Settings:
    paths: Paths
    ingestion: Ingestion
    chunking: Chunking
    context: Context
    embeddings: Embeddings
    vectorstore: VectorStore
    llm: LLM
    retrieval: Retrieval
    rerank: Rerank
    secrets: Secrets


# ----------------------------
# Loader
# ----------------------------


def load_settings(path: str | Path = "settings.toml", require_openai: bool = True) -> Settings:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

    raw = apply_env_overrides(raw)

    def expand(p: str) -> Path:
        return Path(os.path.expandvars(os.path.expanduser(p))).resolve()

    def get_tbl(name: str) -> dict:
        v = raw.get(name, {})
        if not isinstance(v, dict):
            raise TypeError(f"[{name}] must be a table")
        return v

    # Determine whether OpenAI secrets are required
    emb_tbl = get_tbl("embeddings")
    llm_tbl = get_tbl("llm")
    require_openai = require_openai and (
        (emb_tbl.get("backend", emb_tbl.get("provider", "openai")) == "openai")
        or (llm_tbl.get("backend", llm_tbl.get("provider", "openai")) == "openai")
    )

    paths_tbl = get_tbl("paths")
    ingestion_tbl = get_tbl("ingestion")
    chunking_tbl = get_tbl("chunking")
    context_tbl = get_tbl("context")
    vectorstore_tbl = get_tbl("vectorstore")
    retrieval_tbl = get_tbl("retrieval")
    rerank_tbl = get_tbl("rerank")

    # Paths
    vault_dir = expand(paths_tbl["vault_dir"])
    artifacts_dir = expand(paths_tbl.get("artifacts_dir", "artifacts"))
    index_dir = expand(paths_tbl.get("index_dir", str(artifacts_dir / "indexes" / "default")))
    queries_file = expand(paths_tbl.get("queries_file", "eval/datasets/curated_queries.jsonl"))

    # Ingestion
    allowed_exts_raw = ingestion_tbl.get("allowed_extensions", [".md", ".txt"])
    if isinstance(allowed_exts_raw, str):
        # allow comma-separated string too
        allowed_exts = tuple(e.strip() for e in allowed_exts_raw.split(",") if e.strip())
    else:
        allowed_exts = tuple(str(e) for e in allowed_exts_raw)

    ingestion = Ingestion(
        recursive=bool(ingestion_tbl.get("recursive", True)),
        skip_hidden=bool(ingestion_tbl.get("skip_hidden", True)),
        allowed_extensions=allowed_exts,
        expand_embeds=bool(ingestion_tbl.get("expand_embeds", True)),
        max_embed_depth=int(ingestion_tbl.get("max_embed_depth", 4)),
    )

    # Chunking
    chunking = Chunking(
        backend=str(chunking_tbl.get("backend", "fixed")),  # type: ignore
        chunk_size=int(chunking_tbl.get("chunk_size", 800)),
        overlap=int(chunking_tbl.get("overlap", chunking_tbl.get("chunk_overlap", 120))),
        target_chars=int(chunking_tbl.get("target_chars", 4000)),
        hard_max_chars=int(chunking_tbl.get("hard_max_chars", 5200)),
        overlap_blocks=int(chunking_tbl.get("overlap_blocks", 1)),
        include_heading_preamble=bool(chunking_tbl.get("include_heading_preamble", True)),
        proposition_batch_size=int(chunking_tbl.get("proposition_batch_size", 8)),
    )

    # Context
    context = Context(
        max_chunks=int(context_tbl.get("max_chunks", 5)),
        dedupe=bool(context_tbl.get("dedupe", True)),
        include_scores=bool(context_tbl.get("include_scores", False)),
        min_score=context_tbl.get("min_score", None),
        token_budget=int(context_tbl.get("token_budget", 1500)),
    )

    # Embeddings
    cache_db_path_raw = emb_tbl.get("cache_db_path", None)
    embeddings = Embeddings(
        backend=str(emb_tbl.get("backend", emb_tbl.get("provider", "openai"))),  # type: ignore
        model=str(emb_tbl.get("model", "text-embedding-3-large")),
        dummy_dim=int(emb_tbl.get("dummy_dim", 128)),
        cache_embeddings=bool(emb_tbl.get("cache_embeddings", True)),
        cache_db_path=expand(cache_db_path_raw) if isinstance(cache_db_path_raw, str) else None,
        timeout=float(emb_tbl.get("timeout", 30.0)),
        max_retries=int(emb_tbl.get("max_retries", 3)),
    )

    # Vectorstore
    vs_backend = str(vectorstore_tbl.get("backend", "memory"))
    jsonl_dir = vectorstore_tbl.get("jsonl_dir", None)
    qdrant_path = vectorstore_tbl.get("qdrant_path", None)
    vectorstore = VectorStore(
        backend=vs_backend,  # type: ignore[arg-type]
        jsonl_dir=expand(jsonl_dir) if isinstance(jsonl_dir, str) else None,
        qdrant_collection=str(vectorstore_tbl.get("qdrant_collection", "chunks")),
        qdrant_url=vectorstore_tbl.get("qdrant_url", None),
        qdrant_path=expand(qdrant_path) if isinstance(qdrant_path, str) else None,
        qdrant_api_key=vectorstore_tbl.get("qdrant_api_key", None),
    )

    # LLM
    llm = LLM(
        backend=str(llm_tbl.get("backend", llm_tbl.get("provider", "openai"))),  # type: ignore
        model=str(llm_tbl.get("model", "gpt-4.1-mini")),
        temperature=float(llm_tbl.get("temperature", 0.2)),
        max_tokens=int(llm_tbl.get("max_tokens", 1024)),
        timeout=float(llm_tbl.get("timeout", 60.0)),
        max_retries=int(llm_tbl.get("max_retries", 3)),
    )

    # Retrieval
    retrieval = Retrieval(top_k=int(retrieval_tbl.get("top_k", 8)))

    # Rerank
    rerank = Rerank(
        enabled=bool(rerank_tbl.get("enabled", True)),
        backend=str(rerank_tbl.get("backend", "heuristic")),  # type: ignore
        keep_k=int(rerank_tbl.get("keep_k", 4)),
    )

    return Settings(
        paths=Paths(
            vault_dir=vault_dir,
            artifacts_dir=artifacts_dir,
            index_dir=index_dir,
            queries_file=queries_file,
        ),
        ingestion=ingestion,
        chunking=chunking,
        context=context,
        embeddings=embeddings,
        vectorstore=vectorstore,
        llm=llm,
        retrieval=retrieval,
        rerank=rerank,
        secrets=Secrets.from_env(require_openai=require_openai),
    )
