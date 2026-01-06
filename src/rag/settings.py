from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional


# ----------------------------
# Secrets
# ----------------------------

@dataclass(frozen=True, slots=True)
class Secrets:
    openai_api_key: Optional[str]

    @staticmethod
    def from_env(*, require_openai: bool) -> "Secrets":
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


@dataclass(frozen=True, slots=True)
class Ingestion:
    recursive: bool = True
    skip_hidden: bool = True
    allowed_extensions: tuple[str, ...] = (".md", ".txt")
    expand_embeds: bool = True
    max_embed_depth: int = 4


@dataclass(frozen=True, slots=True)
class Chunking:
    backend: Literal["fixed"] = "fixed"
    chunk_size: int = 800
    overlap: int = 120


@dataclass(frozen=True, slots=True)
class Context:
    max_chunks: int = 5
    dedupe: bool = True
    include_scores: bool = False
    min_score: Optional[float] = None


@dataclass(frozen=True, slots=True)
class Embeddings:
    backend: Literal["openai", "dummy"] = "openai"
    model: str = "text-embedding-3-large"
    dummy_dim: int = 128


@dataclass(frozen=True, slots=True)
class VectorStore:
    backend: Literal["memory", "jsonl"] = "memory"
    jsonl_dir: Optional[Path] = None  # only required when backend="jsonl"


@dataclass(frozen=True, slots=True)
class LLM:
    backend: Literal["openai"] = "openai"
    model: str = "gpt-4.1-mini"
    temperature: float = 0.2
    max_tokens: int = 1024


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

def load_settings(path: str | Path = "settings.toml") -> Settings:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("rb") as f:
        raw = tomllib.load(f)

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
    require_openai = (
        emb_tbl.get("backend", emb_tbl.get("provider", "openai")) == "openai") or (
        llm_tbl.get("backend", llm_tbl.get("provider", "openai")) == "openai"
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
        backend=str(chunking_tbl.get("backend", "fixed")), # type: ignore
        chunk_size=int(chunking_tbl.get("chunk_size", 800)),
        overlap=int(chunking_tbl.get("overlap", chunking_tbl.get("chunk_overlap", 120))),
    )

    # Context
    context = Context(
        max_chunks=int(context_tbl.get("max_chunks", 5)),
        dedupe=bool(context_tbl.get("dedupe", True)),
        include_scores=bool(context_tbl.get("include_scores", False)),
        min_score=context_tbl.get("min_score", None),
    )

    # Embeddings
    embeddings = Embeddings(
        backend=str(emb_tbl.get("backend", emb_tbl.get("provider", "openai"))), # type: ignore
        model=str(emb_tbl.get("model", "text-embedding-3-large")),
        dummy_dim=int(emb_tbl.get("dummy_dim", 128)),
    )

    # Vectorstore
    vs_backend = str(vectorstore_tbl.get("backend", "memory"))
    jsonl_dir = vectorstore_tbl.get("jsonl_dir", None)
    vectorstore = VectorStore(
        backend=vs_backend,  # type: ignore[arg-type]
        jsonl_dir=expand(jsonl_dir) if isinstance(jsonl_dir, str) else None,
    )

    # LLM
    llm = LLM(
        backend=str(llm_tbl.get("backend", llm_tbl.get("provider", "openai"))), # type: ignore
        model=str(llm_tbl.get("model", "gpt-4.1-mini")),
        temperature=float(llm_tbl.get("temperature", 0.2)),
        max_tokens=int(llm_tbl.get("max_tokens", 1024)),
    )

    # Retrieval
    retrieval = Retrieval(top_k=int(retrieval_tbl.get("top_k", 8)))

    # Rerank
    rerank = Rerank(
        enabled=bool(rerank_tbl.get("enabled", True)),
        backend=str(rerank_tbl.get("backend", "heuristic")), # type: ignore
        keep_k=int(rerank_tbl.get("keep_k", 4)),
    )

    return Settings(
        paths=Paths(vault_dir=vault_dir, artifacts_dir=artifacts_dir, index_dir=index_dir),
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
