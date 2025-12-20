from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DEFAULT_PROFILES_DIR = Path("profiles")


@dataclass
class RetrievalConfig:
    retrieve_k: int = 30
    context_k: int = 5

    ai_only: bool = False
    include_moc: bool = False

    dedupe: bool = True

    mmr: bool = False
    mmr_lambda: float = 0.7

    rerank: bool = False
    rerank_candidates: int = 20


def load_profile(name: str, profiles_dir: Path = DEFAULT_PROFILES_DIR) -> RetrievalConfig:
    """
    Load profiles/<name>.json into a RetrievalConfig.
    """
    path = profiles_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Profile not found: {path}")

    profile = json.loads(path.read_text(encoding="utf-8"))
    config = RetrievalConfig()

    # Apply only known keys (ignore extras)
    for k, v in profile.items():
        if hasattr(config, k):
            setattr(config, k, v)

    return config


def override_cfg(config: RetrievalConfig, overrides: dict[str, Any]) -> RetrievalConfig:
    """
    Apply CLI overrides on top of a config.
    """
    new_config = RetrievalConfig(**asdict(config))
    for k, v in overrides.items():
        if v is None:
            continue
        if hasattr(new_config, k):
            setattr(new_config, k, v)
    return new_config
