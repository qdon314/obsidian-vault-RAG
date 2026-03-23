"""Port protocol for LLM calls in the benchmark pipeline.

All LLM interactions route through :class:`LLMClient` so the model backend
is swappable and retry / timeout / cost-tracking policies are centralized.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from benchmark.domain.models import StageConfig


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Structured response from an LLM call."""

    text: str
    model: str
    usage: dict[str, int] = field(default_factory=dict)


@runtime_checkable
class LLMClient(Protocol):
    """All LLM calls in the benchmark pipeline route through this port."""

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse: ...
