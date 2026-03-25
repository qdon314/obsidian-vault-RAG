"""OpenAI adapter for the benchmark ``LLMClient`` port.

Requires the ``openai`` extra: ``./scripts/pip install -e '.[openai]'``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from openai import OpenAI

from benchmark.domain.models import StageConfig
from benchmark.ports.llm_client import LLMResponse

# Default retry budget: 6 attempts gives ~30s of exponential backoff,
# enough to recover from a sustained rate-limit window.
_DEFAULT_MAX_RETRIES = 6


@dataclass(frozen=True, slots=True)
class OpenAILLMClient:
    """Benchmark LLMClient adapter backed by the OpenAI chat completions API.

    Args:
        api_key: OpenAI API key.
        max_retries: Number of automatic retries on transient errors / 429s
            (default 6 — wider budget than the SDK default of 2).
        min_delay_s: Minimum seconds to sleep between consecutive calls.
            Use this to stay below a requests-per-minute limit without
            relying solely on reactive retries (e.g. 0.5 → ≤120 RPM,
            1.0 → ≤60 RPM).  Default 0 (no enforced delay).
    """

    api_key: str
    max_retries: int = _DEFAULT_MAX_RETRIES
    min_delay_s: float = 0.0
    _client: OpenAI = field(init=False, repr=False, compare=False, hash=False)
    _last_call_time: float = field(init=False, repr=False, compare=False, hash=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "_client",
            OpenAI(api_key=self.api_key, max_retries=self.max_retries),
        )
        object.__setattr__(self, "_last_call_time", 0.0)

    def complete(self, prompt: str, config: StageConfig) -> LLMResponse:
        if self.min_delay_s > 0:
            elapsed = time.monotonic() - self._last_call_time
            wait = self.min_delay_s - elapsed
            if wait > 0:
                time.sleep(wait)

        resp = self._client.chat.completions.create(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_s,
            messages=[{"role": "user", "content": prompt}],
        )
        object.__setattr__(self, "_last_call_time", time.monotonic())

        text = (resp.choices[0].message.content or "").strip()
        usage = {}
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(text=text, model=resp.model, usage=usage)
