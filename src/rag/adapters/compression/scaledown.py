from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace

from rag.domain.models import CompressionResult, ContextPack

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScaleDownCompressor:
    """Compress a ContextPack via the ScaleDown API before generation.

    Sends rendered_context and prompt to the ScaleDown raw compress endpoint.
    On any API error the original ContextPack is returned unchanged (fail-open).

    Requires:
        pip install -e ".[scaledown]"   # adds httpx
    """

    api_key: str
    api_url: str = "https://api.scaledown.xyz/compress/raw/"
    rate: str = "auto"
    prompt_template: str = "{query}"
    timeout_s: float = 10.0
    max_retries: int = 2
    fail_open: bool = True

    @property
    def name(self) -> str:
        return "scaledown"

    def compress(
        self,
        context_pack: ContextPack,
        *,
        query: str,
        metadata: Mapping[str, object] | None = None,
    ) -> CompressionResult:
        import httpx

        tokens_before = int(context_pack.metadata.get("tokens_used_est", 0))
        prompt = self.prompt_template.format(query=query)

        payload = {
            "context": context_pack.rendered_context,
            "prompt": prompt,
            "scaledown": {"rate": self.rate},
        }

        t0 = time.perf_counter()
        last_exc: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    self.api_url,
                    json=payload,
                    headers={"x-api-key": self.api_key},
                    timeout=self.timeout_s,
                )
                response.raise_for_status()
                data: dict = response.json()
                latency_ms = int((time.perf_counter() - t0) * 1000)

                compressed_text: str = data.get("compressed_prompt", "")
                tokens_after = int(data.get("compressed_prompt_tokens", tokens_before))
                original_tokens = int(data.get("original_prompt_tokens", tokens_before))
                token_savings = int(data.get("token_savings", original_tokens - tokens_after))
                savings_pct = float(
                    data.get(
                        "savings_pct",
                        token_savings / original_tokens if original_tokens > 0 else 0.0,
                    )
                )

                updated_pack = replace(
                    context_pack,
                    rendered_context=compressed_text,
                    metadata={**dict(context_pack.metadata), "tokens_used_est": tokens_after},
                )

                # Capture remaining response fields as extra metadata
                known_keys = {
                    "compressed_prompt",
                    "original_prompt_tokens",
                    "compressed_prompt_tokens",
                    "token_savings",
                    "savings_pct",
                }
                extra = {k: v for k, v in data.items() if k not in known_keys}

                return CompressionResult(
                    context_pack=updated_pack,
                    successful=True,
                    tokens_before=tokens_before,
                    tokens_after=tokens_after,
                    savings_pct=savings_pct,
                    latency_ms=latency_ms,
                    adapter="scaledown",
                    extra=extra,
                )

            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if exc.response.status_code < 500:
                    # Client error (4xx) — retrying will not help
                    logger.warning(
                        "ScaleDown returned non-retriable %d; giving up",
                        exc.response.status_code,
                    )
                    break
                logger.warning(
                    "ScaleDown compress attempt %d/%d server error: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError) as exc:
                last_exc = exc
                logger.warning(
                    "ScaleDown compress attempt %d/%d transient error: %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

        latency_ms = int((time.perf_counter() - t0) * 1000)

        if not self.fail_open:
            raise RuntimeError(
                f"ScaleDown compression failed after {self.max_retries} attempts"
            ) from last_exc

        logger.warning("ScaleDown compression failed (fail_open=True); using original context")
        return CompressionResult(
            context_pack=context_pack,
            successful=False,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
            savings_pct=0.0,
            latency_ms=latency_ms,
            adapter="scaledown",
            extra={"error": str(last_exc)},
        )
