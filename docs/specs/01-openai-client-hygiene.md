# Spec 01: OpenAI Client Hygiene

## Title
Fix Client Instantiation, Add Timeouts, Enable SDK Retry

## Context / Problem

Both `OpenAIEmbedder.embed_texts()` and `OpenAIChatGenerator.generate()` create a new `OpenAI()` client on every call. This wastes connections, prevents HTTP connection pooling, and means no timeout or retry behavior is configured.

## Goals
- Move client instantiation to `__post_init__` (created once, reused)
- Configure request timeouts
- Enable the SDK's built-in exponential backoff retry
- Ensure batching works correctly for large indexing jobs

## Non-Goals
- Async rewrite (unnecessary for CLI)
- Custom retry logic beyond SDK's built-in support
- Circuit breaker pattern

## Proposed Changes

### `src/rag/adapters/embedding/openai_embedder.py`

```python
@dataclass(frozen=True, slots=True)
class OpenAIEmbedder:
    api_key: str
    model: str = "text-embedding-3-small"
    timeout: float = 30.0
    max_retries: int = 3

    def __post_init__(self) -> None:
        object.__setattr__(self, '_client', OpenAI(
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=self.max_retries,
        ))

    def embed_texts(self, texts: Sequence[str], *, metadata=None) -> list[Vector]:
        resp = self._client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in resp.data]
```

### `src/rag/adapters/generation/openai_chat.py`

Same pattern: move `OpenAI()` to `__post_init__`, add timeout and max_retries.

### Configuration (settings.toml)

```toml
[embeddings]
timeout = 30.0
max_retries = 3

[llm]
timeout = 60.0
max_retries = 3
```

### Container Integration

Update `build_container()` to pass timeout/retry config to adapter constructors.

## Acceptance Criteria

- [ ] `OpenAI()` client instantiated once per adapter, not per call
- [ ] Timeout configured on both embedder and generator
- [ ] SDK retry enabled with configurable `max_retries`
- [ ] Existing tests pass unchanged
- [ ] Indexing large corpora batches correctly (SDK handles up to 2048 texts/request)

## Test Plan

```python
def test_client_reused_across_calls():
    """Client is created once, not per embed_texts call."""

def test_timeout_raises_on_slow_response():
    """Slow responses raise TimeoutError."""

def test_retry_on_transient_error():
    """SDK retries on 429/5xx (verified via mock)."""
```

## Risks

| Risk | Mitigation |
|---|---|
| `object.__setattr__` on frozen dataclass | Standard pattern for frozen dataclass post-init; alternative is dropping `frozen=True` |
| Shared client across threads | OpenAI client is thread-safe for sync usage |
