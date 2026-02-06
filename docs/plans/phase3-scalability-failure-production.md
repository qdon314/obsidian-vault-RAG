# Phase 3: Scalability Patterns, Failure Modes & Production Readiness

## Date: 2026-02-05
## Status: COMPLETED

---

## 3.1 Scalability Analysis

### 3.1.1 Current Scaling Characteristics

#### Embedding Layer (src/rag/adapters/embedding/openai_embedder.py)

```python
@dataclass(frozen=True, slots=True)
class OpenAIEmbedder:
    api_key: str
    model: str = "text-embedding-3-small"

    def embed_texts(self, texts: Sequence[str], ...) -> list[Vector]:
        client = OpenAI(api_key=self.api_key)  # New client per call!
        resp = client.embeddings.create(model=self.model, input=list(texts))
        return [list(item.embedding) for item in resp.data]
```

**Issues Identified:**
1. **New client per call** - No connection pooling, high connection overhead
2. **Synchronous blocking** - One request at a time, no concurrency
3. **No batching optimization** - OpenAI supports 2048 texts per request, not utilized
4. **No retry logic** - Single failure = total failure

**Scaling Limits:**
- Throughput: ~10-50 queries/second (limited by network round-trip)
- Latency: 100-500ms per embedding (sequential)
- Cost: No request deduplication, repeated embeddings

---

#### Generation Layer (src/rag/adapters/generation/openai_chat.py)

```python
@dataclass(frozen=True, slots=True)
class OpenAIChatGenerator:
    def generate(self, query: str, context: ContextPack, ...) -> Answer:
        client = OpenAI(api_key=self.api_key)  # New client per call!
        resp = client.chat.completions.create(...)
        return Answer(...)
```

**Issues Identified:**
1. Same client-per-call problem as embeddings
2. No streaming support (waits for full response)
3. No timeout configuration
4. No fallback models

---

#### Vector Store Scaling

| Store | Max Vectors | Search Complexity | Memory | Persistence |
|-------|-------------|-------------------|--------|-------------|
| InMemory | ~100K | O(N) | Full index | None |
| Jsonl | ~1M | O(N) | Full index | JSONL file |
| Qdrant | Unlimited | O(log N) HNSW | Configurable | Disk/Cloud |

**Qdrant Configuration Gaps:**
```python
# Current: No HNSW tuning
QdrantVectorStore(
    collection_name="chunks",
    vector_size=3072,
    # Missing: hnsw_config, optimizers_config, replication_factor
)
```

---

### 3.1.2 Missing Scalability Patterns

#### Async/Concurrency

**Current State:** All I/O is synchronous blocking

**Missing:**
```python
# async embedding with batching
async def embed_texts_batch(texts: list[str]) -> list[Vector]:
    # Process in batches of 2048
    # Concurrent requests with semaphore limiting
    # Connection pooling via httpx.AsyncClient
```

#### Request Batching

**Current:** One embedding per API call
**Optimal:** Batch up to 2048 texts per OpenAI request
**Impact:** 100x reduction in API calls for indexing

#### Caching Strategy

**Current:** SQLite local cache (src/rag/adapters/embedding/sqlite_cache.py)
```python
@dataclass(frozen=True, slots=True)
class CachedEmbedder:
    embedder: Embedder
    db_path: Path  # Local SQLite only
```

**Gaps:**
- No distributed cache (Redis, etc.)
- No cache warming
- No cache invalidation strategy
- No cache hit/miss metrics

#### Load Balancing

**Missing:**
- No multiple API key rotation
- No cross-region failover
- No rate limit distribution

---

### 3.1.3 Scalability Recommendations

| Pattern | Implementation | Effort | Impact |
|---------|---------------|--------|--------|
| Async embedding | httpx.AsyncClient + asyncio | 3 days | 10x throughput |
| Request batching | Batch 2048 per OpenAI call | 2 days | 100x cost reduction |
| Connection pooling | Shared client instance | 1 day | 50% latency reduction |
| Distributed cache | Redis integration | 2 days | 30% cost reduction |
| Query caching | Semantic cache with TTL | 2 days | 50% cost reduction |

---

## 3.2 Failure Mode Analysis

### 3.2.1 Current Failure Handling

#### Known Issues (from docs/KNOWN_ISSUES.md)

```markdown
- ~~IMPORTANT!: answer eval metrics producing unreasonable reports~~
  - ~~high correctness AND high hallucination scores~~
  - ~~high citation coverage AND high hallucination~~
  - ~~REGRESSION: 0% correct abstentions~~
- Move cache embedding into container building logic
- ~~Missing trace id from eval run~~
- ~~Improve eval output naming for easier tracking~~
- ~~Address answer quality in eval output~~
- results_analyzer 
  - API is somewhat misaligned for multi-run use cases
  - Correctness and completeness from gold judge indicate 0 
    completeness and correctness for e.g. qid 1 even though 
    it's an accurate response
```

---

### 3.2.2 Critical Missing Protections

#### 1. No Timeout Handling

**Current Code (openai_embedder.py):**
```python
def embed_texts(self, texts, ...):
    client = OpenAI(api_key=self.api_key)
    resp = client.embeddings.create(...)  # No timeout!
```

**Risk:** Hanging requests block pipeline indefinitely

**Required:**
```python
client = OpenAI(
    api_key=self.api_key,
    timeout=httpx.Timeout(30.0, connect=5.0)
)
```

---

#### 2. No Retry Logic

**Current:** Single attempt, any failure propagates

**Required Pattern:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def embed_with_retry(texts):
    return client.embeddings.create(...)
```

---

#### 3. No Circuit Breaker

**Risk:** Cascading failures when embedding service degrades

**Required:**
```python
from circuitbreaker import circuit

@circuit(failure_threshold=5, recovery_timeout=60)
def embed_with_circuit_breaker(texts):
    return client.embeddings.create(...)
```

---

#### 4. No Graceful Degradation

**Current:** All-or-nothing pipeline

**Required Fallbacks:**
```python
# If embedding fails, fallback to cached similar queries
# If retrieval fails, fallback to keyword search
# If generation fails, return "I don't know" with citations
```

---

#### 5. No Input Validation

**Risk:** Malicious or malformed inputs can cause crashes

**Required:**
```python
def validate_query(query: str) -> None:
    if len(query) > 10000:
        raise ValueError("Query too long")
    if not query.strip():
        raise ValueError("Empty query")
    # PII detection, injection prevention
```

---

### 3.2.3 Failure Mode Matrix

| Component | Failure Mode | Current Behavior | Required Behavior |
|-----------|--------------|------------------|-------------------|
| OpenAI Embeddings | Rate limit (429) | Crash | Exponential backoff retry |
| OpenAI Embeddings | Timeout | Hang indefinitely | Timeout + fallback |
| OpenAI Embeddings | API error (5xx) | Crash | Circuit breaker + cache |
| Qdrant | Connection lost | Crash | Retry + fallback to cache |
| Qdrant | Query timeout | Hang | Timeout + partial results |
| JSONL Store | Disk full | Crash | Alert + read-only mode |
| Query | Malformed input | Undefined | Validation + error message |
| Query | PII detected | Process | Redact or reject |

---

## 3.3 Production Readiness Assessment

### 3.3.1 Security Posture

#### Current State: Minimal

**Strengths:**
- API keys from environment variables (not hardcoded)
- Docker runs as non-root user (appuser)
- Read-only vault mounts in docker-compose

**Critical Gaps:**

| Gap | Risk | Priority |
|-----|------|----------|
| No input sanitization | Injection attacks | P0 |
| No rate limiting | Abuse, cost overruns | P0 |
| No PII detection | Data leakage | P1 |
| No audit logging | Compliance violations | P1 |
| No encryption at rest | Data exposure | P1 |
| No network policies | Lateral movement | P2 |

---

#### Input Security

**Current:** No validation in query_runner.py
```python
def run_query(query: str, ...):  # query used directly
    retrieved = retriever.retrieve(query, ...)
```

**Required:**
```python
from presidio_analyzer import AnalyzerEngine

def sanitize_query(query: str) -> str:
    # PII detection and redaction
    # Prompt injection detection
    # Length limits
    # Character whitelist
```

---

### 3.3.2 Deployment Infrastructure

#### Docker Configuration (Dockerfile)

**Strengths:**
- Multi-stage build (smaller final image)
- Non-root user
- Layer caching optimization

**Gaps:**
- No health check defined
- No resource limits (CPU/memory)
- No graceful shutdown handling

---

#### AWS Infrastructure (infra/)

**Current Components:**
- ECR (container registry)
- ECS Fargate (container orchestration)
- S3 (artifact storage)
- SSM Parameter Store (secrets)

**Gaps:**
- No auto-scaling configuration
- No load balancer
- No VPC/network isolation
- No backup strategy
- No disaster recovery

---

### 3.3.3 Monitoring & Alerting

**Current:** None

**Required Production Setup:**

```yaml
# Prometheus metrics endpoint
metrics:
  - rag_query_latency_seconds (histogram)
  - rag_query_errors_total (counter)
  - rag_embedding_cache_hit_ratio (gauge)
  - rag_openai_requests_total (counter)
  - rag_vector_store_query_duration (histogram)

# Alerts
alerts:
  - HighErrorRate: errors > 1% for 5m
  - HighLatency: p95 > 2s for 10m
  - LowCacheHitRatio: cache_hits < 50%
  - OpenAIRateLimit: rate_limit_errors > 0
```

---

### 3.3.4 Testing Coverage

#### Current Test Structure

From search analysis:
- Unit tests for adapters (chunkers, embedders, rerankers)
- Unit tests for domain logic (filter serialization)
- Unit tests for vector stores

**Coverage Gaps:**

| Test Type | Status | Priority |
|-----------|--------|----------|
| Unit tests | Good | - |
| Integration tests | Missing | P0 |
| Load/stress tests | Missing | P1 |
| Contract tests (OpenAI) | Missing | P1 |
| Chaos engineering | Missing | P2 |
| Security tests | Missing | P1 |

---

#### Required Integration Tests

```python
# tests/integration/test_full_pipeline.py
async def test_full_pipeline():
    container = build_container()
    result = await run_query_async(
        "What is RAG?",
        container=container
    )
    assert result.answer.text
    assert result.latency_ms < 5000
```

---

### 3.3.5 Operational Readiness Checklist

| Requirement | Status | Notes |
|-------------|--------|-------|
| Health checks | Missing | HTTP endpoint needed |
| Graceful shutdown | Missing | SIGTERM handling |
| Log aggregation | Missing | JSON to stdout |
| Metrics export | Missing | Prometheus endpoint |
| Distributed tracing | Missing | OpenTelemetry |
| Secrets management | Partial | Env vars, no rotation |
| Configuration management | Good | TOML + env overrides |
| Database migrations | N/A | No relational DB |
| Backup/restore | Missing | Index backup strategy |
| Runbooks | Missing | Operational procedures |

---

## 3.4 Cost Efficiency Analysis

### 3.4.1 Current Cost Structure

**OpenAI API Costs (estimated):**
- Embeddings: $0.13 / 1M tokens (text-embedding-3-large)
- Generation: $0.60 / 1M tokens (gpt-4.1-mini)

**Inefficiencies:**
1. No request batching (100x overhead for indexing)
2. No query caching (repeated queries cost full price)
3. No embedding cache warming (cold start penalty)
4. No deduplication (same chunks embedded multiple times)

---

### 3.4.2 Cost Optimization Opportunities

| Optimization | Current | Optimized | Savings |
|--------------|---------|-----------|---------|
| Batch embeddings | 1 text/request | 2048 texts/request | 99% API calls |
| Query cache | 0% hit rate | 50% hit rate | 50% embedding cost |
| Embedding cache | SQLite local | Redis distributed | 30% embedding cost |
| Model selection | text-embedding-3-large | text-embedding-3-small | 50% cost |

---

## 3.5 Phase 3 Conclusions

### Scalability Grade: C+

**Blockers:**
- Synchronous I/O only
- No connection pooling
- No request batching
- O(N) search in default stores

**Path to A:**
1. Implement async throughout
2. Add connection pooling
3. Batch embedding requests
4. Configure HNSW properly

---

### Failure Handling Grade: D+

**Blockers:**
- No timeout handling
- No retry logic
- No circuit breaker
- No graceful degradation

**Path to A:**
1. Add timeouts to all external calls
2. Implement exponential backoff retry
3. Add circuit breaker pattern
4. Design fallback strategies

---

### Production Readiness Grade: C

**Blockers:**
- No health checks
- No monitoring/metrics
- No input validation
- No security controls

**Path to A:**
1. Add health check endpoints
2. Implement Prometheus metrics
3. Add input validation and PII detection
4. Create operational runbooks

---

*Next: Phase 4 & 5 - Final Recommendations & Action Plan*
