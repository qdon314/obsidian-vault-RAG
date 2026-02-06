# Configuration Reference

Complete reference for all configuration options in the RAG system.

## Table of Contents

- [Configuration File](#configuration-file)
- [Environment Variables](#environment-variables)
- [Settings Sections](#settings-sections)
- [CLI Overrides](#cli-overrides)
- [Configuration Examples](#configuration-examples)

---

## Configuration File

The primary configuration file is `settings.toml` in the project root.

```toml
# settings.toml - Canonical configuration for the RAG system
# Settings define defaults.
# Containers compose adapters.
# CLI overrides settings via ContainerOverrides.
```

### Configuration Precedence

1. **settings.toml** - Default values
2. **Environment variables** - Secrets (API keys)
3. **CLI arguments** - One-off overrides

---

## Environment Variables

### Required

| Variable | Description | Required When |
|----------|-------------|---------------|
| `OPENAI_API_KEY` | OpenAI API key | Using OpenAI embeddings or generation |

### Setting Environment Variables

**Option 1: .env file** (recommended)
```bash
# Create .env in project root
OPENAI_API_KEY='sk-your-api-key-here'
```

**Option 2: Shell export**
```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

**Option 3: Inline**
```bash
OPENAI_API_KEY='sk-...' python scripts/ask.py --q "query"
```

---

## Settings Sections

### [paths]

File system paths for the system.

```toml
[paths]
vault_dir = "~/obsidian-vault"           # Document corpus location
artifacts_dir = "artifacts"               # Output directory
index_dir = "artifacts/indexes/default"   # Default index location
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `vault_dir` | path | `~/obsidian-vault` | Path to document corpus |
| `artifacts_dir` | path | `artifacts` | Directory for outputs (logs, indexes) |
| `index_dir` | path | `artifacts/indexes/default` | Default index directory |

**Notes:**
- Paths support `~` expansion (home directory)
- Paths support environment variable expansion (`$HOME`)

---

### [ingestion]

Document ingestion settings.

```toml
[ingestion]
recursive = true
skip_hidden = true
allowed_extensions = [".md", ".txt"]
expand_embeds = true
max_embed_depth = 4
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `recursive` | bool | `true` | Recursively walk subdirectories |
| `skip_hidden` | bool | `true` | Skip files/dirs starting with `.` |
| `allowed_extensions` | list | `[".md", ".txt"]` | File extensions to process |
| `expand_embeds` | bool | `true` | Expand Obsidian transclusions (`![[...]]`) |
| `max_embed_depth` | int | `4` | Max recursion depth for transclusions |

**Obsidian Transclusions:**

When `expand_embeds = true`, embedded content like:
```markdown
![[other-note]]
```
Will be expanded inline (up to `max_embed_depth` levels).

---

### [chunking]

Text chunking settings.

```toml
[chunking]
backend = "fixed"
chunk_size = 800
overlap = 120
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | string | `"fixed"` | Chunking strategy (currently only "fixed") |
| `chunk_size` | int | `800` | Characters per chunk |
| `overlap` | int | `120` | Character overlap between chunks |

**Recommendations:**
| Corpus Type | chunk_size | overlap |
|-------------|------------|---------|
| Dense technical docs | 600-800 | 100-120 |
| General notes | 800-1000 | 120-150 |
| Long-form articles | 1000-1200 | 150-200 |

---

### [context]

Context building settings.

```toml
[context]
max_chunks = 5
dedupe = true
include_scores = false
# min_score = 0.5            # Optional threshold
token_budget = 1500
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_chunks` | int | `5` | Maximum chunks in context |
| `dedupe` | bool | `true` | Remove near-duplicate chunks |
| `include_scores` | bool | `false` | Show scores in rendered context |
| `min_score` | float | `None` | Optional similarity threshold |
| `token_budget` | int | `1500` | Maximum tokens for context |

**Token Budget:**

The token budget limits context size to fit LLM context windows:
- GPT-4: 8K-128K tokens available
- Recommended: Leave room for system prompt + response
- Default 1500 tokens ≈ 6000 characters

---

### [embeddings]

Embedding model settings.

```toml
[embeddings]
backend = "openai"
model = "text-embedding-3-large"
dummy_dim = 128
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | string | `"openai"` | `"openai"` or `"dummy"` |
| `model` | string | `"text-embedding-3-large"` | OpenAI model name |
| `dummy_dim` | int | `128` | Vector dimension for dummy embedder |

**OpenAI Models:**

| Model | Dimensions | Cost | Quality |
|-------|------------|------|---------|
| `text-embedding-3-large` | 3072 | Higher | Best |
| `text-embedding-3-small` | 1536 | Lower | Good |
| `text-embedding-ada-002` | 1536 | Medium | Legacy |

**Dummy Embedder:**

Use `backend = "dummy"` for:
- Testing without API costs
- Development environments
- CI/CD pipelines

---

### [vectorstore]

Vector storage settings.

```toml
[vectorstore]
backend = "jsonl"         # "memory" | "jsonl" | "qdrant"
jsonl_dir = "artifacts/indexes/obsidian_index"

# Qdrant-specific settings (only when backend = "qdrant")
# qdrant_collection = "chunks"
# qdrant_url = "http://localhost:6333"  # for remote Qdrant server
# qdrant_path = "artifacts/qdrant"      # for local disk persistence
# qdrant_api_key = "..."                # for Qdrant Cloud
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | string | `"memory"` | `"memory"`, `"jsonl"`, or `"qdrant"` |
| `jsonl_dir` | path | `None` | Directory for JSONL files |
| `qdrant_collection` | string | `"chunks"` | Qdrant collection name |
| `qdrant_url` | string | `None` | Qdrant server URL (remote mode) |
| `qdrant_path` | path | `None` | Local disk path for Qdrant |
| `qdrant_api_key` | string | `None` | API key for Qdrant Cloud |

**Backend Comparison:**

| Backend | Persistence | Speed | Use Case |
|---------|-------------|-------|----------|
| `memory` | No | Fast | Testing, experiments |
| `jsonl` | Yes | Good | Small-medium corpora, human-readable |
| `qdrant` | Yes | Best | Large corpora, production scale |

**JSONL Files:**
```
{jsonl_dir}/
└── chunks.jsonl    # Chunk data + vectors
```

**Qdrant Deployment Modes:**

| Mode | Configuration | Description |
|------|---------------|-------------|
| In-memory | No `url` or `path` | Fast testing, non-persistent |
| Local disk | Set `path` | Persistent local storage |
| Remote server | Set `url` | Connect to Qdrant server |
| Qdrant Cloud | Set `url` + `api_key` | Managed cloud service |

**Qdrant Installation:**
```bash
pip install -e ".[qdrant]"
```

---

### [retrieval]

Retrieval settings.

```toml
[retrieval]
backend = "vector"           # "vector" | "hybrid"
top_k = 8

# Hybrid search settings (only when backend = "hybrid")
[retrieval.hybrid]
primary_weight = 0.7         # Weight for vector results
secondary_weight = 0.3       # Weight for BM25 results
rrf_k = 60                   # RRF fusion constant
bm25_k1 = 1.5               # BM25 term frequency saturation
bm25_b = 0.75               # BM25 length normalization
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | string | `"vector"` | `"vector"` or `"hybrid"` |
| `top_k` | int | `8` | Initial candidates to retrieve |

**Hybrid Sub-section:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `primary_weight` | float | `0.7` | Weight for vector (primary) results in RRF fusion |
| `secondary_weight` | float | `0.3` | Weight for BM25 (secondary) results in RRF fusion |
| `rrf_k` | int | `60` | RRF constant (higher = more rank smoothing) |
| `bm25_k1` | float | `1.5` | BM25 term frequency saturation parameter |
| `bm25_b` | float | `0.75` | BM25 length normalization parameter |

**Backend Comparison:**

| Backend | Description | Speed | Recall | Use Case |
|---------|-------------|-------|--------|----------|
| `vector` | Pure vector similarity | Fastest | Good | General use, default |
| `hybrid` | Vector + BM25 with RRF | Fast | Better | Rare terms, acronyms, proper nouns |

**Hybrid Search:**

Hybrid search combines vector similarity with BM25 keyword search using Reciprocal Rank Fusion (RRF):

```
RRF Score = Σ(weight / (k + rank))
```

Where:
- `weight` is `primary_weight` or `secondary_weight`
- `k` is the RRF constant (default 60)
- `rank` is the position in each result list (1-indexed)

**BM25 Parameters:**

- `k1` (1.5): Controls term frequency saturation. Higher values allow more term frequency influence.
- `b` (0.75): Controls length normalization. 0 = no normalization, 1 = full normalization.

**Guidance:**
- Higher `top_k` → More candidates for reranking
- Typical range: 5-20
- Consider corpus size and diversity
- Use `hybrid` when queries contain rare terms, acronyms, or proper nouns
- BM25 retriever is built in-memory from loaded chunks (no separate index file)

---

### [rerank]

Reranking settings.

```toml
[rerank]
enabled = true
backend = "heuristic"
keep_k = 4
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | bool | `true` | Enable reranking |
| `backend` | string | `"heuristic"` | `"heuristic"` or `"noop"` |
| `keep_k` | int | `4` | Candidates to keep after reranking |

**Backend Comparison:**

| Backend | Description | Speed | Quality |
|---------|-------------|-------|---------|
| `heuristic` | Lexical overlap + diversity | Fast | Improved |
| `noop` | Pass-through (vector only) | Fastest | Baseline |

**Reranking Pipeline:**
```
Retrieve top_k=8 → Rerank → Keep top keep_k=4 → Context
```

---

### [llm]

Language model settings.

```toml
[llm]
backend = "openai"
model = "gpt-4.1-mini"
temperature = 0.2
max_tokens = 1024
```

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `backend` | string | `"openai"` | LLM provider (currently only "openai") |
| `model` | string | `"gpt-4.1-mini"` | Model name |
| `temperature` | float | `0.2` | Generation temperature (0-1) |
| `max_tokens` | int | `1024` | Maximum response tokens |

**Temperature Guidance:**
| Temperature | Behavior | Use Case |
|-------------|----------|----------|
| 0.0-0.2 | Deterministic, focused | Factual QA |
| 0.3-0.5 | Balanced | General use |
| 0.6-1.0 | Creative, varied | Creative tasks |

**Recommended Models:**
| Model | Speed | Quality | Cost |
|-------|-------|---------|------|
| `gpt-4o-mini` | Fast | Good | Low |
| `gpt-4o` | Medium | Better | Medium |
| `gpt-4-turbo` | Slower | Best | High |

---

## CLI Overrides

Command-line flags override settings for a single run.

### build_index.py

```bash
python scripts/build_index.py \
  --corpus PATH \           # Override paths.vault_dir
  --index-name NAME \       # Index directory name
  --use-dummy-embeddings \  # Use dummy embedder
  --chunk-size N \          # Override chunking.chunk_size
  --chunk-overlap N         # Override chunking.overlap
```

### ask.py

```bash
python scripts/ask.py \
  --index NAME \            # Index to query
  --q "QUERY" \             # Query text
  --top-k N \               # Override retrieval.top_k
  --keep-k N \              # Override rerank.keep_k
  --rerank-backend TYPE \   # Override rerank.backend
  --token-budget N          # Override context.token_budget
```

---

## Configuration Examples

### Development Configuration

```toml
# settings.toml for development
[paths]
vault_dir = "./test_vault"
artifacts_dir = "artifacts"

[embeddings]
backend = "dummy"
dummy_dim = 128

[vectorstore]
backend = "memory"

[rerank]
enabled = false
```

### Production Configuration

```toml
# settings.toml for production
[paths]
vault_dir = "/data/obsidian-vault"
artifacts_dir = "/data/artifacts"
index_dir = "/data/indexes/production"

[chunking]
chunk_size = 800
overlap = 120

[embeddings]
backend = "openai"
model = "text-embedding-3-large"

[vectorstore]
backend = "jsonl"
jsonl_dir = "/data/indexes/production"

[retrieval]
top_k = 10

[rerank]
enabled = true
backend = "heuristic"
keep_k = 5

[context]
max_chunks = 6
token_budget = 2000

[llm]
model = "gpt-4o-mini"
temperature = 0.2
```

### High-Quality Configuration

```toml
# settings.toml for maximum quality
[chunking]
chunk_size = 600
overlap = 100

[embeddings]
model = "text-embedding-3-large"

[retrieval]
top_k = 15

[rerank]
enabled = true
keep_k = 6

[context]
max_chunks = 8
token_budget = 3000

[llm]
model = "gpt-4o"
temperature = 0.1
```

### Fast/Cheap Configuration

```toml
# settings.toml for speed/cost optimization
[chunking]
chunk_size = 1000
overlap = 100

[embeddings]
model = "text-embedding-3-small"

[retrieval]
top_k = 5

[rerank]
enabled = false

[context]
max_chunks = 3
token_budget = 1000

[llm]
model = "gpt-4o-mini"
temperature = 0.2
```

### Qdrant Local Configuration

```toml
# settings.toml for Qdrant with local disk persistence
[paths]
vault_dir = "~/obsidian-vault"
artifacts_dir = "artifacts"

[embeddings]
backend = "openai"
model = "text-embedding-3-large"

[vectorstore]
backend = "qdrant"
qdrant_collection = "obsidian_chunks"
qdrant_path = "artifacts/qdrant"

[retrieval]
top_k = 10

[rerank]
enabled = true
backend = "heuristic"
keep_k = 5
```

### Qdrant Cloud Configuration

```toml
# settings.toml for Qdrant Cloud (production)
[paths]
vault_dir = "/data/obsidian-vault"
artifacts_dir = "/data/artifacts"

[embeddings]
backend = "openai"
model = "text-embedding-3-large"

[vectorstore]
backend = "qdrant"
qdrant_collection = "production_chunks"
qdrant_url = "https://your-cluster-id.qdrant.io"
qdrant_api_key = "your-api-key"  # Or use environment variable

[retrieval]
top_k = 15

[rerank]
enabled = true
backend = "heuristic"
keep_k = 6

[context]
max_chunks = 8
token_budget = 3000

[llm]
model = "gpt-4o"
temperature = 0.1
```

---

## Validation

Settings are validated at load time. Common validation errors:

| Error | Cause | Fix |
|-------|-------|-----|
| `Missing config file` | settings.toml not found | Create settings.toml |
| `OPENAI_API_KEY is required` | Missing API key | Set env variable or use dummy |
| `[section] must be a table` | Invalid TOML syntax | Check TOML formatting |
| `jsonl_dir is required` | Missing JSONL path | Set vectorstore.jsonl_dir |

---

## Settings Loading

Settings are loaded via `rag.settings.load_settings()`:

```python
from rag.settings import load_settings, Settings

# Load from default path
settings = load_settings()

# Load from custom path
settings = load_settings("custom_settings.toml")

# Access settings
print(settings.chunking.chunk_size)  # 800
print(settings.embeddings.model)     # text-embedding-3-large
```

### Settings Structure

```python
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
    secrets: Secrets  # Loaded from environment
```
