# Spec 05: Dependency Cleanup

## Title
Remove Dead Dependencies from pyproject.toml

## Context / Problem

`pyproject.toml` lists dependencies from a prior architecture that are no longer imported anywhere:

| Dependency | Size Impact | Used? |
|---|---|---|
| `llama-index` | ~100MB+ | No — custom hex architecture replaced it |
| `chromadb` | ~50MB+ | No — replaced by Qdrant/JSONL stores |
| `torch` | ~2GB | No — not imported in `src/` or `eval/` |
| `llama-index-vector-stores-chroma` | ~10MB | No — companion to chromadb |
| `llama-index-embeddings-huggingface` | ~50MB | No — OpenAI embeddings used instead |

These bloat the Docker image by ~3GB+, slow pip installs, and create a misleading dependency picture.

## Goals
- Remove unused dependencies
- Verify no imports break
- Reduce Docker image size

## Non-Goals
- Refactoring existing code
- Changing runtime behavior

## Proposed Changes

### `pyproject.toml`

Remove from `dependencies`:

```toml
# REMOVE these:
"llama-index",
"chromadb",
"torch",
"llama-index-vector-stores-chroma",
"llama-index-embeddings-huggingface",
```

Resulting `dependencies`:

```toml
dependencies = [
    "python-dotenv",
    "rich",
    "dataclasses-json>=0.6.0",
]
```

## Acceptance Criteria

- [ ] No removed dependency is imported anywhere in `src/` or `eval/`
- [ ] `make test` passes
- [ ] `make lint` passes
- [ ] `make typecheck` passes
- [ ] Docker image builds successfully
- [ ] Docker image size decreases significantly

## Verification

```bash
# Confirm no imports reference removed packages
grep -r "import llama_index" src/ eval/
grep -r "import chromadb" src/ eval/
grep -r "import torch" src/ eval/

# Full test suite
make test && make lint && make typecheck

# Docker build
docker build -t rag-obsidian:test .
docker images rag-obsidian:test  # Compare size to previous
```

## Risks

| Risk | Mitigation |
|---|---|
| Transitive dependency on removed package | Test suite catches breakage; check with `pip check` |
| Scripts outside src/ or eval/ use these | Grep the entire repo, not just src/eval/ |
