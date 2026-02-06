# AGENTS.md — Debug Mode

This file provides debugging guidance to agents when working with code in this repository.

## Critical Command Discipline

**Never run `python`, `pip`, `pytest`, `ruff`, or `streamlit` directly.**

Always use the pinned interpreter via:
- `make <target>` for common workflows
- `./scripts/py ...` for ad-hoc Python commands
- `./scripts/pip ...` for dependency management

## Running Tests

```bash
make test                              # Full test suite
./scripts/py -m pytest tests/foo.py    # Single test file
./scripts/py -m pytest -k test_name    # Filter by test name
```

## Debugging Tools

### Query Logs

Query traces are logged to `artifacts/logs/queries.jsonl`:

```bash
make tail-logs NUM_LOGS=20             # Tail and pretty-print last N logs
```

### Environment Check

```bash
make env-check                         # Verify Python environment
```

## Architecture for Debugging

**Hexagonal Architecture (Ports & Adapters)** - See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

- **Ports** (`src/rag/ports/`): Protocol-based interfaces
- **Domain** (`src/rag/domain/`): Immutable frozen dataclasses
- **Adapters** (`src/rag/adapters/`): Concrete implementations
- **App** (`src/rag/app/`): Orchestration via [`Container`](src/rag/app/container.py:38)

The [`run_query()`](src/rag/app/query_runner.py:15) function generates a complete [`QueryTrace`](src/rag/domain/models.py) with per-stage timing for observability.

## Key Debugging Notes

- Domain objects are frozen dataclasses - use `dataclasses.replace()` for modifications
- All configuration is in [`settings.toml`](settings.toml)
- Tests use fixtures from [`tests/conftest.py`](tests/conftest.py)
