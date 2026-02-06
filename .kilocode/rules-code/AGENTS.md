# AGENTS.md — Code Mode

This file provides coding guidance to agents when working with code in this repository.

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

## Project Architecture

**Hexagonal Architecture (Ports & Adapters)** - See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)

- **Ports** (`src/rag/ports/`): Protocol-based interfaces using PEP 544 structural subtyping
- **Domain** (`src/rag/domain/`): Immutable frozen dataclasses with stable content-hashed IDs
- **Adapters** (`src/rag/adapters/`): Concrete implementations organized by responsibility
- **App** (`src/rag/app/`): Orchestration via [`Container`](src/rag/app/container.py:38) dataclass for dependency injection

## Key Coding Conventions

- **Domain objects are frozen dataclasses** - use `dataclasses.replace()` for modifications
- **Protocols, not inheritance** - adapters implement via structural subtyping (PEP 544)
- **Stable IDs** - `doc_id` and `chunk_id` are content-hashed for reproducibility
- **Metadata threading** - optional `metadata` parameter on port methods for tracing
- **Offset tracking** - chunks store `start_char`/`end_char` for precise sourcing

## Code Style

- Ruff: target Python 3.11, line-length 100, double quotes
- MyPy: `mypy_path = ["src"]` with explicit package bases
- Tests: fixtures in [`tests/conftest.py`](tests/conftest.py), naming checks softened (see `pyproject.toml`)

## Git

- Do NOT auto-commit
- Do NOT add "Co-authored-by" attribution
- Provide suggested commits at end of work
