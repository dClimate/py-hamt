# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

Setup environment:
```bash
uv sync
source .venv/bin/activate
pre-commit install
```

Run all checks (tests, linting, formatting, type checking):
```bash
bash run-checks.sh
```

Run tests:
```bash
# All tests (requires IPFS daemon or Docker)
pytest --ipfs --cov=py_hamt tests/

# Quick tests without IPFS integration
pytest --cov=py_hamt tests/

# Single test file
pytest tests/test_hamt.py

# Coverage report
uv run coverage report --fail-under=100 --show-missing
```

Linting and formatting:
```bash
# Run all pre-commit hooks
uv run pre-commit run --all-files --show-diff-on-failure

# Fix auto-fixable ruff issues
uv run ruff check --fix
```

Type checking and other tools:
```bash
# Type checking is handled by pre-commit hooks (mypy)
# Documentation preview
uv run pdoc py_hamt
```

## Architecture Overview

py-hamt implements a Hash Array Mapped Trie (HAMT) for IPFS/IPLD content-addressed storage. The core architecture follows this pattern:

1. **ContentAddressedStore (CAS)** - Abstract storage layer (store.py)
   - `KuboCAS` - IPFS/Kubo implementation for production
   - `InMemoryCAS` - In-memory implementation for testing

2. **HAMT** - Core data structure (hamt.py)
   - Uses blake3 hashing by default
   - Implements content-addressed trie for efficient key-value storage
   - Supports async operations for large datasets

3. **ZarrHAMTStore** - Zarr integration (zarr_hamt_store.py)
   - Implements zarr.abc.store.Store interface
   - Enables storing large Zarr arrays on IPFS via HAMT
   - Keys stored verbatim, values as raw bytes

4. **Encryption Layer** - Optional encryption (encryption_hamt_store.py)
   - `SimpleEncryptedZarrHAMTStore` for fully encrypted storage

## Key Design Patterns

- All storage operations are async to handle IPFS network calls
- Content addressing means identical data gets same hash/CID
- HAMT provides O(log n) access time for large key sets
- Store abstractions allow swapping storage backends
- Type hints required throughout (mypy enforced)
- 100% test coverage required with hypothesis property-based testing

## IPFS Integration Requirements

Tests require either:
- Local IPFS daemon running (`ipfs daemon`)
- Docker available for containerized IPFS
- Neither (unit tests only, integration tests skip)

The `--ipfs` pytest flag controls IPFS test execution.
