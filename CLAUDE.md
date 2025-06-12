# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development Setup
```bash
# Install dependencies and set up the virtual environment
uv sync
source .venv/bin/activate
pre-commit install
```

### Running Tests
```bash
# Run tests with coverage (requires local IPFS daemon or Docker)
pytest --ipfs --cov=py_hamt tests/

# Run tests without IPFS integration
pytest

# Run all checks (tests, coverage, linting, formatting)
bash run-checks.sh
```

### Code Quality Checks
```bash
# Run all pre-commit hooks
pre-commit run --all-files

# Run linting
ruff check .

# Run formatting
ruff format .

# Run type checking
mypy .
```

### Documentation
```bash
# Generate and view documentation
pdoc py_hamt
```

### Profiling
```bash
# CPU profiling
python -m cProfile -o profile.prof -m pytest

# View CPU profile
snakeviz profile.prof

# Memory profiling
python -m memray run -m pytest

# View memory profile
memray flamegraph <memray-output-file>
```

## Code Architecture

### Core Components

1. **HAMT (Hash Array Mapped Trie)**
   - `py_hamt/hamt.py` - Efficient key-value storage structure for content-addressed systems
   - Core data structure for mapping arbitrary strings to values in a content-addressed system
   - Uses blake3 hashing by default

2. **ContentAddressedStore**
   - `py_hamt/store.py` - Abstract interface for content-addressed storage
   - `KuboCAS` implementation connects to IPFS Kubo daemon
   - Manages async HTTP sessions for communication with IPFS

3. **Zarr Integration**
   - `py_hamt/zarr_hamt_store.py` - Implements Zarr v3 storage interface
   - Allows storing/retrieving Zarr arrays on content-addressed storage
   - `SimpleEncryptedZarrHAMTStore` provides encryption for Zarr data

### Key Design Patterns

1. **Async-first Architecture**
   - All operations use async/await for non-blocking I/O
   - Careful session management for multi-loop safety
   - Proper resource cleanup via context managers and `aclose()` methods

2. **Memory Management**
   - Read caching for performance optimization
   - Vacate mechanisms to clear memory when needed
   - Configurable buffer sizes for HAMT nodes

3. **Content Addressing**
   - Uses IPLD data model with dag-cbor encoding
   - Multiformat CIDs for content identification
   - Efficient serialization and linking of HAMT nodes

## Important Behaviors and Patterns

1. **HAMT Read/Write Modes**
   - The HAMT can be in either read+write mode or read-only mode
   - In read+write mode, the root node ID is not valid until `make_read_only()` is called
   - Use `await hamt.make_read_only()` to flush changes and obtain the valid root node ID

2. **Session Management**
   - KuboCAS manages aiohttp ClientSessions per event loop
   - Always use `async with` or explicitly call `await cas.aclose()` to clean up resources
   - Session cleanup is critical to prevent resource leaks

3. **Zarr Store Pattern**
   - Ensure the HAMT's `values_are_bytes` is set to `True` for Zarr operations
   - Match the HAMT's read_only status with the ZarrHAMTStore's read_only parameter
   - For encryption, use the SimpleEncryptedZarrHAMTStore class

4. **Testing with IPFS**
   - Tests require either a local IPFS daemon or Docker
   - Use the `--ipfs` flag to enable IPFS integration tests
   - When Docker is available, tests will automatically launch a container
