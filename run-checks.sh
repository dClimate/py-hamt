#!/bin/bash

# Run pytest with coverage
uv run pytest --cov=py_hamt tests/

# Check coverage
uv run coverage report --fail-under=100 --show-missing

# Check linting with ruff
uv run ruff check

# Auto format with ruff
uv run ruff format
