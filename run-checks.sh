#!/bin/bash

# Run pytest with coverage
uv run pytest --ipfs --cov=py_hamt tests/

# Check coverage
uv run coverage report --fail-under=100 --show-missing

# Check for linting, formatting, and type checking using the pre-commit hooks found in .pre-commit-config.yaml
uv run pre-commit run --all-files --show-diff-on-failure
