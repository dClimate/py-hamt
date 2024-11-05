<p align="center">
<a href="https://dclimate.net/" target="_blank" rel="noopener noreferrer">
<img width="50%" src="https://user-images.githubusercontent.com/41392423/173133333-79ef15d0-6671-4be3-ac97-457344e9e958.svg" alt="dClimate logo">
</a>
</p>

# py-hamt
This is a python implementation of a HAMT, adapted from [rvagg's IAMap project written in JavaScript](https://github.com/rvagg/iamap).
The HAMT, like IAMap, also abstracts over a backing storage layer. This repository provides two of them:
1. python dictionary which is meant for testing purposes
2. storage on IPLD by interfacing with an ipfs daemon

The functions/classes in this repo are meant to be one-to-one mappings to those in the JS one. There are some minor modifications made however. As a result, the JS code can serve as a canonical guide to implementation and functionality, but this repository aims to be independently understandable.

# Installation
We do not publish this package to PyPI. To add this library to your project, install directly from git. e.g.
```sh
pip install git+https://github.com/dClimate/py-hamt
```

# Motivation
dClimate forked and contributed to this library since we use HAMTs as a [zarr](https://zarr.dev/) storage backend. We construct these HAMTs using IPLD objects, allowing us to integrate Zarrs into the decentralized ecosystem. HAMTs allow us to have efficient querying of Zarrs without storing prohibitively large manifest files which can reach into 10-100s of MBs.

To see `py-hamt` in action, see [ipldstore](https://github.com/dClimate/ipldstore).

# Development Guide
## Setting Up
`py-hamt` uses [uv](https://docs.astral.sh/uv/) for project management. Make sure you install that first.
Once uv is installed, run
```sh
uv sync
```
to create the project virtual environment at `.venv` based on the lockfile `uv.lock`. Don't worry about activating this virtual environment to run tests or formatting and linting, uv will automatically take care of that.

## Running tests
We use `pytest` as our testing framework.
To run the test suite, call uv run with pytest with code coverage.
```sh
uv run pytest --cov=py_hamt tests/
uv run coverage report --fail-under=100 --show-missing
```

## Profiling
We use python's native cProfile for generating profiling information and snakeviz for doing the visualization. Profiling the tests is how we measure usage, since the tests are supposed have complete code coverage anyhow.

Creating the profile requires manual activation of the virtual environment.
```sh
source .venv/bin/activate
python -m cProfile -o profile.prof -m pytest
```

Running the profile viewer can be invoked from uv directly however.
```sh
uv run snakeviz .
```


## Formatting and Linting
We use `ruff` for formatting and linting, and for simplicity's sake just use all of its default settings.
To format
```sh
uv run ruff format
```
To run linting checks
```sh
uv run ruff check
```

## Generating and viewing documentation
`py-hamt` uses [pdoc](https://pdoc.dev/) for its ease of use. On pushes to main the docs folder will be deployed to the repository's GitHub pages, and PRs will contain preview deployments.

To see a documentation preview on your local machine, run
```sh
uv run pdoc
```

## Managing dependencies
Use `uv add` and `uv remove`, e.g. `uv add numpy`. For more information please see the [uv documentation](https://docs.astral.sh/uv/guides/projects/).
