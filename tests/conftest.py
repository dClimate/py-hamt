import httpx
import pytest

# re-export helpers so existing imports keep working (optional)
from testing_utils import create_ipfs, ipld_strategy  # noqa: F401


@pytest.fixture(scope="session")
async def global_client_session():
    """One httpx.AsyncClient shared by the whole test run."""
    async with httpx.AsyncClient() as client:
        yield client
    # httpx's async context manager awaits client.aclose() for us


def pytest_addoption(parser):
    parser.addoption(
        "--ipfs",
        action="store_true",
        default=False,
        help="run tests that require a Kubo daemon",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ipfs: tests that need a live IPFS node")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--ipfs"):
        return  # user explicitly asked → run them
    skip = pytest.mark.skip(reason="needs --ipfs to run")
    for item in items:
        if "ipfs" in item.keywords:
            item.add_marker(skip)
