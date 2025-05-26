import aiohttp
import pytest

# re-export helpers so existing imports keep working (optional)
from testing_utils import ipld_strategy, create_ipfs  # noqa: F401


@pytest.fixture(scope="session")
async def global_client_session():
    """One aiohttp.ClientSession shared by the whole test run."""
    async with aiohttp.ClientSession() as session:
        yield session
    # aiohttp’s async context manager awaits session.close() for us
