import warnings

import pytest

from py_hamt.sharded_zarr_store import MemoryBoundedLRUCache


@pytest.mark.asyncio
async def test_memory_bounded_lru_cache_contains_uses_in_operator() -> None:
    cache = MemoryBoundedLRUCache(max_memory_bytes=1_024)
    present_key = ("present", 0)
    absent_key = ("some", 0)

    await cache.put(present_key, [None])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"coroutine .* was never awaited",
            category=RuntimeWarning,
        )
        assert (present_key in cache) is True
        assert (absent_key in cache) is False
