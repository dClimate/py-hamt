"""
This file can be run by pytest, but is not automatically included since it includes some tests that may run for a long time, and are not useful for verifying the HAMT's correctness properties.

This test suite contains various performance tests, which are meant to be run individually.
"""

import asyncio
import time

import pytest

from py_hamt import HAMT
from py_hamt.store import InMemoryCAS


@pytest.mark.asyncio
async def test_large_kv_set() -> None:
    """This test is meant for finding whether the HAMT performance scales linearly with increasing set size, an issue with HAMT v2.
    Feel free to tune and run the LARGE_KV_SET_SIZE variable as needed for gathering the different timepoints.
    """
    LARGE_KV_SET_SIZE: int = 1_000_000

    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas)
    start: float = time.perf_counter()
    await asyncio.gather(
        *[hamt.set(str(k_int), k_int) for k_int in range(LARGE_KV_SET_SIZE)]
    )
    await hamt.make_read_only()
    end: float = time.perf_counter()
    elapsed: float = end - start
    print(f"Took {elapsed:.2f} seconds")
    assert (
        len([key async for key in hamt.keys()])
        == (await hamt.len())
        == LARGE_KV_SET_SIZE
    )
    for k_int in range(LARGE_KV_SET_SIZE):
        assert (await hamt.get(str(k_int))) == k_int
