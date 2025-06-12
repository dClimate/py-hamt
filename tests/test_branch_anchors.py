# tests/test_branch_anchors.py
import pytest

from py_hamt import HAMT, InMemoryCAS, blake3_hashfn


def find_colliding_keys(count: int = 3) -> list[str]:
    """Find *count* decimal strings whose Blake3 hashes share byte[0]."""
    chosen: list[str] = []
    wanted_byte: int | None = None
    i: int = 0

    while len(chosen) < count:
        k = str(i)
        b0 = blake3_hashfn(k.encode())[0]
        if wanted_byte is None or b0 == wanted_byte:
            wanted_byte = b0
            chosen.append(k)
        i += 1
    return chosen


@pytest.mark.asyncio
async def test_force_every_branch():
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, max_bucket_size=1)

    # ------------------------------------------------------------------
    # 1. write some colliding keys  → guarantees in-buffer links
    # ------------------------------------------------------------------
    keys = find_colliding_keys()  # save for later cache hit
    for k in keys:
        await hamt.set(k, b"x")

    # call cache_size() while still in WRITE mode
    assert await hamt.cache_size() > 0  # executes lines 207-208

    # ------------------------------------------------------------------
    # 2. write-mode cache_vacate()  → covers HAMT.cache_vacate() write branch
    # ------------------------------------------------------------------
    await hamt.cache_vacate()

    # ------------------------------------------------------------------
    # 3. switch to READ-ONLY, touch cache once, vacate again
    #    → covers ReadCacheStore.vacate() (line 177)
    # ------------------------------------------------------------------
    await hamt.make_read_only()
    await hamt.get(keys[0])  # populates read cache
    await hamt.cache_vacate()
    assert await hamt.cache_size() == 0
