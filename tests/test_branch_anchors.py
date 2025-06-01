# tests/test_branch_anchors.py
import pytest
from py_hamt import HAMT
from py_hamt.store import InMemoryCAS


@pytest.mark.asyncio
async def test_force_every_branch():
    cas = InMemoryCAS()
    # 1 → guaranteed bucket overflow so we create links, etc.
    hamt = await HAMT.build(cas=cas, max_bucket_size=1)

    # Three colliding keys              (covers children_in_memory / recurse)
    for k in ("5", "15", "123"):
        await hamt.set(k, b"x")

    # ------------------------------------------------------------------
    # A.  write-mode vacate
    # ------------------------------------------------------------------
    await hamt.cache_vacate()  # write-mode branch

    # ------------------------------------------------------------------
    # B.  read-cache vacate
    # ------------------------------------------------------------------
    await hamt.make_read_only()  # flushes tree, swaps store

    # Build up the read-cache by touching one key
    await hamt.get("5")

    # Now vacate the *read* cache
    await hamt.cache_vacate()

    # quick sanity-check: cache really is empty
    assert await hamt.cache_size() == 0
