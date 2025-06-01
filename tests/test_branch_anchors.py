# tests/test_branch_anchors.py
import pytest
from py_hamt import HAMT
from py_hamt.store import InMemoryCAS


@pytest.mark.asyncio
async def test_force_every_branch():
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, max_bucket_size=1)
    for k in ("5", "15", "123"):
        await hamt.set(k, b"x")

    await hamt.cache_size()
    await hamt.cache_vacate()  # writes -> vacate(write)
    await hamt.make_read_only()  # flush -> vacate(read)
    await hamt.get("5")  # _get_pointer follows link
