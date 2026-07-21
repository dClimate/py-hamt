from typing import Optional

import pytest
from dag_cbor.ipld import IPLDKind

from py_hamt import HAMT, InMemoryCAS

# These keys share the first TWO blake3 hash bytes, forcing a depth-2 subtree;
# "s45" shares only the first byte, so it sits in a bucket of the depth-1 node
# whose collapse must load the sibling subtree from the CAS.
DEEP_KEYS = ("1313", "5428", "11835", "17462", "20814")
SHALLOW_KEY = "s45"
ALL_KEYS = (*DEEP_KEYS, SHALLOW_KEY)


class FlakyCAS(InMemoryCAS):
    """Fails the Nth load after arming, so tests can sweep every load index a
    delete performs — from the root load through the collapse sibling loads."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False
        self.loads_before_failure = 0

    async def load(
        self,
        node_id: IPLDKind,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        suffix: Optional[int] = None,
    ) -> bytes:
        if self.armed:
            if self.loads_before_failure <= 0:
                raise ConnectionError("simulated CAS load failure")
            self.loads_before_failure -= 1
        return await super().load(node_id, offset=offset, length=length, suffix=suffix)


async def _vacated_hamt(cas: FlakyCAS) -> HAMT:
    hamt = await HAMT.build(cas=cas)
    for key in ALL_KEYS:
        await hamt.set(key, key.encode())
    await hamt.cache_vacate()
    return hamt


@pytest.mark.asyncio
async def test_failed_delete_preserves_all_keys() -> None:
    """The collapse-phase CAS failure: walk loads are warmed into the cache, so
    the first load delete performs is the sibling-subtree load that runs after
    the bucket mutation in the pre-fix code."""
    cas = FlakyCAS()
    hamt = await _vacated_hamt(cas)
    assert await hamt.get(SHALLOW_KEY) == SHALLOW_KEY.encode()

    cas.armed = True
    with pytest.raises(ConnectionError):
        await hamt.delete(SHALLOW_KEY)
    cas.armed = False

    for key in ALL_KEYS:
        assert await hamt.get(key) == key.encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("loads_before_failure", range(6))
async def test_delete_is_atomic_at_every_load_index(
    loads_before_failure: int,
) -> None:
    """Sweep the failure across every CAS load a cold delete performs — the
    walk loads (pre-mutation) and the collapse prefetch loads alike. Whenever
    delete raises, every key must remain readable; once the budget exceeds the
    loads a delete needs, the delete must succeed."""
    cas = FlakyCAS()
    hamt = await _vacated_hamt(cas)

    cas.armed = True
    cas.loads_before_failure = loads_before_failure
    try:
        await hamt.delete(SHALLOW_KEY)
    except ConnectionError:
        cas.armed = False
        for key in ALL_KEYS:
            assert await hamt.get(key) == key.encode()
    else:
        cas.armed = False
        with pytest.raises(KeyError):
            await hamt.get(SHALLOW_KEY)
        for key in DEEP_KEYS:
            assert await hamt.get(key) == key.encode()
