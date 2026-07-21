import numpy as np
import pytest
import xarray as xr
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore


@pytest.mark.asyncio
async def test_resize_snapshot_pins_freshly_loaded_shard() -> None:
    """A freshly-fetched resize shard must be pinned against concurrent eviction.

    With an over-budget cache whose other entries are dirty (so unevictable),
    the just-loaded clean shard is the only eviction candidate. Without a pin it
    is dropped between ``put`` and the follow-up ``get`` inside the snapshot,
    raising a spurious ``RuntimeError``. Pinning across fetch+read prevents it.
    """
    ds = xr.Dataset(
        {"temp": (["x"], np.arange(4.0))},
        coords={"x": np.arange(4)},
    ).chunk({"x": 4})
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4,),
        chunk_shape=(4,),
        chunks_per_shard=1,
    )
    ds.to_zarr(store=store, mode="w")
    root_cid = await store.flush()

    store = await ShardedZarrStore.open(cas=cas, read_only=False, root_cid=root_cid)
    array_index = store.array_indices[""]
    assert array_index.num_shards >= 1
    assert array_index.shard_cids[0] is not None

    cache = store._shard_data_cache
    await cache.clear()
    # Squeeze the budget and keep a dirty (unevictable) entry resident so the
    # cache is permanently over budget: the snapshot's fresh shard becomes the
    # sole eviction candidate.
    cache.max_memory_bytes = 1
    await cache.put(9999, [None], is_dirty=True)

    snapshot = await store._snapshot_shards_for_resize(array_index)

    assert 0 in snapshot
    assert len(snapshot[0]) == array_index.chunks_per_shard
