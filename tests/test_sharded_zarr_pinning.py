import asyncio
import aiohttp
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from py_hamt import KuboCAS, ShardedZarrStore


# Helper function to query the IPFS daemon for all pinned CIDs
async def get_pinned_cids(rpc_base_url: str) -> set[str]:
    """Queries the Kubo RPC API and returns a set of all pinned CIDs."""
    url = f"{rpc_base_url}/api/v0/pin/ls"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, params={"type": "all"}) as resp:
                resp.raise_for_status()
                data = await resp.json()
                return set(data.get("Keys", {}).keys())
    except Exception as e:
        pytest.fail(f"Failed to query pinned CIDs from Kubo RPC API: {e}")
        return set()


# Helper function to gather all CIDs from a store instance
async def get_all_dataset_cids(store: ShardedZarrStore) -> set[str]:
    """Helper to collect all CIDs associated with a ShardedZarrStore instance."""
    if store._root_obj is None or store._cid_len is None:
        raise RuntimeError("Store is not properly initialized.")

    cids = set()
    if store._root_cid:
        cids.add(store._root_cid)

    # Gather metadata CIDs
    for cid in store._root_obj.get("metadata", {}).values():
        if cid:
            cids.add(cid)

    # Gather shard and all chunk CIDs within them
    for shard_cid in store._root_obj["chunks"]["shard_cids"]:
        if not shard_cid:
            continue
        cids.add(str(shard_cid))
        try:
            # Load shard data to find the chunk CIDs within
            shard_data = await store.cas.load(shard_cid)
            for i in range(0, len(shard_data), store._cid_len):
                cid_bytes = shard_data[i : i + store._cid_len]
                if all(b == 0 for b in cid_bytes):  # Skip null/empty CID slots
                    continue

                chunk_cid_str = cid_bytes.decode("ascii").rstrip("\x00")
                if chunk_cid_str:
                    cids.add(chunk_cid_str)
        except Exception as e:
            print(f"Warning: Could not load shard {shard_cid} to gather its CIDs: {e}")

    return cids


@pytest.fixture(scope="module")
def random_zarr_dataset_for_pinning():
    """Creates a random xarray Dataset specifically for the pinning test."""
    times = pd.date_range("2025-01-01", periods=50)
    lats = np.linspace(-90, 90, 10)
    lons = np.linspace(-180, 180, 20)

    temp = np.random.randn(len(times), len(lats), len(lons))

    ds = xr.Dataset(
        {"temp": (["time", "lat", "lon"], temp)},
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # Define chunking for the store
    ds = ds.chunk({"time": 10, "lat": 10, "lon": 20})
    yield ds


@pytest.mark.asyncio
async def test_sharded_zarr_store_pinning(
    create_ipfs: tuple[str, str], random_zarr_dataset_for_pinning: xr.Dataset
):
    """
    Tests the pin_entire_dataset and unpin_entire_dataset methods.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset_for_pinning

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # --- 1. Write dataset to the store ---
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=1,  # Use a smaller number to ensure multiple shards
        )
        test_ds.to_zarr(store=store, mode="w", consolidated=True)
        root_cid = await store.flush()
        assert root_cid is not None

        # --- 2. Gather all expected CIDs from the written store ---
        expected_cids = await get_all_dataset_cids(store)
        assert len(expected_cids) > 5  # Sanity check: ensure we have CIDs to test

        # --- 3. Pin the dataset and verify ---
        await store.pin_entire_dataset(target_rpc=rpc_base_url)

        # Allow a moment for pins to register
        await asyncio.sleep(1)

        currently_pinned = await get_pinned_cids(rpc_base_url)

        # Check if all our dataset's CIDs are in the main pin list
        missing_pins = expected_cids - currently_pinned
        assert not missing_pins, (
            f"The following CIDs were expected to be pinned but were not: {missing_pins}"
        )

        # --- 4. Unpin the dataset and verify ---
        await store.unpin_entire_dataset(target_rpc=rpc_base_url)

        # Allow a moment for pins to be removed
        await asyncio.sleep(1)

        pinned_after_unpin = await get_pinned_cids(rpc_base_url)

        # Check that none of our dataset's CIDs are in the pin list anymore
        lingering_pins = expected_cids.intersection(pinned_after_unpin)
        assert not lingering_pins, (
            f"The following CIDs were expected to be unpinned but still exist: {lingering_pins}"
        )
