import asyncio
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.core.buffer

from py_hamt import KuboCAS, ShardedZarrStore


@pytest.fixture(scope="module")
def random_zarr_dataset():
    """Creates a random xarray Dataset for benchmarking."""
    # Using a slightly larger dataset for a more meaningful benchmark
    times = pd.date_range("2024-01-01", periods=100)
    lats = np.linspace(-90, 90, 18)
    lons = np.linspace(-180, 180, 36)

    temp = np.random.randn(len(times), len(lats), len(lons))

    ds = xr.Dataset(
        {
            "temp": (["time", "lat", "lon"], temp),
        },
        coords={"time": times, "lat": lats, "lon": lons},
    )

    # Define chunking for the store
    ds = ds.chunk({"time": 20, "lat": 18, "lon": 36})
    yield ds


@pytest.mark.asyncio
async def test_delete_chunk_success(create_ipfs: tuple[str, str]):
    """Tests successful deletion of a chunk from the store."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize store
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Write a chunk
        chunk_key = "temp/c/0/0"
        chunk_data = b"test_chunk_data"
        proto = zarr.core.buffer.default_buffer_prototype()
        await store.set(chunk_key, proto.buffer.from_bytes(chunk_data))
        assert await store.exists(chunk_key)

        # Delete the chunk
        await store.delete(chunk_key)

        # Verify chunk is deleted in cache and shard is marked dirty
        linear_index = store._get_linear_chunk_index((0, 0))
        shard_idx, index_in_shard = store._get_shard_info(linear_index)
        target_shard_list = await store._load_or_initialize_shard_cache(shard_idx)
        assert target_shard_list[index_in_shard] is None
        assert shard_idx in store._shard_data_cache._dirty_shards

        # Flush and verify persistence
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert not await store_read.exists(chunk_key)
        assert await store_read.get(chunk_key, proto) is None


@pytest.mark.asyncio
async def test_delete_metadata_success(create_ipfs: tuple[str, str]):
    """Tests successful deletion of a metadata key."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize store
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Write metadata
        metadata_key = "temp/zarr.json"
        metadata = json.dumps({"shape": [20, 20], "dtype": "float32"}).encode("utf-8")
        proto = zarr.core.buffer.default_buffer_prototype()
        await store.set(metadata_key, proto.buffer.from_bytes(metadata))
        assert await store.exists(metadata_key)

        # Delete metadata
        await store.delete(metadata_key)

        # Verify metadata is deleted and root is marked dirty
        assert metadata_key not in store._root_obj["metadata"]
        assert store._dirty_root is True

        # Flush and verify persistence
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert not await store_read.exists(metadata_key)
        assert await store_read.get(metadata_key, proto) is None


@pytest.mark.asyncio
async def test_delete_nonexistent_key(create_ipfs: tuple[str, str]):
    """Tests deletion of a nonexistent metadata key."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize store
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Temp write to temp/c/0/0 to ensure it exists
        proto = zarr.core.buffer.default_buffer_prototype()
        await store.set("temp/c/0/0", proto.buffer.from_bytes(b"test_data"))

        # flush it
        await store.flush()
        assert not store._shard_data_cache._dirty_shards  # No dirty shards after flush

        # Try to delete nonexistent metadata key
        with pytest.raises(KeyError, match="Metadata key 'nonexistent.json' not found"):
            await store.delete("nonexistent.json")

        # Try to delete nonexistent chunk key (out of bounds)
        with pytest.raises(IndexError, match="Chunk coordinate"):
            await store.delete("temp/c/3/0")  # Out of bounds for 2x2 chunk grid

        # Try to delete nonexistent chunk key (within bounds but not set)
        await store.delete("temp/c/0/0")  # Should not raise, as it sets to None
        assert not await store.exists("temp/c/0/0")
        assert (
            store._shard_data_cache._dirty_shards
        )  # Shard is marked dirty even if chunk was already None


@pytest.mark.asyncio
async def test_delete_read_only_store(create_ipfs: tuple[str, str]):
    """Tests deletion attempt on a read-only store."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize writable store and add data
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        chunk_key = "temp/c/0/0"
        proto = zarr.core.buffer.default_buffer_prototype()
        await store_write.set(chunk_key, proto.buffer.from_bytes(b"test_data"))
        metadata_key = "temp/zarr.json"
        await store_write.set(
            metadata_key, proto.buffer.from_bytes(b'{"shape": [20, 20]}')
        )
        root_cid = await store_write.flush()

        # Open as read-only
        store_read_only = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )

        # Try to delete chunk
        with pytest.raises(
            PermissionError, match="Cannot delete from a read-only store"
        ):
            await store_read_only.delete(chunk_key)

        # Try to delete metadata
        with pytest.raises(
            PermissionError, match="Cannot delete from a read-only store"
        ):
            await store_read_only.delete(metadata_key)


@pytest.mark.asyncio
async def test_delete_concurrency(create_ipfs: tuple[str, str]):
    """Tests concurrent delete operations to ensure shard locking works."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize store
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Write multiple chunks
        proto = zarr.core.buffer.default_buffer_prototype()
        chunk_keys = ["temp/c/0/0", "temp/c/1/0", "temp/c/0/1"]
        for key in chunk_keys:
            await store.set(key, proto.buffer.from_bytes(f"data_{key}".encode("utf-8")))
            assert await store.exists(key)

        # Define concurrent delete tasks
        async def delete_task(key):
            await store.delete(key)

        # Run concurrent deletes
        tasks = [delete_task(key) for key in chunk_keys]
        await asyncio.gather(*tasks)

        # Verify all chunks are deleted
        for key in chunk_keys:
            assert not await store.exists(key)
            assert await store.get(key, proto) is None

        # Verify shards are marked dirty
        assert (
            store._shard_data_cache._dirty_shards
        )  # At least one shard should be dirty

        # Flush and verify persistence
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        for key in chunk_keys:
            assert not await store_read.exists(key)
            assert await store_read.get(key, proto) is None


@pytest.mark.asyncio
async def test_delete_with_dataset(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """Tests deletion of chunks and metadata in a store with a full dataset."""
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset
    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Write dataset
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store, mode="w")
        root_cid = await store.flush()

        # Re-open store
        store = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=False, root_cid=root_cid
        )

        # Delete a chunk
        chunk_key = "temp/c/0/0/0"
        assert await store.exists(chunk_key)
        await store.delete(chunk_key)
        assert not await store.exists(chunk_key)

        # Delete metadata
        metadata_key = "temp/zarr.json"
        assert await store.exists(metadata_key)
        await store.delete(metadata_key)
        assert not await store.exists(metadata_key)

        # Flush and verify
        new_root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=new_root_cid
        )
        assert not await store_read.exists(chunk_key)
        assert not await store_read.exists(metadata_key)

        # Verify other data remains intact
        other_chunk_key = "temp/c/1/0/0"
        assert await store_read.exists(other_chunk_key)


@pytest.mark.asyncio
async def test_supports_writes_property(create_ipfs: tuple[str, str]):
    """Tests the supports_writes property."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test writable store
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        assert store_write.supports_writes is True

        # Test read-only store
        root_cid = await store_write.flush()
        store_read_only = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert store_read_only.supports_writes is False


@pytest.mark.asyncio
async def test_supports_partial_writes_property(create_ipfs: tuple[str, str]):
    """Tests the supports_partial_writes property."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test for both read-only and writable stores
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        assert store_write.supports_partial_writes is False

        root_cid = await store_write.flush()
        store_read_only = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert store_read_only.supports_partial_writes is False
