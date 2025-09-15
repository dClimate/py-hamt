import asyncio

import dag_cbor
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
async def test_graft_store_success(create_ipfs: tuple[str, str]):
    """Tests successful grafting of a source store onto a target store."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize source store
        source_shape = (20, 20)
        chunk_shape = (10, 10)
        chunks_per_shard = 4
        source_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=source_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )

        # Write some chunk data to source store
        proto = zarr.core.buffer.default_buffer_prototype()
        chunk_key = "temp/c/0/0"
        chunk_data = b"test_source_data"
        await source_store.set(chunk_key, proto.buffer.from_bytes(chunk_data))
        source_root_cid = await source_store.flush()

        # Initialize target store with larger shape to accommodate graft
        target_shape = (40, 20)
        target_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=target_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )

        # Graft source store onto target store at offset (1, 0)
        chunk_offset = (1, 0)
        await target_store.graft_store(source_root_cid, chunk_offset)

        # Verify grafted data
        grafted_chunk_key = "temp/c/1/0"  # Offset (1,0) corresponds to chunk (1,0)
        assert await target_store.exists(grafted_chunk_key)
        grafted_data = await target_store.get(grafted_chunk_key, proto)
        assert grafted_data is not None
        assert grafted_data.to_bytes() == chunk_data

        # Verify original chunk position in source store is not present in target
        assert not await target_store.exists("temp/c/0/0")

        # Verify target store's geometry unchanged
        assert target_store._array_shape == target_shape
        assert target_store._chunks_per_dim == (4, 2)  # ceil(40/10) = 4
        assert target_store._total_chunks == 8  # 4 * 2
        assert target_store._num_shards == 2  # ceil(8/4) = 2
        assert (
            target_store._shard_data_cache._dirty_shards
        )  # Grafting marks shards as dirty

        # Flush and verify persistence
        target_root_cid = await target_store.flush()
        target_store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=target_root_cid
        )
        assert await target_store_read.exists(grafted_chunk_key)
        read_data = await target_store_read.get(grafted_chunk_key, proto)
        assert read_data is not None
        assert read_data.to_bytes() == chunk_data
        assert target_store_read._array_shape == target_shape


@pytest.mark.asyncio
async def test_graft_store_with_dataset(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """Tests grafting a store containing a full dataset."""
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset
    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Write dataset to source store
        source_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=source_store, mode="w")
        source_root_cid = await source_store.flush()

        # Initialize target store with larger shape
        target_shape = (
            array_shape_tuple[0] + 20,
            array_shape_tuple[1],
            array_shape_tuple[2],
        )
        target_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=target_shape,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )

        # Graft source store at offset (1, 0, 0)
        chunk_offset = (1, 0, 0)
        await target_store.graft_store(source_root_cid, chunk_offset)

        # Verify grafted chunk data
        proto = zarr.core.buffer.default_buffer_prototype()
        source_chunk_key = "temp/c/0/0/0"
        target_chunk_key = "temp/c/1/0/0"  # Offset by 1 in time dimension
        assert await target_store.exists(target_chunk_key)
        source_data = await source_store.get(source_chunk_key, proto)
        target_data = await target_store.get(target_chunk_key, proto)
        assert source_data is not None
        assert target_data is not None
        assert source_data.to_bytes() == target_data.to_bytes()

        # Verify metadata is not grafted
        assert not await target_store.exists("temp/zarr.json")

        # Flush and verify persistence
        target_root_cid = await target_store.flush()
        target_store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=target_root_cid
        )
        assert await target_store_read.exists(target_chunk_key)
        read_data = await target_store_read.get(target_chunk_key, proto)
        assert read_data is not None
        assert read_data.to_bytes() == source_data.to_bytes()


@pytest.mark.asyncio
async def test_graft_store_empty_source(create_ipfs: tuple[str, str]):
    """Tests grafting an empty source store."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize empty source store
        source_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        source_root_cid = await source_store.flush()

        # Initialize target store
        target_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(40, 40),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Graft empty source store
        await target_store.graft_store(source_root_cid, chunk_offset=(1, 1))

        # Verify no chunks were grafted
        assert not await target_store.exists("temp/c/1/1")
        assert (
            not target_store._shard_data_cache._dirty_shards
        )  # No shards marked dirty since no changes

        # Flush and verify
        target_root_cid = await target_store.flush()
        target_store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=target_root_cid
        )
        assert not await target_store_read.exists("temp/c/1/1")


@pytest.mark.asyncio
async def test_graft_store_invalid_cases(create_ipfs: tuple[str, str]):
    """Tests error handling in graft_store."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize target store
        target_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(40, 40),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Test read-only target store
        target_store_read_only = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=True,
            root_cid=await target_store.flush(),
        )
        with pytest.raises(
            PermissionError, match="Cannot graft onto a read-only store"
        ):
            await target_store_read_only.graft_store("some_cid", chunk_offset=(0, 0))

        # Test invalid source CID
        # invalid_cid = "invalid_cid"
        # with pytest.raises(ValueError, match="Store to graft could not be loaded"):
        #     await target_store.graft_store(invalid_cid, chunk_offset=(0, 0))

        # Test source store with invalid configuration
        invalid_root_obj = {
            "manifest_version": "sharded_zarr_v1",
            "metadata": {},
            "chunks": {
                "array_shape": [10, 10],
                "chunk_shape": [5, 5],
                "sharding_config": {"chunks_per_shard": 4},
                "shard_cids": [None] * 4,
            },
        }
        invalid_root_cid = await kubo_cas.save(
            dag_cbor.encode(invalid_root_obj), codec="dag-cbor"
        )
        with pytest.raises(ValueError, match="Inconsistent number of shards"):
            await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=invalid_root_cid
            )
        # source_store._chunks_per_dim = None  # Simulate unconfigured store
        # with pytest.raises(ValueError, match="Inconsistent number of shards"):
        #     await target_store.graft_store(invalid_root_cid, chunk_offset=(0, 0))

        # Test grafting out-of-bounds offset
        source_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        # Write some data to source store
        proto = zarr.core.buffer.default_buffer_prototype()
        await source_store.set("temp/c/0/0", proto.buffer.from_bytes(b"data"))
        source_root_cid = await source_store.flush()
        with pytest.raises(ValueError, match="Shard index 10 out of bounds."):
            await target_store.graft_store(
                source_root_cid, chunk_offset=(10, 0)
            )  # Out of bounds for target (4x4 chunks)


@pytest.mark.asyncio
async def test_graft_store_concurrency(create_ipfs: tuple[str, str]):
    """Tests concurrent graft_store operations to ensure shard locking works."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize source stores
        source_shape = (20, 20)
        chunk_shape = (10, 10)
        chunks_per_shard = 4
        source_store1 = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=source_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )
        proto = zarr.core.buffer.default_buffer_prototype()
        await source_store1.set("temp/c/0/0", proto.buffer.from_bytes(b"data1"))
        source_cid1 = await source_store1.flush()

        source_store2 = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=source_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )
        await source_store2.set("temp/c/0/0", proto.buffer.from_bytes(b"data2"))
        source_cid2 = await source_store2.flush()

        # Initialize target store
        target_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(40, 40),
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )

        # Define graft tasks
        async def graft_task(cid, offset):
            await target_store.graft_store(cid, chunk_offset=offset)

        # Run concurrent grafts
        tasks = [
            graft_task(source_cid1, (1, 1)),
            graft_task(source_cid2, (2, 2)),
        ]
        await asyncio.gather(*tasks)

        # Verify grafted data
        assert await target_store.exists("temp/c/1/1")
        assert await target_store.exists("temp/c/2/2")
        data1 = await target_store.get("temp/c/1/1", proto)
        data2 = await target_store.get("temp/c/2/2", proto)
        assert data1 is not None
        assert data2 is not None
        assert data1.to_bytes() in [b"data1", b"data2"]
        assert data2.to_bytes() in [b"data1", b"data2"]
        assert data1.to_bytes() != data2.to_bytes()  # Ensure distinct data

        # Flush and verify persistence
        target_root_cid = await target_store.flush()
        target_store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=target_root_cid
        )
        assert await target_store_read.exists("temp/c/1/1")
        assert await target_store_read.exists("temp/c/2/2")


@pytest.mark.asyncio
async def test_graft_store_overlapping_chunks(create_ipfs: tuple[str, str]):
    """Tests grafting when target already has data at some chunk positions."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize source store
        source_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        proto = zarr.core.buffer.default_buffer_prototype()
        source_chunk_key = "temp/c/0/0"
        source_data = b"source_data"
        await source_store.set(source_chunk_key, proto.buffer.from_bytes(source_data))
        source_root_cid = await source_store.flush()

        # Initialize target store with some existing data
        target_store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(40, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        target_chunk_key = "temp/c/1/1"
        existing_data = b"existing_data"
        await target_store.set(target_chunk_key, proto.buffer.from_bytes(existing_data))

        # Graft source store at offset (1, 0)
        await target_store.graft_store(source_root_cid, chunk_offset=(1, 0))

        # Verify that existing data was not overwritten
        read_data = await target_store.get(target_chunk_key, proto)
        assert read_data is not None
        assert read_data.to_bytes() == existing_data
        assert (
            target_store._shard_data_cache._dirty_shards
        )  # Shard is marked dirty due to attempted write

        # Verify other grafted chunks
        grafted_chunk_key = "temp/c/1/0"  # Corresponds to source (0,0) at offset (1,0)
        assert await target_store.exists(grafted_chunk_key)
        read_data = await target_store.get(grafted_chunk_key, proto)
        assert read_data is not None
        assert read_data.to_bytes() == source_data

        # Flush and verify
        target_root_cid = await target_store.flush()
        target_store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=target_root_cid
        )
        targeted_target_chunk = await target_store_read.get(target_chunk_key, proto)
        assert targeted_target_chunk is not None
        assert targeted_target_chunk.to_bytes() == existing_data

        grafted_chunk_data = await target_store_read.get(grafted_chunk_key, proto)
        assert grafted_chunk_data is not None
        assert grafted_chunk_data.to_bytes() == source_data
