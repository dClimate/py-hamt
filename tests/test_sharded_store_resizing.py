import asyncio
import json
import math

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
async def test_resize_store_success(create_ipfs: tuple[str, str]):
    """Tests successful resizing of the store's main shard index."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize store
        initial_shape = (20, 20)
        chunk_shape = (10, 10)
        chunks_per_shard = 4
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=initial_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )

        # Verify initial geometry
        assert store._array_shape == initial_shape
        assert store._chunk_shape == chunk_shape
        assert store._chunks_per_shard == chunks_per_shard
        initial_chunks_per_dim = (2, 2)  # ceil(20/10) = 2
        assert store._chunks_per_dim == initial_chunks_per_dim
        assert store._total_chunks == 4  # 2 * 2
        assert store._num_shards == 1  # ceil(4/4) = 1
        assert len(store._root_obj["chunks"]["shard_cids"]) == 1

        # Resize to a larger shape
        new_shape = (30, 30)
        await store.resize_store(new_shape=new_shape)

        # Verify updated geometry
        assert store._array_shape == new_shape
        assert store._chunks_per_dim == (3, 3)  # ceil(30/10) = 3
        assert store._total_chunks == 9  # 3 * 3
        assert store._num_shards == 3  # ceil(9/4) = 3
        assert len(store._root_obj["chunks"]["shard_cids"]) == 3
        assert store._root_obj["chunks"]["array_shape"] == list(new_shape)
        assert store._dirty_root is True

        # Verify shard cids extended correctly
        assert store._root_obj["chunks"]["shard_cids"][1] is None
        assert store._root_obj["chunks"]["shard_cids"][2] is None

        # Flush and verify persistence
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert store_read._array_shape == new_shape
        assert store_read._num_shards == 3
        assert len(store_read._root_obj["chunks"]["shard_cids"]) == 3

        # Resize to a smaller shape
        smaller_shape = (10, 10)
        await store.resize_store(new_shape=smaller_shape)
        assert store._array_shape == smaller_shape
        assert store._chunks_per_dim == (1, 1)  # ceil(10/10) = 1
        assert store._total_chunks == 1  # 1 * 1
        assert store._num_shards == 1  # ceil(1/4) = 1
        assert len(store._root_obj["chunks"]["shard_cids"]) == 1
        assert store._dirty_root is True

        # Flush and verify
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert store_read._array_shape == smaller_shape
        assert store_read._num_shards == 1


@pytest.mark.asyncio
async def test_resize_store_zero_sized_array(create_ipfs: tuple[str, str]):
    """Tests resizing to/from a zero-sized array."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Initialize with zero-sized dimension
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 0),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )
        assert store._total_chunks == 0
        assert store._num_shards == 0
        assert len(store._root_obj["chunks"]["shard_cids"]) == 0

        # Resize to non-zero shape
        new_shape = (20, 20)
        await store.resize_store(new_shape=new_shape)
        assert store._array_shape == new_shape
        assert store._chunks_per_dim == (2, 2)
        assert store._total_chunks == 4
        assert store._num_shards == 1
        assert len(store._root_obj["chunks"]["shard_cids"]) == 1
        assert store._dirty_root is True

        # Resize back to zero-sized
        zero_shape = (0, 20)
        await store.resize_store(new_shape=zero_shape)
        assert store._array_shape == zero_shape
        assert store._total_chunks == 0
        assert store._num_shards == 0
        assert len(store._root_obj["chunks"]["shard_cids"]) == 0

        # Verify persistence
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert store_read._array_shape == zero_shape
        assert store_read._num_shards == 0


@pytest.mark.asyncio
async def test_resize_store_invalid_cases(create_ipfs: tuple[str, str]):
    """Tests error handling in resize_store."""
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

        # Test read-only store
        store_read_only = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=True,
            root_cid=await store.flush(),
        )
        with pytest.raises(PermissionError, match="Cannot resize a read-only store"):
            await store_read_only.resize_store(new_shape=(30, 30))

        # Test wrong number of dimensions
        with pytest.raises(
            ValueError,
            match="New shape must have the same number of dimensions as the old shape",
        ):
            await store.resize_store(new_shape=(30, 30, 30))

        # Test uninitialized store (simulate by setting attributes to None)
        store._chunk_shape = None  # type: ignore
        store._chunks_per_shard = None  # type: ignore
        with pytest.raises(
            RuntimeError, match="Store is not properly initialized for resizing"
        ):
            await store.resize_store(new_shape=(30, 30))


@pytest.mark.asyncio
async def test_resize_variable_success(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """Tests successful resizing of a variable's metadata."""
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
        variable_name = "temp"
        zarr_metadata_key = f"{variable_name}/zarr.json"
        assert zarr_metadata_key in store._root_obj["metadata"]

        # Resize variable
        new_shape = (150, 18, 36)  # Extend time dimension
        await store.resize_variable(variable_name=variable_name, new_shape=new_shape)

        # Verify metadata updated
        new_metadata_cid = store._root_obj["metadata"][zarr_metadata_key]
        new_metadata_bytes = await kubo_cas.load(new_metadata_cid)
        new_metadata = json.loads(new_metadata_bytes)
        assert new_metadata["shape"] == list(new_shape)
        assert store._dirty_root is True

        # Verify store's main array shape unchanged
        assert store._array_shape == array_shape_tuple

        # Flush and verify persistence
        new_root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=new_root_cid
        )
        read_metadata_cid = store_read._root_obj["metadata"][zarr_metadata_key]
        read_metadata_bytes = await kubo_cas.load(read_metadata_cid)
        read_metadata = json.loads(read_metadata_bytes)
        assert read_metadata["shape"] == list(new_shape)


@pytest.mark.asyncio
async def test_resize_variable_invalid_cases(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """Tests error handling in resize_variable."""
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

        # Test read-only store
        store_read_only = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        with pytest.raises(PermissionError, match="Cannot resize a read-only store"):
            await store_read_only.resize_variable("temp", new_shape=(150, 18, 36))

        # Test non-existent variable
        with pytest.raises(
            KeyError,
            match="Cannot find metadata for key 'nonexistent/zarr.json' to resize",
        ):
            await store.resize_variable("nonexistent", new_shape=(150, 18, 36))

        # Test invalid metadata (simulate by setting invalid metadata)
        invalid_metadata = json.dumps({"not_shape": [1, 2, 3]}).encode("utf-8")
        invalid_cid = await kubo_cas.save(invalid_metadata, codec="raw")
        store._root_obj["metadata"]["invalid/zarr.json"] = invalid_cid
        with pytest.raises(ValueError, match="Shape not found in metadata"):
            await store.set(
                "invalid/zarr.json",
                zarr.core.buffer.default_buffer_prototype().buffer.from_bytes(
                    invalid_metadata
                ),
            )


@pytest.mark.asyncio
async def test_resize_store_with_data_preservation(create_ipfs: tuple[str, str]):
    """Tests that resizing the store preserves existing data."""
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
        chunk_data = b"test_data"
        proto = zarr.core.buffer.default_buffer_prototype()
        await store.set(chunk_key, proto.buffer.from_bytes(chunk_data))
        root_cid = await store.flush()

        # Verify chunk exists
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert await store_read.exists(chunk_key)
        read_chunk = await store_read.get(chunk_key, proto)
        assert read_chunk is not None
        assert read_chunk.to_bytes() == chunk_data

        # Resize store
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=False, root_cid=root_cid
        )
        new_shape = (30, 30)
        await store_write.resize_store(new_shape=new_shape)
        new_root_cid = await store_write.flush()

        # Verify chunk still exists
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=new_root_cid
        )
        assert await store_read.exists(chunk_key)
        read_chunk = await store_read.get(chunk_key, proto)
        assert read_chunk is not None
        assert read_chunk.to_bytes() == chunk_data
        assert store_read._array_shape == new_shape
        assert store_read._num_shards == 3  # ceil((3*3)/4) = 3


@pytest.mark.asyncio
async def test_resize_store_in_set_method(create_ipfs: tuple[str, str]):
    """Tests that setting zarr.json triggers resize_store appropriately."""
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

        # Set zarr.json with a new shape
        new_shape = [30, 30]
        metadata = json.dumps({"shape": new_shape, "dtype": "float32"}).encode("utf-8")
        proto = zarr.core.buffer.default_buffer_prototype()
        await store.set("temp/zarr.json", proto.buffer.from_bytes(metadata))

        # Verify resize occurred
        assert store._array_shape == tuple(new_shape)
        assert store._chunks_per_dim == (3, 3)
        assert store._total_chunks == 9
        assert store._num_shards == 3
        assert store._root_obj["chunks"]["array_shape"] == new_shape
        assert len(store._root_obj["chunks"]["shard_cids"]) == 3

        # Verify metadata stored
        assert "temp/zarr.json" in store._root_obj["metadata"]
        root_cid = await store.flush()
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        metadata_buffer = await store_read.get("temp/zarr.json", proto)
        assert metadata_buffer is not None
        assert json.loads(metadata_buffer.to_bytes())["shape"] == new_shape


@pytest.mark.asyncio
async def test_resize_concurrency(create_ipfs: tuple[str, str]):
    """Tests concurrent resize_store operations to ensure locking works."""
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

        # Define multiple resize tasks
        async def resize_task(shape):
            await store.resize_store(new_shape=shape)

        # Run concurrent resize operations
        tasks = [
            resize_task((30, 30)),
            resize_task((40, 40)),
            resize_task((50, 50)),
        ]
        await asyncio.gather(*tasks)

        # Verify final state (last resize should win, but all are safe due to locking)
        assert store._array_shape in [(30, 30), (40, 40), (50, 50)]
        expected_chunks_per_dim = tuple(math.ceil(s / 10) for s in store._array_shape)
        assert store._chunks_per_dim == expected_chunks_per_dim
        assert store._total_chunks == math.prod(expected_chunks_per_dim)
        assert store._num_shards == math.ceil(store._total_chunks / 4)
        assert len(store._root_obj["chunks"]["shard_cids"]) == store._num_shards
        assert store._dirty_root is True
