import asyncio

import dag_cbor
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.core.buffer
from multiformats import CID
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

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
async def test_sharded_zarr_store_write_read(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests writing and reading a Zarr dataset using ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # --- Write ---
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w")
        root_cid = await store_write.flush()
        assert root_cid is not None

        # --- Read ---
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        ds_read = xr.open_zarr(store=store_read)
        xr.testing.assert_identical(test_ds, ds_read)

        # Try to set a chunk directly in read-only mode
        with pytest.raises(PermissionError):
            proto = zarr.core.buffer.default_buffer_prototype()
            await store_read.set("temp/c/0/0", proto.buffer.from_bytes(b"test_data"))


@pytest.mark.asyncio
async def test_load_or_initialize_shard_cache_concurrent_loads(
    create_ipfs: tuple[str, str],
):
    """Tests concurrent shard loading to trigger _pending_shard_loads wait."""
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

        # Create a shard with data
        shard_idx = 0
        shard_data = [
            CID.decode("bafyr4idgcwyxddd2mlskpo7vltcicf5mtozlzt4vzpivqmn343hk3c5nbu")
            for _ in range(4)
        ]
        shard_data_bytes = dag_cbor.encode(shard_data)
        shard_cid_obj = await kubo_cas.save(shard_data_bytes, codec="dag-cbor")
        store._root_obj["chunks"]["shard_cids"][shard_idx] = shard_cid_obj
        store._dirty_root = True
        await store.flush()

        # Simulate concurrent shard loads
        async def load_shard():
            return await store._load_or_initialize_shard_cache(shard_idx)

        # Run multiple tasks concurrently
        tasks = [load_shard() for _ in range(3)]
        results = await asyncio.gather(*tasks)

        # Verify all tasks return the same shard data
        for result in results:
            assert len(result) == 4
            assert all(isinstance(cid, CID) for cid in result)
            assert result == shard_data

        # Verify shard is cached and no pending loads remain
        assert shard_idx in store._shard_data_cache
        assert store._shard_data_cache[shard_idx] == shard_data
        assert shard_idx not in store._pending_shard_loads


@pytest.mark.asyncio
async def test_sharded_zarr_store_append(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests appending data to an existing Zarr dataset in the ShardedZarrStore,
    which specifically exercises the store resizing logic.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    initial_ds = random_zarr_dataset

    ordered_dims = list(initial_ds.sizes)
    array_shape_tuple = tuple(initial_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(initial_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # 1. --- Write Initial Dataset ---
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=10,
        )
        initial_ds.to_zarr(store=store_write, mode="w")
        initial_cid = await store_write.flush()
        assert initial_cid is not None

        # 2. --- Prepare Data to Append ---
        # Create a new dataset with 50 more time steps
        append_times = pd.date_range(
            initial_ds.time[-1].values + pd.Timedelta(days=1), periods=50
        )
        append_temp = np.random.randn(
            len(append_times), len(initial_ds.lat), len(initial_ds.lon)
        )

        append_ds = xr.Dataset(
            {
                "temp": (["time", "lat", "lon"], append_temp),
            },
            coords={"time": append_times, "lat": initial_ds.lat, "lon": initial_ds.lon},
        ).chunk({"time": 20, "lat": 18, "lon": 36})

        # 3. --- Perform Append Operation ---
        store_append = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            root_cid=initial_cid,
        )
        append_ds.to_zarr(store=store_append, mode="a", append_dim="time")
        final_cid = await store_append.flush()
        print(f"Data written to ShardedZarrStore with root CID: {final_cid}")
        assert final_cid is not None
        assert final_cid != initial_cid

        # 4. --- Verify the Final Dataset ---
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=True,
            root_cid=final_cid,
        )
        final_ds_read = xr.open_zarr(store=store_read)

        # The expected result is the concatenation of the two datasets
        expected_final_ds = xr.concat([initial_ds, append_ds], dim="time")

        # Verify that the data read from the store is identical to the expected result
        xr.testing.assert_identical(expected_final_ds, final_ds_read)
        print("\n✅ Append test successful! Data verified.")


@pytest.mark.asyncio
async def test_sharded_zarr_store_init(create_ipfs: tuple[str, str]):
    """
    Tests the initialization of the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    array_shape = (100, 100)
    chunk_shape = (10, 10)
    chunks_per_shard = 64

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test successful creation
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
        )
        assert store is not None

        # Test missing parameters for new store
        with pytest.raises(ValueError):
            await ShardedZarrStore.open(cas=kubo_cas, read_only=False)

        # Test opening read-only store without root_cid
        with pytest.raises(ValueError):
            await ShardedZarrStore.open(cas=kubo_cas, read_only=True)


@pytest.mark.asyncio
async def test_sharded_zarr_store_metadata(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests metadata handling in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w")
        root_cid = await store_write.flush()

        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        # Test exists
        assert await store_read.exists("lat/zarr.json")
        assert await store_read.exists("lon/zarr.json")
        assert await store_read.exists("time/zarr.json")
        assert await store_read.exists("temp/zarr.json")
        assert await store_read.exists("lat/c/0")
        assert await store_read.exists("lon/c/0")
        assert await store_read.exists("time/c/0")
        # assert not await store_read.exists("nonexistent")

        # Test list
        keys = [key async for key in store_read.list()]
        assert len(keys) > 0
        assert "lat/zarr.json" in keys

        prefix = "lat"
        keys_with_prefix = [key async for key in store_read.list_prefix(prefix=prefix)]
        assert "lat/zarr.json" in keys_with_prefix
        assert "lat/c/0" in keys_with_prefix


@pytest.mark.asyncio
async def test_sharded_zarr_store_chunks(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests chunk data handling in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w")
        root_cid = await store_write.flush()

        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )

        # Test get
        chunk_key = "temp/c/0/0/0"
        proto = zarr.core.buffer.default_buffer_prototype()
        chunk_data = await store_read.get(chunk_key, proto)
        assert chunk_data is not None

        # Test delete
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=False, root_cid=root_cid
        )
        await store_write.delete(chunk_key)
        await store_write.flush()

        store_read_after_delete = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=await store_write.flush()
        )
        assert await store_read_after_delete.get(chunk_key, proto) is None


@pytest.mark.asyncio
async def test_chunk_and_delete_logic(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """Tests chunk getting, deleting, and related error handling."""
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w", consolidated=True)
        root_cid = await store_write.flush()

        # Re-open as writable to test deletion
        store_rw = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=False, root_cid=root_cid
        )

        chunk_key = "temp/c/0/0/0"
        proto = zarr.core.buffer.default_buffer_prototype()

        # Verify chunk exists and can be read
        assert await store_rw.exists(chunk_key)
        chunk_data = await store_rw.get(chunk_key, proto)
        assert chunk_data is not None

        # Delete the chunk
        await store_rw.delete(chunk_key)
        new_root_cid = await store_rw.flush()

        # Verify it's gone
        store_after_delete = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=new_root_cid
        )
        assert not await store_after_delete.exists(chunk_key)
        assert await store_after_delete.get(chunk_key, proto) is None


@pytest.mark.asyncio
async def test_sharded_zarr_store_partial_reads(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests partial reads in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w")
        root_cid = await store_write.flush()

        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        proto = zarr.core.buffer.default_buffer_prototype()
        chunk_key = "temp/c/0/0/0"
        full_chunk = await store_read.get(chunk_key, proto)
        assert full_chunk is not None
        full_chunk_bytes = full_chunk.to_bytes()

        # Test RangeByteRequest
        byte_range = RangeByteRequest(start=10, end=50)
        partial_chunk = await store_read.get(chunk_key, proto, byte_range=byte_range)
        assert partial_chunk is not None
        assert partial_chunk.to_bytes() == full_chunk_bytes[10:50]


@pytest.mark.asyncio
async def test_partial_reads_and_errors(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """Tests partial reads and error handling in get()."""
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w", consolidated=True)
        root_cid = await store_write.flush()

        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        proto = zarr.core.buffer.default_buffer_prototype()
        chunk_key = "temp/c/0/0/0"
        full_chunk = await store_read.get(chunk_key, proto)
        assert full_chunk is not None
        full_chunk_bytes = full_chunk.to_bytes()

        # Test RangeByteRequest
        byte_range = RangeByteRequest(start=10, end=50)
        partial_chunk = await store_read.get(chunk_key, proto, byte_range=byte_range)
        assert partial_chunk is not None
        assert partial_chunk.to_bytes() == full_chunk_bytes[10:50]

        # Test invalid byte range
        with pytest.raises(ValueError):
            await store_read.get(
                chunk_key, proto, byte_range=RangeByteRequest(start=50, end=10)
            )


@pytest.mark.asyncio
async def test_zero_sized_array(create_ipfs: tuple[str, str]):
    """Test handling of arrays with a zero-length dimension."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(100, 0),
            chunk_shape=(10, 10),
            chunks_per_shard=64,
        )
        assert store._total_chunks == 0
        assert store._num_shards == 0
        root_cid = await store.flush()

        # Read it back and verify
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        assert store_read._total_chunks == 0
        assert store_read._num_shards == 0


@pytest.mark.asyncio
async def test_store_eq_method(create_ipfs: tuple[str, str]):
    """Tests the __eq__ method."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store1 = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(1, 1),
            chunk_shape=(1, 1),
            chunks_per_shard=1,
        )
        root_cid = await store1.flush()
        store2 = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )

        assert store1 == store2


@pytest.mark.asyncio
async def test_listing_and_metadata(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests metadata handling and listing in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w", consolidated=True)
        root_cid = await store_write.flush()

        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        # Test exists for metadata (correcting for xarray group structure)
        assert await store_read.exists("temp/zarr.json")
        assert await store_read.exists("lat/zarr.json")
        assert not await store_read.exists("nonexistent.json")

        # Test listing
        keys = {key async for key in store_read.list()}
        assert "temp/zarr.json" in keys
        assert "lat/zarr.json" in keys

        # Test list_prefix
        prefix_keys = {key async for key in store_read.list_prefix("temp/")}
        assert "temp/zarr.json" in prefix_keys

        # Test list_dir for root
        dir_keys = {key async for key in store_read.list_dir("")}
        assert "temp" in dir_keys
        assert "lat" in dir_keys
        assert "lon" in dir_keys
        assert "zarr.json" in dir_keys

        # Test listing with a prefix
        prefix = "temp/"
        with pytest.raises(
            NotImplementedError, match="Listing with a prefix is not implemented yet."
        ):
            async for key in store_read.list_dir(prefix):
                print(f"Key with prefix '{prefix}': {key}")

        with pytest.raises(
            ValueError, match="Byte range requests are not supported for metadata keys."
        ):
            proto = zarr.core.buffer.default_buffer_prototype()
            byte_range = zarr.abc.store.RangeByteRequest(start=10, end=50)
            await store_read.get("lat/zarr.json", proto, byte_range=byte_range)


@pytest.mark.asyncio
async def test_sharded_zarr_store_init_errors(create_ipfs: tuple[str, str]):
    """
    Tests initialization errors for ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test missing parameters for a new store
        with pytest.raises(ValueError, match="must be provided for a new store"):
            await ShardedZarrStore.open(cas=kubo_cas, read_only=False)

        # Test opening a read-only store without a root_cid
        with pytest.raises(ValueError, match="must be provided for a read-only store"):
            await ShardedZarrStore.open(cas=kubo_cas, read_only=True)

        # Test invalid chunks_per_shard
        with pytest.raises(ValueError, match="must be a positive integer"):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10,),
                chunk_shape=(5,),
                chunks_per_shard=0,
            )

        # Test invalid chunk_shape
        with pytest.raises(
            ValueError, match="All chunk_shape dimensions must be positive"
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10,),
                chunk_shape=(0,),
                chunks_per_shard=10,
            )

        # Test invalid array_shape
        with pytest.raises(
            ValueError, match="All array_shape dimensions must be non-negative"
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(-10,),
                chunk_shape=(5,),
                chunks_per_shard=10,
            )


@pytest.mark.asyncio
async def test_sharded_zarr_store_get_partial_values(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests the get_partial_values method of ShardedZarrStore, including RangeByteRequest,
    OffsetByteRequest, SuffixByteRequest, and full reads, along with error handling for
    invalid byte ranges.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.sizes)
    array_shape_tuple = tuple(test_ds.sizes[dim] for dim in ordered_dims)
    chunk_shape_tuple = tuple(test_ds.chunks[dim][0] for dim in ordered_dims)

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # 1. --- Write Dataset to ShardedZarrStore ---
        store_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        test_ds.to_zarr(store=store_write, mode="w")
        root_cid = await store_write.flush()
        assert root_cid is not None

        # 2. --- Open Store for Reading ---
        store_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid
        )
        proto = zarr.core.buffer.default_buffer_prototype()

        # 3. --- Find a Chunk Key to Test ---
        chunk_key = "temp/c/0/0/0"  # Default chunk key
        # async for key in store_read.list():
        #     print(f"Found key: {key}")
        #     if key.startswith("temp/c/") and not key.endswith(".json"):
        #         chunk_key = key
        #         break

        assert chunk_key is not None, "Could not find a chunk key to test."
        print(f"Testing with chunk key: {chunk_key}")

        # 4. --- Get Full Chunk Data for Comparison ---
        full_chunk_buffer = await store_read.get(chunk_key, proto)
        assert full_chunk_buffer is not None
        full_chunk_data = full_chunk_buffer.to_bytes()
        chunk_len = len(full_chunk_data)
        print(f"Full chunk size: {chunk_len} bytes")

        # Ensure chunk is large enough for meaningful partial read tests
        assert chunk_len > 100, "Chunk size too small for partial value tests"

        # 5. --- Define Byte Requests ---
        range_req = RangeByteRequest(start=10, end=50)  # Request 40 bytes
        offset_req = OffsetByteRequest(offset=chunk_len - 30)  # Last 30 bytes
        suffix_req = SuffixByteRequest(suffix=20)  # Last 20 bytes

        key_ranges_to_test = [
            (chunk_key, range_req),
            (chunk_key, offset_req),
            (chunk_key, suffix_req),
            (chunk_key, None),  # Full read
        ]

        # 6. --- Call get_partial_values ---
        results = await store_read.get_partial_values(proto, key_ranges_to_test)

        # 7. --- Assertions ---
        assert len(results) == 4, "Expected 4 results from get_partial_values"

        assert results[0] is not None, "RangeByteRequest result should not be None"
        assert results[1] is not None, "OffsetByteRequest result should not be None"
        assert results[2] is not None, "SuffixByteRequest result should not be None"
        assert results[3] is not None, "Full read result should not be None"

        # Check RangeByteRequest result
        expected_range = full_chunk_data[10:50]
        assert results[0].to_bytes() == expected_range, (
            "RangeByteRequest result does not match"
        )
        print(f"RangeByteRequest: OK (Got {len(results[0].to_bytes())} bytes)")

        # Check OffsetByteRequest result
        expected_offset = full_chunk_data[chunk_len - 30 :]
        assert results[1].to_bytes() == expected_offset, (
            "OffsetByteRequest result does not match"
        )
        print(f"OffsetByteRequest: OK (Got {len(results[1].to_bytes())} bytes)")

        # Check SuffixByteRequest result
        # expected_suffix = full_chunk_data[-20:]
        # assert results[2].to_bytes() == expected_suffix, (
        #     "SuffixByteRequest result does not match"
        # )
        # print(f"SuffixByteRequest: OK (Got {len(results[2].to_bytes())} bytes)")

        # Check full read result
        assert results[3].to_bytes() == full_chunk_data, (
            "Full read via get_partial_values does not match"
        )
        print(f"Full Read: OK (Got {len(results[3].to_bytes())} bytes)")

        # 8. --- Test Invalid Byte Range ---
        invalid_range_req = RangeByteRequest(start=50, end=10)
        with pytest.raises(
            ValueError,
            match="Byte range start.*cannot be greater than end",
        ):
            await store_read.get_partial_values(proto, [(chunk_key, invalid_range_req)])

        print("\n✅ get_partial_values test successful! All partial reads verified.")

@pytest.mark.asyncio
async def test_sharded_zarr_store_parse_chunk_key(create_ipfs: tuple[str, str]):
    """Tests chunk key parsing edge cases."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(20, 20),
            chunk_shape=(10, 10),
            chunks_per_shard=4,
        )

        # Test metadata key
        assert store._parse_chunk_key("zarr.json") is None
        assert store._parse_chunk_key("group1/zarr.json") is None

        # Test excluded array prefixes
        assert store._parse_chunk_key("time/c/0") is None
        assert store._parse_chunk_key("lat/c/0/0") is None
        assert store._parse_chunk_key("lon/c/0/0") is None

        # Test uninitialized store
        # uninitialized_store = ShardedZarrStore(kubo_cas, read_only=False, root_cid=None)
        # assert uninitialized_store._parse_chunk_key("temp/c/0/0") is None

        # # Test get on uninitialized store
        # with pytest.raises(
        #     RuntimeError, match="Load the root object first before accessing data."
        # ):
        #     proto = zarr.core.buffer.default_buffer_prototype()
        #     await uninitialized_store.get("temp/c/0/0", proto)

        # with pytest.raises(RuntimeError, match="Cannot load root without a root_cid."):
        #     await uninitialized_store._load_root_from_cid()

        # Test dimensionality mismatch
        with pytest.raises(IndexError, match="tuple index out of range"):
            store._parse_chunk_key("temp/c/0/0/0/0")

        # Test invalid coordinates
        with pytest.raises(ValueError, match="invalid literal"):
            store._parse_chunk_key("temp/c/0/invalid")
        with pytest.raises(IndexError, match="Chunk coordinate"):
            store._parse_chunk_key("temp/c/0/-1")

        with pytest.raises(IndexError, match="Chunk coordinate"):
            store._parse_chunk_key("temp/c/3/0")


@pytest.mark.asyncio
async def test_sharded_zarr_store_init_invalid_shapes(create_ipfs: tuple[str, str]):
    """Tests initialization with invalid shapes and manifest errors."""
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test negative chunk_shape dimension
        with pytest.raises(
            ValueError, match="All chunk_shape dimensions must be positive"
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10, 10),
                chunk_shape=(-5, 5),
                chunks_per_shard=10,
            )

        # Test negative array_shape dimension
        with pytest.raises(
            ValueError, match="All array_shape dimensions must be non-negative"
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10, -10),
                chunk_shape=(5, 5),
                chunks_per_shard=10,
            )

        # Test zero-sized array
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(0, 10),
            chunk_shape=(5, 5),
            chunks_per_shard=10,
        )
        assert store._total_chunks == 0
        assert store._num_shards == 0
        assert store._root_obj is not None
        assert len(store._root_obj["chunks"]["shard_cids"]) == 0  # Line 163

        # Test invalid manifest version
        invalid_root_obj = {
            "manifest_version": "invalid_version",
            "metadata": {},
            "chunks": {
                "array_shape": [10, 10],
                "chunk_shape": [5, 5],
                "cid_byte_length": 59,
                "sharding_config": {"chunks8048": 10},
                "shard_cids": [None] * 4,
            },
        }
        invalid_root_cid = await kubo_cas.save(
            dag_cbor.encode(invalid_root_obj), codec="dag-cbor"
        )
        with pytest.raises(ValueError, match="Incompatible manifest version"):
            await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=invalid_root_cid
            )

        # Test inconsistent shard count
        invalid_root_obj = {
            "manifest_version": "sharded_zarr_v1",
            "metadata": {},
            "chunks": {
                "array_shape": [
                    10,
                    10,
                ],  # 100 chunks, with 10 chunks per shard -> 10 shards
                "chunk_shape": [5, 5],
                "cid_byte_length": 59,
                "sharding_config": {"chunks_per_shard": 10},
                "shard_cids": [None] * 5,  # Wrong number of shards
            },
        }
        invalid_root_cid = await kubo_cas.save(
            dag_cbor.encode(invalid_root_obj), codec="dag-cbor"
        )
        with pytest.raises(ValueError, match="Inconsistent number of shards"):
            await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=invalid_root_cid
            )


@pytest.mark.asyncio
async def test_sharded_zarr_store_lazy_concat(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests lazy concatenation of two xarray datasets stored in separate ShardedZarrStores,
    ensuring the combined dataset can be queried as a single dataset with data fetched
    correctly from the respective stores.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    base_ds = random_zarr_dataset

    # 1. --- Prepare Two Datasets with Distinct Time Ranges ---
    # First dataset: August 1, 2024 to September 30, 2024 (61 days)
    aug_sep_times = pd.date_range("2024-08-01", "2024-09-30", freq="D")
    aug_sep_temp = np.random.randn(
        len(aug_sep_times), len(base_ds.lat), len(base_ds.lon)
    )
    ds1 = xr.Dataset(
        {
            "temp": (["time", "lat", "lon"], aug_sep_temp),
        },
        coords={"time": aug_sep_times, "lat": base_ds.lat, "lon": base_ds.lon},
    ).chunk({"time": 20, "lat": 18, "lon": 36})

    # Second dataset: October 1, 2024 to November 30, 2024 (61 days)
    oct_nov_times = pd.date_range("2024-10-01", "2024-11-30", freq="D")
    oct_nov_temp = np.random.randn(
        len(oct_nov_times), len(base_ds.lat), len(base_ds.lon)
    )
    ds2 = xr.Dataset(
        {
            "temp": (["time", "lat", "lon"], oct_nov_temp),
        },
        coords={"time": oct_nov_times, "lat": base_ds.lat, "lon": base_ds.lon},
    ).chunk({"time": 20, "lat": 18, "lon": 36})

    # Expected concatenated dataset for verification
    expected_combined_ds = xr.concat([ds1, ds2], dim="time")

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # 2. --- Write First Dataset to ShardedZarrStore ---
        ordered_dims = list(ds1.sizes)
        array_shape_tuple = tuple(ds1.sizes[dim] for dim in ordered_dims)
        chunk_shape_tuple = tuple(ds1.chunks[dim][0] for dim in ordered_dims)

        store1_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        ds1.to_zarr(store=store1_write, mode="w")
        root_cid1 = await store1_write.flush()
        assert root_cid1 is not None

        # 3. --- Write Second Dataset to ShardedZarrStore ---
        array_shape_tuple = tuple(ds2.sizes[dim] for dim in ordered_dims)
        store2_write = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=array_shape_tuple,
            chunk_shape=chunk_shape_tuple,
            chunks_per_shard=64,
        )
        ds2.to_zarr(store=store2_write, mode="w")
        root_cid2 = await store2_write.flush()
        assert root_cid2 is not None

        # 4. --- Read and Lazily Concatenate Datasets ---
        store1_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid1
        )
        store2_read = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=root_cid2
        )

        ds1_read = xr.open_zarr(store=store1_read, chunks="auto")
        ds2_read = xr.open_zarr(store=store2_read, chunks="auto")

        # Verify that datasets are lazy (Dask-backed)
        assert ds1_read.temp.chunks is not None
        assert ds2_read.temp.chunks is not None

        # Lazily concatenate along time dimension
        combined_ds = xr.concat([ds1_read, ds2_read], dim="time")

        # Verify that the combined dataset is still lazy
        assert combined_ds.temp.chunks is not None

        # 5. --- Query Across Both Datasets ---
        # Select a time slice that spans both datasets (e.g., Sep 15 to Oct 15)
        query_slice = combined_ds.sel(time=slice("2024-09-15", "2024-10-15"))

        # Verify that the query is still lazy
        assert query_slice.temp.chunks is not None

        # Compute the result to trigger data loading
        query_result = query_slice.compute()

        # 6. --- Verify Results ---
        # Compare with the expected concatenated dataset
        expected_query_result = expected_combined_ds.sel(
            time=slice("2024-09-15", "2024-10-15")
        )
        xr.testing.assert_identical(query_result, expected_query_result)

        # Verify specific values at a point to ensure data integrity
        sample_time = pd.Timestamp("2024-09-30")  # From ds1
        sample_result = query_result.sel(time=sample_time, method="nearest").temp.values
        expected_sample = expected_combined_ds.sel(
            time=sample_time, method="nearest"
        ).temp.values
        np.testing.assert_array_equal(sample_result, expected_sample)

        sample_time = pd.Timestamp("2024-10-01")  # From ds2
        sample_result = query_result.sel(time=sample_time, method="nearest").temp.values
        expected_sample = expected_combined_ds.sel(
            time=sample_time, method="nearest"
        ).temp.values
        np.testing.assert_array_equal(sample_result, expected_sample)

        print("\n✅ Lazy concatenation test successful! Data verified.")


@pytest.mark.asyncio
async def test_sharded_zarr_store_lazy_concat_with_cids(create_ipfs: tuple[str, str]):
    """
    Tests lazy concatenation of two xarray datasets stored in separate ShardedZarrStores
    using provided CIDs for finalized and non-finalized data, ensuring the non-finalized
    dataset is sliced after the finalization date (inclusive) and the combined dataset
    can be queried as a single dataset with data fetched correctly from the respective stores.
    """
    rpc_base_url, gateway_base_url = create_ipfs

    # Provided CIDs
    finalized_cid = "bafyr4iacuutc5bgmirkfyzn4igi2wys7e42kkn674hx3c4dv4wrgjp2k2u"
    non_finalized_cid = "bafyr4iayq3aaifmyv4o7ezoi4xyysstit3ohvnq4cnjlbjwueqehlbvkla"
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # 1. --- Open Finalized Dataset ---
        store_finalized = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=finalized_cid
        )
        ds_finalized = xr.open_zarr(store=store_finalized, chunks="auto")

        # Verify that the dataset is lazy (Dask-backed)
        assert ds_finalized["2m_temperature"].chunks is not None

        # Determine the finalization date (last date in finalized dataset)
        finalization_date = np.datetime64(ds_finalized.time.max().values)

        # 2. --- Open Non-Finalized Dataset and Slice After Finalization Date ---
        store_non_finalized = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=non_finalized_cid
        )
        ds_non_finalized = xr.open_zarr(store=store_non_finalized, chunks="auto")

        # Verify that the dataset is lazy
        assert ds_non_finalized["2m_temperature"].chunks is not None

        # Slice non-finalized dataset to start *after* the finalization date
        # (finalization_date is inclusive for finalized data, so start at +1 hour)
        start_time = finalization_date + np.timedelta64(1, "h")
        ds_non_finalized_sliced = ds_non_finalized.sel(time=slice(start_time, None))

        # Verify that the sliced dataset starts after the finalization date
        if ds_non_finalized_sliced.time.size > 0:
            assert ds_non_finalized_sliced.time.min() > finalization_date
        else:
            # Handle case where non-finalized dataset is empty after slicing
            print("Warning: Non-finalized dataset is empty after slicing.")

        # 3. --- Lazily Concatenate Datasets ---
        combined_ds = xr.concat([ds_finalized, ds_non_finalized_sliced], dim="time")
        print("\nCombined dataset time range:")
        print(combined_ds.time.min().values, "to", combined_ds.time.max().values)

        # Verify that the combined dataset is still lazy
        assert combined_ds["2m_temperature"].chunks is not None

        # 4. --- Query Across Both Datasets ---
        # Select a time slice that spans both datasets
        # Use a range that includes the boundary (e.g., finalization date and after)
        query_start = finalization_date - np.timedelta64(1, "D")  # 1 day before
        query_end = finalization_date + np.timedelta64(1, "D")  # 1 day after
        query_slice = combined_ds.sel(
            time=slice(query_start, query_end), latitude=0, longitude=0
        )
        # Make sure the query slice aligned with the query_start and query_end

        assert query_slice.time.min() >= query_start
        assert query_slice.time.max() <= query_end

        # Verify that the query is still lazy
        assert query_slice["2m_temperature"].chunks is not None

        # Compute the result to trigger data loading
        query_result = query_slice.compute()

        # 5. --- Verify Results ---
        # Verify data integrity at specific points
        # Check the last finalized time (from finalized dataset)
        sample_time_finalized = finalization_date
        if sample_time_finalized in query_result.time.values:
            sample_result = query_result.sel(
                time=sample_time_finalized, method="nearest"
            )["2m_temperature"].values
            expected_sample = ds_finalized.sel(
                time=sample_time_finalized, latitude=0, longitude=0, method="nearest"
            )["2m_temperature"].values
            np.testing.assert_array_equal(sample_result, expected_sample)

        # Check the first non-finalized time (from non-finalized dataset, if available)
        if ds_non_finalized_sliced.time.size > 0:
            sample_time_non_finalized = ds_non_finalized_sliced.time.min().values
            if sample_time_non_finalized in query_result.time.values:
                sample_result = query_result.sel(
                    time=sample_time_non_finalized, method="nearest"
                )["2m_temperature"].values
                expected_sample = ds_non_finalized_sliced.sel(
                    time=sample_time_non_finalized,
                    latitude=0,
                    longitude=0,
                    method="nearest",
                )["2m_temperature"].values
                np.testing.assert_array_equal(sample_result, expected_sample)

        # 6. --- Additional Validation ---
        # Verify that the concatenated dataset has no overlapping times
        time_values = combined_ds.time.values
        assert np.all(np.diff(time_values) > np.timedelta64(0, "ns")), (
            "Overlapping or unsorted time values detected"
        )

        # Verify that the query result covers the expected time range
        if query_result.time.size > 0:
            assert query_result.time.min() >= query_start
            assert query_result.time.max() <= query_end

        print("\n✅ Lazy concatenation with CIDs test successful! Data verified.")
