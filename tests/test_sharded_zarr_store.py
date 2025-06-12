import dag_cbor
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.core.buffer
from zarr.abc.store import RangeByteRequest

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

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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

        # Test deleting a non-existent key
        with pytest.raises(KeyError):
            await store_rw.delete("nonexistent/c/0/0/0")

        # Test deleting an already deleted key
        with pytest.raises(KeyError):
            await store_rw.delete(chunk_key)


@pytest.mark.asyncio
async def test_sharded_zarr_store_partial_reads(
    create_ipfs: tuple[str, str], random_zarr_dataset: xr.Dataset
):
    """
    Tests partial reads in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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

    ordered_dims = list(test_ds.dims)
    array_shape_tuple = tuple(test_ds.dims[dim] for dim in ordered_dims)
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


# @pytest.mark.asyncio
# async def test_sharded_zarr_store_init_invalid_shapes(create_ipfs: tuple[str, str]):
#     """Tests initialization with invalid shapes and manifest errors."""
#     rpc_base_url, gateway_base_url = create_ipfs
#     async with KuboCAS(
#         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
#     ) as kubo_cas:
#         # Test negative chunk_shape dimension (line 136)
#         with pytest.raises(
#             ValueError, match="All chunk_shape dimensions must be positive"
#         ):
#             await ShardedZarrStore.open(
#                 cas=kubo_cas,
#                 read_only=False,
#                 array_shape=(10, 10),
#                 chunk_shape=(-5, 5),
#                 chunks_per_shard=10,
#             )

#         # Test negative array_shape dimension (line 141)
#         with pytest.raises(
#             ValueError, match="All array_shape dimensions must be non-negative"
#         ):
#             await ShardedZarrStore.open(
#                 cas=kubo_cas,
#                 read_only=False,
#                 array_shape=(10, -10),
#                 chunk_shape=(5, 5),
#                 chunks_per_shard=10,
#             )

#         # Test zero-sized array (lines 150, 163) - reinforce existing test
#         store = await ShardedZarrStore.open(
#             cas=kubo_cas,
#             read_only=False,
#             array_shape=(0, 10),
#             chunk_shape=(5, 5),
#             chunks_per_shard=10,
#         )
#         assert store._total_chunks == 0
#         assert store._num_shards == 0
#         assert len(store._root_obj["chunks"]["shard_cids"]) == 0  # Line 163
#         root_cid = await store.flush()

#         # Test invalid manifest version (line 224)
#         invalid_root_obj = {
#             "manifest_version": "invalid_version",
#             "metadata": {},
#             "chunks": {
#                 "array_shape": [10, 10],
#                 "chunk_shape": [5, 5],
#                 "cid_byte_length": 59,
#                 "sharding_config": {"chunks8048": 10},
#                 "shard_cids": [None] * 4,
#             },
#         }
#         invalid_root_cid = await kubo_cas.save(
#             dag_cbor.encode(invalid_root_obj), codec="dag-cbor"
#         )
#         with pytest.raises(ValueError, match="Incompatible manifest version"):
#             await ShardedZarrStore.open(
#                 cas=kubo_cas, read_only=True, root_cid=invalid_root_cid
#             )

#         # Test inconsistent shard count (line 236)
#         invalid_root_obj = {
#             "manifest_version": "sharded_zarr_v1",
#             "metadata": {},
#             "chunks": {
#                 "array_shape": [
#                     10,
#                     10,
#                 ],  # 100 chunks, with 10 chunks per shard -> 10 shards
#                 "chunk_shape": [5, 5],
#                 "cid_byte_length": 59,
#                 "sharding_config": {"chunks_per_shard": 10},
#                 "shard_cids": [None] * 5,  # Wrong number of shards
#             },
#         }
#         invalid_root_cid = await kubo_cas.save(
#             dag_cbor.encode(invalid_root_obj), codec="dag-cbor"
#         )
#         with pytest.raises(ValueError, match="Inconsistent number of shards"):
#             await ShardedZarrStore.open(
#                 cas=kubo_cas, read_only=True, root_cid=invalid_root_cid
#             )


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
        uninitialized_store = ShardedZarrStore(kubo_cas, read_only=False, root_cid=None)
        assert uninitialized_store._parse_chunk_key("temp/c/0/0") is None

        # Test get on uninitialized store
        with pytest.raises(
            RuntimeError, match="Load the root object first before accessing data."
        ):
            proto = zarr.core.buffer.default_buffer_prototype()
            await uninitialized_store.get("temp/c/0/0", proto)

        with pytest.raises(RuntimeError, match="Cannot load root without a root_cid."):
            await uninitialized_store._load_root_from_cid()

        # Test dimensionality mismatch
        assert store._parse_chunk_key("temp/c/0/0/0") is None  # 3D key for 2D array

        # Test invalid coordinates
        assert (
            store._parse_chunk_key("temp/c/3/0") is None
        )  # Out of bounds (3 >= 2 chunks)
        assert store._parse_chunk_key("temp/c/0/invalid") is None  # Non-integer
        assert store._parse_chunk_key("temp/c/0/-1") is None  # Negative coordinate


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
