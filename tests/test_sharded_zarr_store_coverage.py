
import dag_cbor
import pytest
import zarr.abc.store
import zarr.core.buffer

from py_hamt import KuboCAS
from py_hamt.sharded_zarr_store import ShardedZarrStore


@pytest.mark.asyncio
async def test_sharded_zarr_store_init_exceptions(create_ipfs: tuple[str, str]):
    """
    Tests various initialization exceptions in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test RuntimeError when base shape information is not set
        # with pytest.raises(RuntimeError, match="Base shape information is not set."):
        #     store = ShardedZarrStore(kubo_cas, False, None)
        #     store._update_geometry()

        # Test ValueError for non-positive chunk_shape dimensions
        with pytest.raises(
            ValueError, match="All chunk_shape dimensions must be positive."
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10, 10),
                chunk_shape=(10, 0),
                chunks_per_shard=10,
            )

        # Test ValueError for non-negative array_shape dimensions
        with pytest.raises(
            ValueError, match="All array_shape dimensions must be non-negative."
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10, -10),
                chunk_shape=(10, 10),
                chunks_per_shard=10,
            )

        # Test ValueError when array_shape is not provided for a new store
        with pytest.raises(
            ValueError,
            match="array_shape and chunk_shape must be provided for a new store.",
        ):
            await ShardedZarrStore.open(cas=kubo_cas, read_only=False, chunk_shape=(10, 10))

        # Test ValueError for non-positive chunks_per_shard
        with pytest.raises(ValueError, match="chunks_per_shard must be a positive integer."):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10, 10),
                chunk_shape=(10, 10),
                chunks_per_shard=0,
            )

        # Test ValueError when root_cid is not provided for a read-only store
        with pytest.raises(ValueError, match="root_cid must be provided for a read-only store."):
            await ShardedZarrStore.open(cas=kubo_cas, read_only=True)


@pytest.mark.asyncio
async def test_sharded_zarr_store_load_root_exceptions(create_ipfs: tuple[str, str]):
    """
    Tests exceptions raised during the loading of the root object.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Test RuntimeError when _root_cid is not set
        # with pytest.raises(RuntimeError, match="Cannot load root without a root_cid."):
        #     store = ShardedZarrStore(kubo_cas, True, None)
        #     await store._load_root_from_cid()

        # Test ValueError for an incompatible manifest version
        invalid_manifest_root = {
            "manifest_version": "invalid_version",
            "chunks": {
                "array_shape": [10],
                "chunk_shape": [5],
                "sharding_config": {"chunks_per_shard": 1},
                "shard_cids": [],
            },
        }
        invalid_manifest_cid = await kubo_cas.save(
            dag_cbor.encode(invalid_manifest_root), codec="dag-cbor"
        )
        with pytest.raises(ValueError, match="Incompatible manifest version"):
            await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=invalid_manifest_cid
            )

        # Test ValueError for an inconsistent number of shards
        inconsistent_shards_root = {
            "manifest_version": "sharded_zarr_v1",
            "chunks": {
                "array_shape": [10],
                "chunk_shape": [5],
                "sharding_config": {"chunks_per_shard": 1},
                "shard_cids": [None, None, None],  # Should be 2 shards, but array shape dictates 2 total chunks
            },
        }
        inconsistent_shards_cid = await kubo_cas.save(
            dag_cbor.encode(inconsistent_shards_root), codec="dag-cbor"
        )
        with pytest.raises(ValueError, match="Inconsistent number of shards"):
            await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=inconsistent_shards_cid
            )


@pytest.mark.asyncio
async def test_sharded_zarr_store_shard_handling_exceptions(
    create_ipfs: tuple[str, str], caplog
):
    """
    Tests exceptions and logging during shard handling.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(10,),
            chunk_shape=(5,),
            chunks_per_shard=1,
        )

        # Test TypeError when a shard does not decode to a list
        invalid_shard_cid = await kubo_cas.save(
            dag_cbor.encode({"not": "a list"}), "dag-cbor"
        )
        store._root_obj["chunks"]["shard_cids"][0] = invalid_shard_cid
        with pytest.raises(TypeError, match="Shard 0 did not decode to a list."):
            await store._load_or_initialize_shard_cache(0)

        # bad __eq__ method
        assert store != { "not a ShardedZarrStore": "test" }


@pytest.mark.asyncio
async def test_sharded_zarr_store_get_set_exceptions(create_ipfs: tuple[str, str]):
    """
    Tests exceptions raised during get and set operations.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(10,),
            chunk_shape=(5,),
            chunks_per_shard=1,
        )
        proto = zarr.core.buffer.default_buffer_prototype()

        # Test RuntimeError when root object is not loaded in get
        # store_no_root = ShardedZarrStore(kubo_cas, True, "some_cid")
        # with pytest.raises(
        #     RuntimeError, match="Load the root object first before accessing data."
        # ):
        #     await store_no_root.get("key", proto)

        # Set some bytes to /c/0 to ensure it exists
        await store.set(
            "/c/0",
            proto.buffer.from_bytes(b'{"shape": [10], "dtype": "float32"}'),
        )

        # Test ValueError for invalid byte range in get
        with pytest.raises(
            ValueError,
            match="Byte range start .* cannot be greater than end .*",
        ):
            await store.get(
                "/c/0", proto, byte_range=zarr.abc.store.RangeByteRequest(start=10, end=5)
            )

        # Test NotImplementedError for set_partial_values
        with pytest.raises(NotImplementedError):
            await store.set_partial_values([])

        # Test ValueError when shape is not found in metadata during set
        with pytest.raises(ValueError, match="Shape not found in metadata."):
            await store.set("test/zarr.json", proto.buffer.from_bytes(b'{"not": "a shape"}'))


@pytest.mark.asyncio
async def test_sharded_zarr_store_other_exceptions(create_ipfs: tuple[str, str]):
    """
    Tests other miscellaneous exceptions in the ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        store = await ShardedZarrStore.open(
            cas=kubo_cas,
            read_only=False,
            array_shape=(10,),
            chunk_shape=(5,),
            chunks_per_shard=1,
        )

        # Test RuntimeError for uninitialized store in flush
        # store_no_root = ShardedZarrStore(kubo_cas, False, None)
        # with pytest.raises(RuntimeError, match="Store not initialized for writing."):
        #     await store_no_root.flush()

        # Test ValueError when resizing a store with a different number of dimensions
        with pytest.raises(
            ValueError,
            match="New shape must have the same number of dimensions as the old shape.",
        ):
            await store.resize_store(new_shape=(10, 10))

        # Test KeyError when resizing a variable that doesn't exist
        with pytest.raises(
            KeyError,
            match="Cannot find metadata for key 'nonexistent/zarr.json' to resize.",
        ):
            await store.resize_variable("nonexistent", new_shape=(20,))
    

        # Test RuntimeError when listing a store with no root object
        # with pytest.raises(RuntimeError, match="Root object not loaded."):
        #     async for _ in store_no_root.list():
        #         pass

        # # Test RuntimeError when listing directories of a store with no root object
        # with pytest.raises(RuntimeError, match="Root object not loaded."):
        #     async for _ in store_no_root.list_dir(""):
        #         pass

        # with pytest.raises(ValueError, match="Linear chunk index cannot be negative."):
        #     await store_no_root._get_shard_info(-1)