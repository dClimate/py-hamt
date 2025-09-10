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
            await ShardedZarrStore.open(
                cas=kubo_cas, read_only=False, chunk_shape=(10, 10)
            )

        # Test ValueError for non-positive chunks_per_shard
        with pytest.raises(
            ValueError, match="chunks_per_shard must be a positive integer."
        ):
            await ShardedZarrStore.open(
                cas=kubo_cas,
                read_only=False,
                array_shape=(10, 10),
                chunk_shape=(10, 10),
                chunks_per_shard=0,
            )

        # Test ValueError when root_cid is not provided for a read-only store
        with pytest.raises(
            ValueError, match="root_cid must be provided for a read-only store."
        ):
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
                "shard_cids": [
                    None,
                    None,
                    None,
                ],  # Should be 2 shards, but array shape dictates 2 total chunks
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
        assert store != {"not a ShardedZarrStore": "test"}


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
                "/c/0",
                proto,
                byte_range=zarr.abc.store.RangeByteRequest(start=10, end=5),
            )

        # Test NotImplementedError for set_partial_values
        with pytest.raises(NotImplementedError):
            await store.set_partial_values([])

        # Test ValueError when shape is not found in metadata during set
        with pytest.raises(ValueError, match="Shape not found in metadata."):
            await store.set(
                "test/zarr.json", proto.buffer.from_bytes(b'{"not": "a shape"}')
            )


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


@pytest.mark.asyncio
async def test_memory_bounded_lru_cache_empty_shard():
    """Test line 40: empty shard handling in _get_shard_size"""
    from py_hamt.sharded_zarr_store import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(max_memory_bytes=1000)
    empty_shard = []

    # Test that empty shard is handled correctly (line 40)
    size = cache._get_shard_size(empty_shard)
    assert size > 0  # sys.getsizeof should return some size even for empty list

    await cache.put(0, empty_shard)
    retrieved = await cache.get(0)
    assert retrieved == empty_shard


@pytest.mark.asyncio
async def test_memory_bounded_lru_cache_update_existing():
    from multiformats import CID

    from py_hamt.sharded_zarr_store import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(max_memory_bytes=10000)
    test_cid = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")

    # First put
    shard1 = [test_cid] * 2
    await cache.put(0, shard1)

    shard2 = [test_cid] * 3
    await cache.put(0, shard2, is_dirty=True)

    retrieved = await cache.get(0)
    assert retrieved == shard2
    assert cache.dirty_cache_size == 1


@pytest.mark.asyncio
async def test_memory_bounded_lru_cache_eviction_break():
    """Test line 96: eviction break when no clean shards available"""
    from multiformats import CID

    from py_hamt.sharded_zarr_store import MemoryBoundedLRUCache

    cache = MemoryBoundedLRUCache(max_memory_bytes=500)  # Small cache
    test_cid = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")

    # Add several dirty shards to fill cache
    large_shard = [test_cid] * 10
    for i in range(3):
        await cache.put(i, large_shard, is_dirty=True)

    # Try to add another large shard - should trigger line 96 break
    huge_shard = [test_cid] * 20
    await cache.put(3, huge_shard)

    # All dirty shards should still be in cache (not evicted)
    for i in range(3):
        assert await cache.get(i) is not None
    assert cache.dirty_cache_size == 3


@pytest.mark.asyncio
async def test_sharded_zarr_store_duplicate_root_loading():
    """Test line 277: duplicate root object loading"""
    import dag_cbor

    from py_hamt.sharded_zarr_store import ShardedZarrStore

    # Create mock CAS that returns malformed data to trigger line 277
    class MockCAS:
        def __init__(self):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            # Return valid DAG-CBOR for a sharded zarr root
            root_obj = {
                "manifest_version": "sharded_zarr_v1",
                "metadata": {},
                "chunks": {
                    "array_shape": [10],
                    "chunk_shape": [5],
                    "sharding_config": {"chunks_per_shard": 1},
                    "shard_cids": [None, None],
                },
            }
            return dag_cbor.encode(root_obj)

    # Create store with mock CAS
    mock_cas = MockCAS()
    store = ShardedZarrStore(mock_cas, True, "test_cid")

    # This should trigger line 277 where root_obj gets set twice
    await store._load_root_from_cid()

    assert store._root_obj is not None
    assert store._array_shape == (10,)


@pytest.mark.asyncio
async def test_sharded_zarr_store_invalid_root_object_structure():
    import dag_cbor

    from py_hamt.sharded_zarr_store import ShardedZarrStore

    class MockCAS:
        def __init__(self, root_obj):
            self.root_obj = root_obj

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            return dag_cbor.encode(self.root_obj)

    # Test line 274: root object is not a dict
    mock_cas_not_dict = MockCAS("not a dictionary")
    store = ShardedZarrStore(mock_cas_not_dict, True, "test_cid")
    with pytest.raises(ValueError, match="Root object is not a valid dictionary"):
        await store._load_root_from_cid()

    # Test line 274: root object missing 'chunks' key
    mock_cas_no_chunks = MockCAS({
        "metadata": {},
        "manifest_version": "sharded_zarr_v1",
    })
    store = ShardedZarrStore(mock_cas_no_chunks, True, "test_cid")
    with pytest.raises(ValueError, match="Root object is not a valid dictionary"):
        await store._load_root_from_cid()

    mock_cas_invalid_shard_cids = MockCAS({
        "manifest_version": "sharded_zarr_v1",
        "metadata": {},
        "chunks": {
            "array_shape": [10],
            "chunk_shape": [5],
            "sharding_config": {"chunks_per_shard": 1},
            "shard_cids": "not a list",  # Should be a list
        },
    })
    store = ShardedZarrStore(mock_cas_invalid_shard_cids, True, "test_cid")
    with pytest.raises(ValueError, match="shard_cids is not a list"):
        await store._load_root_from_cid()

    # Test line 280: generic exception handling (invalid DAG-CBOR)
    class MockCASInvalidDagCbor:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            return b"invalid dag-cbor data"  # This will cause dag_cbor.decode to fail

    mock_cas_invalid_cbor = MockCASInvalidDagCbor()
    store = ShardedZarrStore(mock_cas_invalid_cbor, True, "test_cid")
    with pytest.raises(ValueError, match="Failed to decode root object"):
        await store._load_root_from_cid()


@pytest.mark.asyncio
async def test_sharded_zarr_store_invalid_manifest_version():
    import dag_cbor

    from py_hamt.sharded_zarr_store import ShardedZarrStore

    class MockCAS:
        def __init__(self, manifest_version):
            self.manifest_version = manifest_version

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            root_obj = {
                "manifest_version": self.manifest_version,
                "metadata": {},
                "chunks": {
                    "array_shape": [10],
                    "chunk_shape": [5],
                    "sharding_config": {"chunks_per_shard": 1},
                    "shard_cids": [None, None],
                },
            }
            return dag_cbor.encode(root_obj)

    mock_cas = MockCAS("wrong_version")
    store = ShardedZarrStore(mock_cas, True, "test_cid")

    with pytest.raises(ValueError, match="Incompatible manifest version"):
        await store._load_root_from_cid()


@pytest.mark.asyncio
async def test_sharded_zarr_store_shard_fetch_retry():
    from py_hamt.sharded_zarr_store import ShardedZarrStore

    class MockCAS:
        def __init__(self, fail_count=2):
            self.fail_count = fail_count
            self.attempts = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            self.attempts += 1
            if self.attempts <= self.fail_count:
                raise ConnectionError("Mock connection error")
            # Success on final attempt
            import dag_cbor

            return dag_cbor.encode([None] * 4)

        async def save(self, data, codec=None):
            from multiformats import CID

            return CID.decode(
                "bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm"
            )

    mock_cas = MockCAS(fail_count=2)  # Fail twice, succeed on 3rd attempt
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=4,
    )

    # Set up a shard CID to fetch
    from multiformats import CID

    shard_cid = CID.decode(
        "bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm"
    )
    store._root_obj["chunks"]["shard_cids"][0] = shard_cid

    shard_data = await store._load_or_initialize_shard_cache(0)
    assert shard_data is not None
    assert len(shard_data) == 4

    # Test case where all retries fail
    mock_cas_fail = MockCAS(fail_count=5)  # Fail more than max_retries
    store_fail = await ShardedZarrStore.open(
        cas=mock_cas_fail,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=4,
    )
    store_fail._root_obj["chunks"]["shard_cids"][0] = shard_cid

    with pytest.raises(RuntimeError, match="Failed to fetch shard 0 after 3 attempts"):
        await store_fail._load_or_initialize_shard_cache(0)


@pytest.mark.asyncio
async def test_sharded_zarr_store_with_read_only_clone_attribute():
    """Test line 490: with_read_only clone attribute assignment"""
    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=2,
    )

    # Create clone with different read_only status (should hit line 490)
    clone = store.with_read_only(True)

    # Verify line 490: clone._root_obj = self._root_obj
    assert clone._root_obj is store._root_obj
    assert clone.read_only is True
    assert store.read_only is False


@pytest.mark.asyncio
async def test_sharded_zarr_store_get_method_line_565():
    """Test line 565: get method start (line 565 is the method definition)"""
    import zarr.core.buffer

    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid, offset=None, length=None, suffix=None):
            return b"metadata_content"

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=2,
    )

    # Add metadata to test the get method
    from multiformats import CID

    metadata_cid = CID.decode(
        "bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm"
    )
    store._root_obj["metadata"]["test.json"] = metadata_cid

    # Test get method (line 565 is the method signature)
    proto = zarr.core.buffer.default_buffer_prototype()
    result = await store.get("test.json", proto)

    assert result is not None
    assert result.to_bytes() == b"metadata_content"


@pytest.mark.asyncio
async def test_sharded_zarr_store_exists_exception_handling():
    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=2,
    )

    # This should trigger the exception handling and return False
    exists = await store.exists("invalid/chunk/key/format")
    assert exists is False

    # Test exists with valid chunk key that's out of bounds (should also return False)
    exists = await store.exists("c/100")  # Out of bounds chunk
    assert exists is False


@pytest.mark.asyncio
async def test_sharded_zarr_store_cas_save_failure():
    """Test RuntimeError when cas.save fails in set method"""
    import zarr.core.buffer

    from py_hamt import ShardedZarrStore

    class MockCASFailingSave:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def save(self, data, codec=None):
            # Always fail to save
            raise ConnectionError("Mock CAS save failure")

    mock_cas = MockCASFailingSave()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=2,
    )

    proto = zarr.core.buffer.default_buffer_prototype()
    test_data = proto.buffer.from_bytes(b"test_data")

    with pytest.raises(RuntimeError, match="Failed to save data for key test_key"):
        await store.set("test_key", test_data)


@pytest.mark.asyncio
async def test_sharded_zarr_store_flush_dirty_shard_not_found():
    """Test RuntimeError when dirty shard not found in cache during flush"""
    from unittest.mock import patch

    from multiformats import CID

    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def save(self, data, codec=None):
            return CID.decode(
                "bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm"
            )

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=2,
    )

    # First put a shard in the cache and mark it as dirty
    test_cid = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")
    shard_data = [test_cid, None]
    await store._shard_data_cache.put(0, shard_data, is_dirty=True)

    # Verify the shard is dirty
    assert store._shard_data_cache.dirty_cache_size == 1
    assert 0 in store._shard_data_cache._dirty_shards

    # Mock the cache.get to return None for the dirty shard (simulating cache corruption)
    original_get = store._shard_data_cache.get

    async def mock_get_returns_none(shard_idx):
        if shard_idx == 0:  # Return None for the dirty shard
            return None
        return await original_get(shard_idx)

    with patch.object(
        store._shard_data_cache, "get", side_effect=mock_get_returns_none
    ):
        # This should hit line 529 (RuntimeError for dirty shard not found in cache)
        with pytest.raises(RuntimeError, match="Dirty shard 0 not found in cache"):
            await store.flush()


@pytest.mark.asyncio
async def test_sharded_zarr_store_failed_to_load_or_initialize_shard():
    """Test RuntimeError when shard fails to load or initialize"""
    from unittest.mock import patch

    from multiformats import CID

    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            import dag_cbor

            return dag_cbor.encode([None] * 4)

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=4,
    )

    # Set up a shard CID to fetch
    test_cid = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")
    store._root_obj["chunks"]["shard_cids"][0] = test_cid

    # Mock the cache to always return None, even after put operations
    async def mock_get_always_none(shard_idx):
        return None  # Always return None to simulate cache failure

    async def mock_put_does_nothing(shard_idx, shard_data, is_dirty=False):
        pass  # Do nothing, so cache remains empty

    with patch.object(store._shard_data_cache, "get", side_effect=mock_get_always_none):
        with patch.object(
            store._shard_data_cache, "put", side_effect=mock_put_does_nothing
        ):
            with pytest.raises(
                RuntimeError, match="Failed to load or initialize shard 0"
            ):
                await store._load_or_initialize_shard_cache(0)


@pytest.mark.asyncio
async def test_sharded_zarr_store_timeout_cleanup_logic():
    """Test timeout cleanup logic in _load_or_initialize_shard_cache"""
    import asyncio
    from unittest.mock import patch

    from multiformats import CID

    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            # Never completes to simulate timeout
            await asyncio.sleep(100)

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=4,
    )

    # Set up a shard CID
    test_cid = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")
    store._root_obj["chunks"]["shard_cids"][0] = test_cid

    # Manually create a pending load event to simulate the scenario
    pending_event = asyncio.Event()
    store._pending_shard_loads[0] = pending_event

    # Verify the pending load is set up
    assert 0 in store._pending_shard_loads
    assert not store._pending_shard_loads[0].is_set()

    # Mock wait_for to properly await the coroutine but still raise TimeoutError
    async def mock_wait_for(coro, timeout=None):
        # Properly cancel the coroutine to avoid the warning
        if hasattr(coro, "close"):
            coro.close()
        raise asyncio.TimeoutError()

    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        with pytest.raises(RuntimeError, match="Timeout waiting for shard 0 to load"):
            await store._load_or_initialize_shard_cache(0)

    # The event should be set and removed from pending loads
    assert 0 not in store._pending_shard_loads  # Should be cleaned up


@pytest.mark.asyncio
async def test_sharded_zarr_store_pending_load_cache_miss():
    """Test RuntimeError when pending load completes but shard not found in cache"""
    import asyncio
    from unittest.mock import patch

    from multiformats import CID

    from py_hamt import ShardedZarrStore

    class MockCAS:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        async def load(self, cid):
            import dag_cbor

            return dag_cbor.encode([None] * 4)

    mock_cas = MockCAS()
    store = await ShardedZarrStore.open(
        cas=mock_cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=4,
    )

    # Set up a shard CID
    test_cid = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")
    store._root_obj["chunks"]["shard_cids"][0] = test_cid

    # Create a pending load event and manually add it
    pending_event = asyncio.Event()
    store._pending_shard_loads[0] = pending_event

    # Set up mocks: wait_for succeeds (doesn't timeout) but cache.get returns None
    async def mock_wait_for(coro, timeout=None):
        # Properly handle the coroutine to avoid warnings
        if hasattr(coro, "close"):
            coro.close()
        # Simulate successful wait - the pending event gets set
        pending_event.set()
        return True  # Successful wait

    async def mock_cache_get(shard_idx):
        # Always return None to simulate cache miss after pending load
        return None

    # Test the scenario where pending load "completes" but shard not in cache (line 428-430)
    with patch("asyncio.wait_for", side_effect=mock_wait_for):
        with patch.object(store._shard_data_cache, "get", side_effect=mock_cache_get):
            with pytest.raises(
                RuntimeError,
                match="Shard 0 not found in cache after pending load completed",
            ):
                await store._load_or_initialize_shard_cache(0)
