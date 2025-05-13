import asyncio
import random
from collections.abc import MutableMapping

import dag_cbor
from dag_cbor import IPLDKind
from multiformats import CID
from hypothesis import given
from hypothesis import strategies as st
import pytest

from py_hamt import HAMT, DictStore
from py_hamt.hamt import Node

from testing_utils import key_value_list

@pytest.mark.asyncio
@given(key_value_list)
async def test_fuzz(kvs: list[tuple[str, IPLDKind]]):
    store = DictStore()
    hamt = await HAMT.build(store=store)

    # Sequential setting and getting, while randomly changing the bucket size which the HAMT should be able to withstand
    for key, value in kvs:
        hamt.max_bucket_size = random.randint(1, 10)
        await hamt.set(key, value)
        assert (await hamt.get(key)) == value

    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == len(kvs)
    await asyncio.gather(*[hamt.delete(k) for k, _ in kvs]) # delete everything
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == 0

    # Set entirely concurrently thus with unpredictable order
    await asyncio.gather(*[hamt.set(k, v) for k, v in kvs])
    await hamt.make_read_only()
    await hamt.enable_write()
    await hamt.make_read_only()

    # Verify completely concurrently
    async def verify(k, v):
        assert (await hamt.get(k)) == v

    await asyncio.gather(*[verify(k, v) for k, v in kvs])

    await hamt.enable_write()
    # Delete and empty it out again
    await asyncio.gather(*[hamt.delete(k) for k, _ in kvs])
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == 0

    # Re insert all items but now with a bucket size that forces linking, which actually runs the link following code branches, otherwise we would miss 100% code coverage
    hamt.max_bucket_size = 1
    await asyncio.gather(*[hamt.set(k, v) for k, v in kvs])
    await hamt.make_read_only() # cover code branch in keys() that does not need to acquire lock for read only mode
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == len(kvs)
    await hamt.enable_write()

    # HAMT should be throwing errors on keys that do not exist
    ks = [k for k, _ in kvs]
    key_that_cannot_exist = "".join(ks).join(
        "string to account for empty string key case"
    )
    with pytest.raises(KeyError):
        await hamt.get(key_that_cannot_exist)
    with pytest.raises(KeyError):
        await hamt.delete(key_that_cannot_exist)

    hamt_keys = [key async for key in hamt.keys()]
    hamt_key_set = set(hamt_keys)
    keys_set = set()
    for key, _ in kvs:
        keys_set.add(key)

    assert hamt_key_set == keys_set

    # Make sure all pointers actually exist in the store, this should not raise any exceptions
    for k,_ in kvs:
        # Callers must handle acquiring a lock themselves, so do these entirely sequentially
        pointer: bytes = await hamt._get_pointer(k) # type: ignore we know that DictStore only returns bytes for its Link type
        await store.load(pointer)

    await hamt.make_read_only()
    with pytest.raises(Exception, match="Cannot call set on a read only HAMT"):
        await hamt.set("foo", "bar")
    with pytest.raises(Exception, match="Cannot call delete on a read only HAMT"):
        await hamt.delete("foo")

    # Test that when a hamt has its root manually initialized by a user, that the key count is accurate
    # We can only get the root node id here since hamt is in read mode and thus the in memory tree has been entirely flushed
    new_hamt = await HAMT.build(store=store, root_node_id=hamt.root_node_id)
    assert len([key async for key in new_hamt.keys()]) == (await new_hamt.len()) == len(kvs)

key_bytes_list = st.lists(
    st.tuples(st.text(), st.binary()),
    min_size=0,
    max_size=10000,
    unique_by=lambda x: x[
        0
    ],  # ensure unique keys, otherwise we can't do the length and size checks when using these KVs for the HAMT
)

@pytest.mark.asyncio
@given(kvs=key_bytes_list)
async def test_values_are_bytes(kvs: list[tuple[str, bytes]]):
    store = DictStore()
    hamt = await HAMT.build(store=store, values_are_bytes=True)

    await asyncio.gather(*[hamt.set(k, v) for k, v in kvs])
    for k,v in kvs:
        value = await hamt.get(k)
        assert value == v
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == len(kvs)
    await asyncio.gather(*[hamt.delete(k) for k, _ in kvs]) # delete everything
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == 0

# # Mostly for complete code coverage's sake
# def test_remaining_exceptions():
#     memory_store = DictStore()
#     with pytest.raises(Exception, match="ID not found in store"):
#         memory_store.load(b"foo")

#     with pytest.raises(ValueError, match="Data not a valid Node serialization"):
#         bad_serialization = dag_cbor.encode([])
#         Node.deserialize(bad_serialization)

#     hamt = HAMT(
#         store=memory_store,
#     )
#     root_node = Node.deserialize(memory_store.load(hamt.root_node_id))  # type: ignore
#     buckets = root_node.get_buckets()
#     links = root_node.get_links()
#     # 4 is the map key for the string "foo"
#     buckets["4"] = []
#     links["4"] = b"bar"
#     bad_node_id = memory_store.save(root_node.serialize())
#     bad_hamt = HAMT(store=memory_store, root_node_id=bad_node_id)

#     with pytest.raises(
#         Exception,
#         match="Key in both buckets and links of the node, invariant violated",
#     ):
#         bad_hamt["foo"] = b"bar2"

#     with pytest.raises(
#         Exception,
#         match="Key in both buckets and links of the node, invariant violated",
#     ):
#         del bad_hamt["foo"]


# def test_key_rewrite():
#     hamt = HAMT(store=DictStore())
#     hamt["foo"] = b"bar"
#     assert b"bar" == hamt["foo"]
#     assert len(hamt) == 1

#     hamt["foo"] = bytes("something else", "utf-8")
#     assert len(hamt) == 1
#     assert b"something else" == hamt["foo"]

#     hamt["foo"] = CID("base32", 1, "dag-cbor", ("blake3", bytes(32)))
#     assert len(hamt) == 1


# async def test_cache_clear():
#     hamt = await HAMT.build(store=DictStore(), read_cache_limit=10000)
#     await asyncio.gather(*[hamt.set(str(key_int), key_int) for key_int in range(50)])


# # Test that is guaranteed to induce overfull buckets that then requires our hamt to follow deeper into the tree to do insertions
# async def test_link_following():
#     hamt = await HAMT.build(store=DictStore())
#     hamt.max_bucket_size = 1
#     # The first byte of the blake3 hash of each of these is the same, 10110011
#     kvs = [(str(5), b""), (str(15), b""), (str(123), b"")]
#     for k, v in kvs:
#         await hamt.set(k, v)
#     assert (await hamt.len()) == 3
#     for k, _ in kvs:
#         await hamt.delete(k)
#     assert (await hamt.len()) == 0


# Run this with varying cache sizes to see the impact on performance of the cache when using IPFSStore()
# Commented out since this increases test time a lot
# def test_and_print_perf():
#     import time
#     num_ops = 50
#     # usual cache size
#     hamt = HAMT(store=IPFSStore())
#     start_time = time.time()
#     for key_int in range(num_ops):
#         hamt[str(key_int)] = key_int
#     end_time = time.time()
#     op_avg_cache = (end_time - start_time) / 100
#     # no cache
#     hamt = HAMT(store=IPFSStore(), max_cache_size_bytes=0)
#     start_time = time.time()
#     for key_int in range(num_ops):
#         hamt[str(key_int)] = key_int
#     end_time = time.time()
#     op_avg_no_cache = (end_time - start_time) / 100
#     print(f"Improvement of {(1 - op_avg_cache / op_avg_no_cache) * 100:.2f}%")


# Sometimes useful to lace throughout the test lines when debugging cache problems
# def find_cache_incosistency(hamt: HAMT):
#     for id in hamt.cache:
#         cache_node = hamt.cache[id]
#         store_node = Node.deserialize(hamt.store.load(id))
#         if cache_node.data != store_node.data:
#             print("*** Inconsistency found")
#             print(f"Cache Node: {cache_node.data}")
#             print(f"Store Node: {store_node.data}")
#             print(f"HAMT cache: {hamt.cache}")
#             print(f"HAMT id_to_time: {hamt.id_to_time}")
#             print(f"HAMT last accessed time: {hamt.last_accessed_times}")
#             raise Exception("Cache inconsistency")
