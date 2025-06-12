import asyncio
import random
from typing import cast

import pytest
from dag_cbor import IPLDKind
from hypothesis import given, settings
from hypothesis import strategies as st
from multiformats import CID
from testing_utils import key_value_list

from py_hamt import InMemoryCAS
from py_hamt.hamt import HAMT, Node


@pytest.mark.asyncio
@given(key_value_list)
@settings(
    deadline=1000
)  # increase for github CI which sometimes takes longer than the default 250 ms
async def test_fuzz(kvs: list[tuple[str, IPLDKind]]) -> None:
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas)

    # Sequential setting and getting, while randomly changing the bucket size which the HAMT should be able to withstand
    for key, value in kvs:
        hamt.max_bucket_size = random.randint(1, 10)
        await hamt.set(key, value)
        assert (await hamt.get(key)) == value

    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == len(kvs)
    await asyncio.gather(*[hamt.delete(k) for k, _ in kvs])  # delete everything
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == 0

    # Set entirely concurrently thus with unpredictable order
    await asyncio.gather(*[hamt.set(k, v) for k, v in kvs])
    await hamt.make_read_only()
    await hamt.enable_write()
    await hamt.make_read_only()

    # Verify completely concurrently
    async def verify(k: str, v: IPLDKind):
        assert (await hamt.get(k)) == v

    await asyncio.gather(*[verify(k, v) for k, v in kvs])

    await hamt.enable_write()
    # Delete and empty it out again
    await asyncio.gather(*[hamt.delete(k) for k, _ in kvs])
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == 0

    # Re insert all items but now with a bucket size that forces linking, which actually runs the link following code branches, otherwise we would miss 100% code coverage
    hamt.max_bucket_size = 1
    await asyncio.gather(*[hamt.set(k, v) for k, v in kvs])
    await hamt.make_read_only()  # cover code branch in keys() that does not need to acquire lock for read only mode
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == len(kvs)
    await hamt.enable_write()

    # HAMT should be throwing errors on keys that do not exist
    ks: list[str] = [k for k, _ in kvs]
    key_that_cannot_exist = "".join(ks).join(
        "string to account for empty string key case"
    )
    with pytest.raises(KeyError):
        await hamt.get(key_that_cannot_exist)
    with pytest.raises(KeyError):
        await hamt.delete(key_that_cannot_exist)

    hamt_keys: list[str] = [key async for key in hamt.keys()]
    hamt_key_set: set[str] = set(hamt_keys)
    keys_set: set[str] = set()
    for key, _ in kvs:
        keys_set.add(key)

    assert hamt_key_set == keys_set

    # Make sure all pointers actually exist in the store, this should not raise any exceptions
    for k, _ in kvs:
        pointer: bytes = cast(
            bytes, await hamt.get_pointer(k)
        )  # we know that InMemoryCAS only returns bytes for its IDs
        await cas.load(pointer)

    await hamt.make_read_only()
    with pytest.raises(Exception, match="Cannot call set on a read only HAMT"):
        await hamt.set("foo", "bar")
    with pytest.raises(Exception, match="Cannot call delete on a read only HAMT"):
        await hamt.delete("foo")

    async def verify_kvs_and_len(h: HAMT) -> None:
        for k, v in kvs:
            assert (await h.get(k)) == v
        assert len([key async for key in h.keys()]) == (await h.len()) == len(kvs)

    # Test that when a hamt has its root manually initialized by a user, that the key count is accurate
    # We can only get the root node id here since hamt is in read mode and thus the in memory tree has been entirely flushed
    new_hamt = await HAMT.build(cas=cas, root_node_id=hamt.root_node_id, read_only=True)
    await verify_kvs_and_len(new_hamt)

    # Most for code coverage's sake
    with pytest.raises(
        Exception,
        match="Node was attempted to be written to the read cache",
    ):
        await new_hamt.node_store.save(bytes(), Node())

    # None of the keys should be in a new hamt using the same store but with a fresh root node
    empty_hamt = await HAMT.build(cas=cas)
    for k, _ in kvs:
        with pytest.raises(KeyError):
            await empty_hamt.get(k)
        assert (
            len([key async for key in empty_hamt.keys()])
            == (await empty_hamt.len())
            == 0
        )

    # invalid bucket size
    with pytest.raises(
        ValueError, match="Bucket size maximum must be a positive integer"
    ):
        await HAMT.build(cas=cas, max_bucket_size=-1)

    # Cache size management code branches coverage, do it all in async concurrency

    small_cache_size_bytes: int = 1000
    # Read cache
    read_hamt = await HAMT.build(
        cas=cas, root_node_id=hamt.root_node_id, read_only=True
    )

    async def get_and_vacate(k: str, v: IPLDKind) -> None:
        assert (await read_hamt.get(k)) == v
        if (await read_hamt.cache_size()) > small_cache_size_bytes:
            await read_hamt.cache_vacate()
            assert (await read_hamt.cache_size()) == 0

    await asyncio.gather(*[get_and_vacate(k, v) for k, v in kvs])

    # In memory tree while writing
    # set small max bucket size to force more linking and more nodes
    small_memory_tree = await HAMT.build(cas=cas, max_bucket_size=1)

    async def set_and_vacate(k: str, v: IPLDKind) -> None:
        await small_memory_tree.set(k, v)
        assert (await small_memory_tree.get(k)) == v
        if (await small_memory_tree.cache_size()) > small_cache_size_bytes:
            await small_memory_tree.cache_vacate()
            assert (await small_memory_tree.cache_size()) == 0
        assert (await small_memory_tree.get(k)) == v

    await asyncio.gather(*[set_and_vacate(k, v) for k, v in kvs])


@pytest.mark.asyncio
# This is a list of keys to arbitrary bytes
@given(
    kvs=st.lists(
        st.tuples(st.text(), st.binary()),
        min_size=0,
        max_size=10000,
        unique_by=lambda x: x[
            0
        ],  # ensure unique keys, otherwise we can't do the length and size checks when using these KVs for the HAMT
    )
)
async def test_values_are_bytes(kvs: list[tuple[str, bytes]]):
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)

    await asyncio.gather(*[hamt.set(k, v) for k, v in kvs])
    for k, v in kvs:
        value = await hamt.get(k)
        assert value == v
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == len(kvs)
    await asyncio.gather(*[hamt.delete(k) for k, _ in kvs])  # delete everything
    assert len([key async for key in hamt.keys()]) == (await hamt.len()) == 0


@pytest.mark.asyncio
async def test_invalid_node():
    with pytest.raises(
        Exception,
        match="Invalid dag-cbor encoded data from the store was attempted to be decoded",
    ):
        Node.deserialize(bytes())


@pytest.mark.asyncio
async def test_key_rewrite():
    hamt = await HAMT.build(cas=InMemoryCAS())
    await hamt.set("foo", b"bar")
    assert b"bar" == await hamt.get("foo")
    assert (await hamt.len()) == 1

    await hamt.set("foo", bytes("something else", "utf-8"))
    assert (await hamt.len()) == 1
    assert b"something else" == (await hamt.get("foo"))

    await hamt.set("foo", CID("base32", 1, "dag-cbor", ("blake3", bytes(32))))
    assert (await hamt.len()) == 1


# Test that is guaranteed to induce overfull buckets that then requires our hamt to follow deeper into the tree to do insertions
@pytest.mark.asyncio
async def test_link_following():
    hamt = await HAMT.build(cas=InMemoryCAS())
    hamt.max_bucket_size = 1
    # The first byte of the blake3 hash of each of these is the same, 10110011
    kvs = [(str(5), b""), (str(15), b""), (str(123), b"")]
    for k, v in kvs:
        await hamt.set(k, v)
    assert (await hamt.len()) == 3
    for k, _ in kvs:
        await hamt.delete(k)
    assert (await hamt.len()) == 0
