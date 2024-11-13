import dag_cbor
from hypothesis import given, strategies as st
import pytest

from py_hamt.hamt import Node, Hamt, blake3_hashfn
from py_hamt.dict_store import DictStore

memory_store = DictStore()

key_value_lists = st.lists(
    st.tuples(st.text(), st.binary()),
    min_size=0,
    max_size=10000,
    unique_by=lambda x: x[0],  # ensure unique keys
)


@pytest.mark.asyncio
@given(key_value_lists)
async def test_fuzz(kvs):
    """Test that all inserted items can be retrieved correctly."""
    # Use a smaller max bucket size to make overfull buckets more likely
    for bucket_size in [1, 2]:
        hamt = await Hamt.create(
            store=memory_store,
            hash_fn=blake3_hashfn,
            max_bucket_size=bucket_size,
            read_only=False,
        )

        # Insert all items
        for key, value in kvs:
            await hamt.set(key, value)
            assert await hamt.has(key)

        assert hamt.size() == len(kvs)

        ks = [k for k, _ in kvs]
        key_that_cannot_exist = "".join(ks).join(
            "string to account for empty string key case"
        )
        assert not await hamt.has(key_that_cannot_exist)

        for key, value in kvs:
            assert await hamt.get(key) == value

        hamt_keys = [key async for key in hamt.keys()]
        assert len(hamt_keys) == len(kvs)

        hamt_key_set = set(hamt_keys)
        keys_set = set()
        for key, _ in kvs:
            keys_set.add(key)

        assert hamt_key_set == keys_set

        # Make sure all ids actually exist in the store
        store_ids = [id async for id in hamt.ids()]
        for id in store_ids:
            await memory_store.load(id)

        hamt.make_read_only()
        with pytest.raises(Exception, match="Cannot call set on a read only HAMT"):
            await hamt.set("foo", b"bar")
        hamt.enable_write()

        hamt.make_read_only()
        with pytest.raises(Exception, match="Cannot call delete on a read only HAMT"):
            await hamt.delete("foo")
        hamt.enable_write()

        # Now delete all keys
        await hamt.delete(key_that_cannot_exist)
        for key, _ in kvs:
            await hamt.delete(key)
        assert hamt.size() == 0


# Mostly for complete code coverage's sake
@pytest.mark.asyncio
async def test_remaining_exceptions():
    with pytest.raises(Exception, match="ID not found in store"):
        await memory_store.load(b"foo")

    with pytest.raises(ValueError, match="Data not a valid Node serialization"):
        bad_serialization = dag_cbor.encode([])
        Node.deserialize(bad_serialization)

    hamt = await Hamt.create(
        store=memory_store,
        hash_fn=blake3_hashfn,
        max_bucket_size=5,
        read_only=False,
    )
    root_node = Node.deserialize(await memory_store.load(hamt.root_node_id))
    buckets: dict[str, list[dict[str, bytes]]] = root_node.data["b"]  # type: ignore
    links: dict[str, bytes] = root_node.data["c"]  # type: ignore
    # 224 was found by just testing the hamt with blake3 hash function to see what the hash and thus map key ends up being
    buckets["224"] = []
    links["224"] = b"bar"
    bad_node_id = await memory_store.save(root_node.serialize())
    hamt.root_node_id = bad_node_id

    with pytest.raises(
        Exception, match="Key in both buckets and links of the node, invariant violated"
    ):
        await hamt.set("foo", b"bar2")

    with pytest.raises(
        Exception, match="Key in both buckets and links of the node, invariant violated"
    ):
        await hamt.delete("foo")


@pytest.mark.asyncio
async def test_key_rewrite():
    hamt = await Hamt.create(
        store=memory_store, hash_fn=blake3_hashfn, max_bucket_size=5, read_only=False
    )
    await hamt.set("foo", bytes("bar", "utf-8"))
    assert b"bar" == await hamt.get("foo")
    assert hamt.size() == 1

    await hamt.set("foo", bytes("something else", "utf-8"))
    assert hamt.size() == 1
    assert b"something else" == await hamt.get("foo")
