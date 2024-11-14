from copy import deepcopy
from typing_extensions import MutableMapping

import dag_cbor
from hypothesis import given, strategies as st
import pytest

from py_hamt import HAMT, blake3_hashfn, DictStore
from py_hamt.hamt import Node

memory_store = DictStore()

key_value_lists = st.lists(
    st.tuples(st.text(), st.binary()),
    min_size=0,
    max_size=1000000,
    unique_by=lambda x: x[0],  # ensure unique keys
)


@given(key_value_lists)
def test_fuzz(kvs):
    """Test that all inserted items can be retrieved correctly."""
    # Use a smaller max bucket size to make overfull buckets more likely
    for bucket_size in [1, 2]:
        hamt = HAMT(
            store=memory_store,
            hash_fn=blake3_hashfn,
            max_bucket_size=bucket_size,
            read_only=False,
        )
        assert isinstance(hamt, MutableMapping)

        # Insert all items
        for key, value in kvs:
            hamt[key] = value

        assert len(hamt) == len(kvs)

        ks = [k for k, _ in kvs]
        key_that_cannot_exist = "".join(ks).join(
            "string to account for empty string key case"
        )
        with pytest.raises(KeyError):
            hamt[key_that_cannot_exist]
        with pytest.raises(KeyError):
            del hamt[key_that_cannot_exist]

        for key, value in kvs:
            assert hamt[key] == value

        hamt_keys = list(hamt)
        assert len(hamt_keys) == len(kvs)

        hamt_key_set = set(hamt_keys)
        keys_set = set()
        for key, _ in kvs:
            keys_set.add(key)

        assert hamt_key_set == keys_set

        # Make sure all ids actually exist in the store, this should not raies any exceptions
        store_ids = list(hamt.ids())
        for id in store_ids:
            memory_store.load(id)

        hamt.make_read_only()
        with pytest.raises(Exception, match="Cannot call set on a read only HAMT"):
            hamt["foo"] = b"bar"
        hamt.enable_write()

        hamt.make_read_only()
        with pytest.raises(Exception, match="Cannot call delete on a read only HAMT"):
            del hamt["foo"]
        hamt.enable_write()

        copy_hamt = deepcopy(hamt)

        # Now delete all keys
        for key, _ in kvs:
            del hamt[key]
        assert len(hamt) == 0
        assert len(copy_hamt) == len(kvs)


# Mostly for complete code coverage's sake
def test_remaining_exceptions():
    with pytest.raises(Exception, match="ID not found in store"):
        memory_store.load(b"foo")

    with pytest.raises(ValueError, match="Data not a valid Node serialization"):
        bad_serialization = dag_cbor.encode([])
        Node.deserialize(bad_serialization)

    hamt = HAMT(
        store=memory_store,
    )
    root_node = Node.deserialize(memory_store.load(hamt.root_node_id))
    buckets: dict[str, list[dict[str, bytes]]] = root_node.data["b"]  # type: ignore
    links: dict[str, bytes] = root_node.data["c"]  # type: ignore
    # 224 was found by just testing the hamt with blake3 hash function to see what the hash and thus map key ends up being
    buckets["224"] = []
    links["224"] = b"bar"
    bad_node_id = memory_store.save(root_node.serialize())
    bad_hamt = HAMT(store=memory_store, root_node_id=bad_node_id)

    with pytest.raises(
        Exception, match="Key in both buckets and links of the node, invariant violated"
    ):
        bad_hamt["foo"] = b"bar2"

    with pytest.raises(
        Exception, match="Key in both buckets and links of the node, invariant violated"
    ):
        del bad_hamt["foo"]


def test_key_rewrite():
    hamt = HAMT(store=memory_store)
    hamt["foo"] = b"bar"
    assert b"bar" == hamt["foo"]
    assert len(hamt) == 1

    hamt["foo"] = bytes("something else", "utf-8")
    assert len(hamt) == 1
    assert b"something else" == hamt["foo"]


# Test that is guaranteed to induce overfull buckets that then requires our hamt to follow deeper into the tree to do insertions
def test_link_following():
    hamt = HAMT(store=memory_store, max_bucket_size=1)
    kvs = [("\x0e", b""), ("Ù\x9aÛôå", b""), ("\U000e1d41\U000fef3e\x89", b"")]
    for k, v in kvs:
        hamt[k] = v
    assert len(hamt) == 3
    for k, _ in kvs:
        del hamt[k]
    assert len(hamt) == 0
