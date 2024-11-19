from copy import deepcopy
import random
from collections.abc import MutableMapping

import dag_cbor
from multiformats import CID
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import pytest

from py_hamt import HAMT, blake3_hashfn, DictStore
from py_hamt.hamt import Node


def cid_strategy() -> SearchStrategy:
    """Generate random CIDs for testing."""

    # Strategy for generating random hash digests
    digests = st.binary(min_size=32, max_size=32).map(
        lambda d: bytes.fromhex("1220") + d  # 0x12 = sha2-256, 0x20 = 32 bytes
    )

    # Generate CIDv1 (more flexible)
    cidv1 = st.tuples(st.just("base32"), st.just(1), st.just("dag-cbor"), digests)

    # Combine the strategies and create CIDs
    return cidv1.map(lambda args: CID(*args))


def ipld_strategy() -> SearchStrategy:
    return st.one_of(
        [
            st.none(),
            st.booleans(),
            st.integers(min_value=-9223372036854775808, max_value=9223372036854775807),
            st.floats(allow_infinity=False, allow_nan=False),
            st.text(),
            st.binary(),
            cid_strategy(),
        ]
    )


key_value_lists = st.lists(
    st.tuples(st.text(), ipld_strategy()),
    min_size=0,
    max_size=1000000,
    unique_by=lambda x: x[
        0
    ],  # ensure unique keys, otherwise we can't do the length and size checks
)


@given(key_value_lists)
def test_fuzz(kvs):
    memory_store = DictStore()
    hamt = HAMT(
        store=memory_store,
        hash_fn=blake3_hashfn,
        read_only=False,
    )
    assert isinstance(hamt, MutableMapping)

    # Delete and reinsert but this time with varying bucket size while inserting
    # The HAMT should be able to withstand its bucket size changing randomly while working
    for key, value in kvs:
        hamt.max_bucket_size = random.randint(1, 10)
        hamt[key] = value
    assert len(hamt) == len(kvs)
    for key, _ in kvs:
        del hamt[key]
    assert len(hamt) == 0
    # Re insert all items but now with a bucket size that forces linking, which actually runs the link following code branches, otherwise we would miss 100% code coverage
    hamt.max_bucket_size = 1
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
        memory_store.load(id)  # type: ignore

    hamt.make_read_only()
    with pytest.raises(Exception, match="Cannot call set on a read only HAMT"):
        hamt["foo"] = b"bar"
    hamt.enable_write()

    hamt.make_read_only()
    with pytest.raises(Exception, match="Cannot call delete on a read only HAMT"):
        del hamt["foo"]
    hamt.enable_write()

    copy_hamt = deepcopy(hamt)

    # Modify our current hamt and since the copy is a deep copy, they should be different
    for key, _ in kvs:
        del hamt[key]
    assert len(hamt) == 0
    assert len(copy_hamt) == len(kvs)


# Mostly for complete code coverage's sake
def test_remaining_exceptions():
    memory_store = DictStore()
    with pytest.raises(Exception, match="ID not found in store"):
        memory_store.load(b"foo")

    with pytest.raises(ValueError, match="Data not a valid Node serialization"):
        bad_serialization = dag_cbor.encode([])
        Node.deserialize(bad_serialization)

    hamt = HAMT(
        store=memory_store,
    )
    root_node = Node.deserialize(memory_store.load(hamt.root_node_id))  # type: ignore
    buckets = root_node.get_buckets()
    links = root_node.get_links()
    # 224 was found by just testing the hamt with blake3 hash function to see what the hash and thus map key ends up being
    buckets["224"] = []
    links["224"] = b"bar"
    bad_node_id = memory_store.save(root_node.serialize())
    bad_hamt = HAMT(store=memory_store, root_node_id=bad_node_id)

    with pytest.raises(
        Exception,
        match="Key in both buckets and links of the node, invariant violated",
    ):
        bad_hamt["foo"] = b"bar2"

    with pytest.raises(
        Exception,
        match="Key in both buckets and links of the node, invariant violated",
    ):
        del bad_hamt["foo"]


def test_key_rewrite():
    memory_store = DictStore()
    hamt = HAMT(store=memory_store)
    hamt["foo"] = b"bar"
    assert b"bar" == hamt["foo"]
    assert len(hamt) == 1

    hamt["foo"] = bytes("something else", "utf-8")
    assert len(hamt) == 1
    assert b"something else" == hamt["foo"]

    hamt["foo"] = CID("base32", 1, "dag-cbor", ("blake3", bytes(32)))
    assert len(hamt) == 1


# Test that is guaranteed to induce overfull buckets that then requires our hamt to follow deeper into the tree to do insertions
def test_link_following():
    memory_store = DictStore()
    hamt = HAMT(store=memory_store, max_bucket_size=1)
    kvs = [("\x0e", b""), ("Ù\x9aÛôå", b""), ("\U000e1d41\U000fef3e\x89", b"")]
    for k, v in kvs:
        hamt[k] = v
    assert len(hamt) == 3
    for k, _ in kvs:
        del hamt[k]
    assert len(hamt) == 0
