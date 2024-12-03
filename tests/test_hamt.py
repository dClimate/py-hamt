from copy import deepcopy
import random
from collections.abc import MutableMapping

import dag_cbor
from dag_cbor import IPLDKind
from multiformats import CID
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
import pytest

from py_hamt import HAMT, DictStore
from py_hamt.hamt import Node
from py_hamt.store import Store


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
    max_size=10000,
    unique_by=lambda x: x[
        0
    ],  # ensure unique keys, otherwise we can't do the length and size checks
)


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


@given(key_value_lists)
def test_fuzz(kvs: list[tuple[str, IPLDKind]]):
    store = DictStore()
    hamt = HAMT(
        store=store,
    )
    assert isinstance(hamt, MutableMapping)

    # The HAMT should be able to withstand its bucket size changing in between operations
    for key, value in kvs:
        hamt.max_bucket_size = random.randint(1, 10)
        hamt[key] = value
        assert hamt[key] == value
    assert len(hamt) == len(kvs)
    assert len(list(hamt)) == len(hamt)
    for key, _ in kvs:
        del hamt[key]
    assert len(hamt) == 0
    assert len(list(hamt)) == len(hamt)
    # Re insert all items but now with a bucket size that forces linking, which actually runs the link following code branches, otherwise we would miss 100% code coverage
    hamt.max_bucket_size = 1
    for key, value in kvs:
        hamt[key] = value
        assert hamt[key] == value
    assert len(hamt) == len(kvs)
    assert len(list(hamt)) == len(hamt)

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

    assert len(hamt) == len(kvs)
    assert len(list(hamt)) == len(kvs)

    hamt_keys = list(hamt)
    hamt_key_set = set(hamt_keys)
    keys_set = set()
    for key, _ in kvs:
        keys_set.add(key)

    assert hamt_key_set == keys_set

    # Make sure all ids actually exist in the store, this should not raies any exceptions
    store_ids: list[bytes] = list(hamt.ids())  # type: ignore
    for id in store_ids:
        store.load(id)

    hamt.make_read_only()
    with pytest.raises(Exception, match="Cannot call set on a read only HAMT"):
        hamt["foo"] = b"bar"
    with pytest.raises(Exception, match="Cannot call delete on a read only HAMT"):
        del hamt["foo"]
    hamt.enable_write()

    copy_hamt = deepcopy(hamt)

    # Modify our current hamt and since the copy is a deep copy, they should be different
    for key, _ in kvs:
        del hamt[key]
    assert len(hamt) == 0
    assert len(copy_hamt) == len(kvs)

    # Test that when a hamt has its root manually initialized by a user, that the key count is accurate
    hamt_re_initialized = HAMT(store=store, root_node_id=copy_hamt.root_node_id)
    assert len(hamt_re_initialized) == len(kvs)


class BadStore(Store):
    """ "
    Bad store implementation that should cause errors
    """

    def save_raw(self, data: bytes) -> IPLDKind:
        return 0

    def save_dag_cbor(self, data: bytes) -> IPLDKind:
        return 0

    def load(self, id: IPLDKind) -> bytes:
        return bytes(0)


def test_bad_store():
    bad_store = BadStore()
    hamt = HAMT(store=bad_store, max_cache_size_bytes=0)
    with pytest.raises(
        Exception,
        match="Invalid dag-cbor encoded data from the store was attempted to be decoded",
    ):
        hamt["test"] = 3


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
    hamt = HAMT(store=DictStore())
    hamt["foo"] = b"bar"
    assert b"bar" == hamt["foo"]
    assert len(hamt) == 1

    hamt["foo"] = bytes("something else", "utf-8")
    assert len(hamt) == 1
    assert b"something else" == hamt["foo"]

    hamt["foo"] = CID("base32", 1, "dag-cbor", ("blake3", bytes(32)))
    assert len(hamt) == 1


def test_cache_clear():
    hamt = HAMT(store=DictStore(), max_cache_size_bytes=1000)
    for key_int in range(50):
        hamt[str(key_int)] = key_int


# Test that is guaranteed to induce overfull buckets that then requires our hamt to follow deeper into the tree to do insertions
def test_link_following():
    store = DictStore()
    hamt = HAMT(store=store)
    hamt.max_bucket_size = 1
    kvs = [("\x0e", b""), ("Ù\x9aÛôå", b""), ("\U000e1d41\U000fef3e\x89", b"")]
    for k, v in kvs:
        hamt[k] = v
    assert len(hamt) == 3
    for k, _ in kvs:
        del hamt[k]
    assert len(hamt) == 0


# Run this with varying cache sizes to see the impact on performance of the cache when using IPFSStore()
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
