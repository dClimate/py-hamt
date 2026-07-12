from typing import Literal

import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from py_hamt import HAMT, InMemoryCAS

Operation = tuple[Literal["set", "delete"], str]

# These numeric strings have the same first byte in their BLAKE3 hashes. The
# additional keys exercise interleavings outside that colliding root bucket.
COLLIDING_KEYS = ("5", "15", "123", "317", "349")
KEY_POOL = COLLIDING_KEYS + ("0", "1", "2", "alpha", "omega")
DELETE_AFTER_SPLIT: list[Operation] = [
    *(("set", key) for key in COLLIDING_KEYS),
    *(("delete", key) for key in COLLIDING_KEYS[1:]),
]


@pytest.mark.asyncio
async def test_delete_collapses_split_tree_to_canonical_root() -> None:
    """Canonical deletion changes CIDs only for formerly non-canonical trees."""
    cas = InMemoryCAS()

    mutated_hamt = await HAMT.build(cas=cas, max_bucket_size=1)
    await mutated_hamt.set("5", b"v")
    await mutated_hamt.set("15", b"v")
    await mutated_hamt.delete("15")
    await mutated_hamt.make_read_only()

    fresh_hamt = await HAMT.build(cas=cas, max_bucket_size=1)
    await fresh_hamt.set("5", b"v")
    await fresh_hamt.make_read_only()

    assert await mutated_hamt.get("5") == await fresh_hamt.get("5") == b"v"
    assert mutated_hamt.root_node_id == fresh_hamt.root_node_id, (
        "determinism violation: identical key-value content produced different "
        "root node IDs after delete"
    )


@pytest.mark.parametrize("max_bucket_size", range(1, 5))
@pytest.mark.asyncio
@given(
    operations=st.lists(
        st.tuples(
            st.sampled_from(("set", "delete")),
            st.sampled_from(KEY_POOL),
        ),
        min_size=1,
        max_size=30,
    )
)
@example(operations=DELETE_AFTER_SPLIT)
@settings(max_examples=50, deadline=None, derandomize=True)
async def test_operation_history_does_not_change_root_id(
    max_bucket_size: int, operations: list[Operation]
) -> None:
    cas = InMemoryCAS()
    mutated_hamt = await HAMT.build(cas=cas, max_bucket_size=max_bucket_size)
    surviving_values: dict[str, bytes] = {}

    for operation, key in operations:
        if operation == "set":
            value = f"value:{key}".encode()
            await mutated_hamt.set(key, value)
            surviving_values[key] = value
        elif key in surviving_values:
            await mutated_hamt.delete(key)
            del surviving_values[key]

    await mutated_hamt.make_read_only()

    fresh_hamt = await HAMT.build(cas=cas, max_bucket_size=max_bucket_size)
    for key, value in sorted(surviving_values.items()):
        await fresh_hamt.set(key, value)
    await fresh_hamt.make_read_only()

    assert mutated_hamt.root_node_id == fresh_hamt.root_node_id, (
        "determinism violation: identical surviving key-value content produced "
        f"different root node IDs for max_bucket_size={max_bucket_size}"
    )
