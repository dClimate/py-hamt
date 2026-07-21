from copy import deepcopy as stdlib_deepcopy
from typing import Any

import pytest

from py_hamt import HAMT, InMemoryCAS
from py_hamt import hamt as hamt_module


def two_byte_prefix_hash(data: bytes) -> bytes:
    """Force a three-node path while retaining deterministic key distribution."""
    return b"\x00\x00" + hamt_module.blake3_hashfn(data)[:30]


def one_byte_colliding_hash(_: bytes) -> bytes:
    return b"\x00"


def cascading_overflow_hash(data: bytes) -> bytes:
    """Collide twice before distributing keys at the third tree level."""
    return b"\x00\x00" + data


@pytest.mark.asyncio
async def test_happy_path_overwrites_do_not_deepcopy_the_traversal_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamt = await HAMT.build(
        cas=InMemoryCAS(),
        hash_fn=two_byte_prefix_hash,
        values_are_bytes=True,
    )
    key_count = 3000
    for index in range(key_count):
        await hamt.set(f"key-{index}", f"value-{index}".encode())

    deepcopy_call_count = 0

    def counting_deepcopy(value: Any, memo: dict[int, Any] | None = None) -> Any:
        nonlocal deepcopy_call_count
        deepcopy_call_count += 1
        return stdlib_deepcopy(value, memo)

    monkeypatch.setattr(hamt_module, "deepcopy", counting_deepcopy)

    happy_set_count = 128
    for index in range(happy_set_count):
        await hamt.set(f"key-{index}", f"updated-{index}".encode())

    deepcopy_bound = 2 * happy_set_count
    assert deepcopy_call_count <= deepcopy_bound, (
        f"happy-path overwrites made {deepcopy_call_count} deepcopy calls; "
        f"expected at most {deepcopy_bound} for {happy_set_count} sets"
    )


@pytest.mark.asyncio
async def test_write_perf_fix_preserves_golden_root_cid() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS(), values_are_bytes=True)
    for index in range(512):
        await hamt.set(f"k{index}", f"value-{index}".encode())

    await hamt.make_read_only()

    assert isinstance(hamt.root_node_id, bytes)
    # Captured from current branch code at commit f0c2171. This guards
    # serialized-output stability while the H6 write-performance fix is applied.
    golden_root_cid = (
        "1e2006c31744d8396c93ff78f618082c2fd5b07bf5f3a8d328a54d14dc11cae79165"
    )
    assert hamt.root_node_id.hex() == golden_root_cid


@pytest.mark.asyncio
async def test_detached_subtree_cascading_overflow_preserves_all_values() -> None:
    hamt = await HAMT.build(
        cas=InMemoryCAS(),
        hash_fn=cascading_overflow_hash,
        max_bucket_size=1,
        values_are_bytes=True,
    )
    expected_values = {"a": b"value-a", "b": b"value-b"}

    for key, value in expected_values.items():
        await hamt.set(key, value)

    for key, value in expected_values.items():
        assert await hamt.get(key) == value


@pytest.mark.asyncio
async def test_failed_bucket_reflow_remains_atomic_and_writable() -> None:
    hamt = await HAMT.build(
        cas=InMemoryCAS(),
        hash_fn=one_byte_colliding_hash,
        max_bucket_size=2,
        values_are_bytes=True,
    )
    committed_values = {"a": b"value-a", "b": b"value-b"}
    for key, value in committed_values.items():
        await hamt.set(key, value)

    with pytest.raises(IndexError):
        await hamt.set("c", b"value-c")

    for key, value in committed_values.items():
        assert await hamt.get(key) == value

    await hamt.set("a", b"updated-a")
    assert await hamt.get("a") == b"updated-a"
