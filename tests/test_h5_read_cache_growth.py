from collections import defaultdict
from typing import DefaultDict, cast

import pytest
from dag_cbor.ipld import IPLDKind

from py_hamt import HAMT, InMemoryCAS
from py_hamt.hamt import InMemoryTreeStore, Node


def retained_node_count(node_store: object) -> int:
    """Count Nodes retained in any dictionary owned by a node store."""
    total = 0
    for attribute in vars(node_store).values():
        if isinstance(attribute, dict) and all(
            isinstance(value, Node) for value in attribute.values()
        ):
            total += len(attribute)
    return total


class CountingInMemoryCAS(InMemoryCAS):
    def __init__(self) -> None:
        super().__init__()
        self.load_calls: DefaultDict[IPLDKind, int] = defaultdict(int)

    async def load(
        self,
        id: IPLDKind,
        offset: int | None = None,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes:
        self.load_calls[id] += 1
        return await super().load(id, offset=offset, length=length, suffix=suffix)


@pytest.mark.asyncio
async def test_repeated_get_does_not_grow_buffer() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    for index in range(20):
        await hamt.set(f"key-{index}", f"value-{index}")
    await hamt.cache_vacate()

    assert await hamt.get("key-0") == "value-0"
    count_after_first_get = retained_node_count(hamt.node_store)

    for _ in range(50):
        assert await hamt.get("key-0") == "value-0"
    count_after_repeated_gets = retained_node_count(hamt.node_store)

    assert count_after_repeated_gets == count_after_first_get, (
        "repeated reads retained additional nodes: "
        f"{count_after_first_get} after warm-up, "
        f"{count_after_repeated_gets} after 50 more gets"
    )


@pytest.mark.asyncio
async def test_second_load_of_same_cas_id_hits_cache() -> None:
    cas = CountingInMemoryCAS()
    hamt = await HAMT.build(cas=cas)
    for index in range(4):
        await hamt.set(f"key-{index}", f"value-{index}")
    await hamt.cache_vacate()

    root_node_id = hamt.root_node_id
    calls_before_first_get = cas.load_calls[root_node_id]
    assert await hamt.get("key-0") == "value-0"
    first_get_loads = cas.load_calls[root_node_id] - calls_before_first_get

    calls_before_second_get = cas.load_calls[root_node_id]
    assert await hamt.get("key-0") == "value-0"
    second_get_loads = cas.load_calls[root_node_id] - calls_before_second_get

    assert first_get_loads > 0
    assert second_get_loads == 0, (
        "the second lookup reloaded the same root node from CAS "
        f"{second_get_loads} time(s)"
    )


async def build_reference_hamt() -> tuple[HAMT, dict[str, str]]:
    values = {f"key-{index}": f"value-{index}" for index in range(32)}
    hamt = await HAMT.build(cas=InMemoryCAS())
    for key, value in values.items():
        await hamt.set(key, value)
    await hamt.make_read_only()
    return hamt, values


@pytest.mark.asyncio
async def test_vacate_root_cid_golden() -> None:
    first_hamt, values = await build_reference_hamt()
    second_hamt, _ = await build_reference_hamt()

    assert first_hamt.root_node_id == second_hamt.root_node_id
    for key, value in values.items():
        assert await first_hamt.get(key) == value
        assert await second_hamt.get(key) == value


@pytest.mark.asyncio
async def test_clean_cache_handles_unhashable_ids_and_contributes_to_size() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    node_store = cast(InMemoryTreeStore, hamt.node_store)
    unhashable_id = cast(IPLDKind, [])

    assert node_store.get_clean_node(unhashable_id) is None
    node_store.cache_clean_node(unhashable_id, Node())
    node_store.cache_clean_node(b"existing-node", Node())
    node_store.remove_clean_node(unhashable_id)

    cached_node = Node()
    node_store.cache_clean_node(b"clean-node", cached_node)
    assert node_store.size() >= len(cached_node.serialize())


@pytest.mark.asyncio
async def test_reserialize_prunes_an_empty_child_from_its_parent() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    node_store = cast(InMemoryTreeStore, hamt.node_store)
    root_id = hamt.root_node_id
    root_node = await node_store.load(root_id)
    empty_child = Node()
    child_id = await node_store.add_to_buffer(empty_child)
    child_slot = 7
    root_node.set_link(child_slot, child_id)
    node_stack = [(root_id, root_node), (child_id, empty_child)]
    link_path = [0, child_slot]

    await hamt._reserialize_and_link(node_stack, link_path)

    assert node_stack == [(root_id, root_node)]
    assert link_path == [0]
    assert root_node.data[child_slot] == {}


@pytest.mark.asyncio
async def test_collect_subtree_entries_follows_links_and_honors_limit() -> None:
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas)
    child = Node()
    child.data[0] = {"first": "value", "second": "value"}
    child_id = await cas.save(child.serialize(), codec="dag-cbor")
    parent = Node()
    parent.set_link(3, child_id)

    assert await hamt._collect_subtree_entries(parent, limit=2) == {
        "first": "value",
        "second": "value",
    }
    assert await hamt._collect_subtree_entries(parent, limit=1) is None
