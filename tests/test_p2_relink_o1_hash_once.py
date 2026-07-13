import pytest
from dag_cbor import IPLDKind

from py_hamt import InMemoryCAS
from py_hamt.hamt import HAMT, Node


@pytest.mark.asyncio
async def test_golden_root_id_500_keys() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())

    for index in range(500):
        await hamt.set(f"key-{index}", f"value-{index}")

    await hamt.make_read_only()

    assert bytes(hamt.root_node_id).hex() == (
        "1e20330cb3ccd5ef3940490f30afd6a852e1e7545df16e9ed2b754405c883ba0cab0"
    )


@pytest.mark.asyncio
async def test_get_link_calls_bounded_during_bulk_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    number_of_keys = 3_000
    get_link_calls = 0
    original_get_link = Node.get_link

    def counting_get_link(node: Node, index: int) -> IPLDKind:
        nonlocal get_link_calls
        get_link_calls += 1
        return original_get_link(node, index)

    monkeypatch.setattr(Node, "get_link", counting_get_link)

    hamt = await HAMT.build(cas=InMemoryCAS())
    for index in range(number_of_keys):
        await hamt.set(f"key-{index}", f"value-{index}")

    maximum_get_link_calls = 20 * number_of_keys
    # Pre-fix measurement: 194,838 calls for 3,000 keys (bound: 60,000).
    assert get_link_calls <= maximum_get_link_calls, (
        f"get_link called {get_link_calls} times for {number_of_keys} sets; "
        f"expected at most {maximum_get_link_calls} (20 * N)"
    )
