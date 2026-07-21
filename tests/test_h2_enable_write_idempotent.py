import pytest

from py_hamt import HAMT, InMemoryCAS


@pytest.mark.asyncio
@pytest.mark.parametrize("redundant_calls", [1, 2, 3])
async def test_enable_write_is_idempotent_while_already_writable(
    redundant_calls: int,
) -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    await hamt.set("foo", b"bar")

    for _ in range(redundant_calls):
        await hamt.enable_write()

    assert await hamt.get("foo") == b"bar"
    assert [key async for key in hamt.keys()] == ["foo"]
    assert await hamt.len() == 1
