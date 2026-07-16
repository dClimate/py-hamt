from collections.abc import AsyncIterator

import pytest

from py_hamt import HAMT, InMemoryCAS


@pytest.mark.asyncio
async def test_len_does_not_materialize_key_snapshot_via_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    for index in range(50):
        await hamt.set(f"key-{index}", f"value-{index}")

    async def keys_must_not_be_called() -> AsyncIterator[str]:
        raise AssertionError("len() must not materialize a key snapshot via keys()")
        yield

    monkeypatch.setattr(hamt, "keys", keys_must_not_be_called)

    assert await hamt.len() == 50


@pytest.mark.asyncio
async def test_len_matches_set_and_delete_operations_in_write_and_read_only_modes() -> (
    None
):
    hamt = await HAMT.build(cas=InMemoryCAS())

    for index in range(12):
        await hamt.set(f"key-{index}", f"value-{index}")
    assert await hamt.len() == 12

    await hamt.delete("key-2")
    await hamt.delete("key-9")
    assert await hamt.len() == 10

    await hamt.make_read_only()
    assert await hamt.len() == 10
