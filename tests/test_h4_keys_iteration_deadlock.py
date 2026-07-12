import asyncio

import pytest

from py_hamt import HAMT, InMemoryCAS


@pytest.mark.asyncio
async def test_set_inside_keys_iteration_does_not_deadlock() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    await hamt.set("existing", b"value")

    async def iterate_and_set() -> None:
        async for key in hamt.keys():
            if key == "existing":
                await hamt.set("added", b"new value")

    try:
        await asyncio.wait_for(iterate_and_set(), timeout=5)
    except TimeoutError:
        pytest.fail(
            "DEADLOCK: set() inside hamt.keys() iteration did not complete "
            "within 5 seconds"
        )

    assert await hamt.get("existing") == b"value"
    assert await hamt.get("added") == b"new value"


@pytest.mark.asyncio
async def test_get_inside_keys_iteration_does_not_deadlock() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS())
    expected_values = {"first": b"one", "second": b"two"}
    for key, value in expected_values.items():
        await hamt.set(key, value)

    values_seen: dict[str, bytes] = {}

    async def iterate_and_get() -> None:
        async for key in hamt.keys():
            value = await hamt.get(key)
            assert isinstance(value, bytes)
            values_seen[key] = value

    try:
        await asyncio.wait_for(iterate_and_get(), timeout=5)
    except TimeoutError:
        pytest.fail(
            "DEADLOCK: get() inside hamt.keys() iteration did not complete "
            "within 5 seconds"
        )

    assert values_seen == expected_values
