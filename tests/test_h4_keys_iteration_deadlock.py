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


@pytest.mark.asyncio
async def test_keys_iteration_uses_snapshot_when_mutated_between_yields() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS(), max_bucket_size=1)
    original_values = {f"key-{index}": f"value-{index}".encode() for index in range(20)}
    for key, value in original_values.items():
        await hamt.set(key, value)

    keys_iterator = hamt.keys()
    first_key = await anext(keys_iterator)

    deleted_key = next(key for key in original_values if key != first_key)
    await asyncio.gather(
        hamt.delete(deleted_key),
        hamt.set("added", b"new value"),
    )
    iterated_keys = {first_key, *[key async for key in keys_iterator]}

    assert iterated_keys == set(original_values)

    expected_values = original_values | {"added": b"new value"}
    del expected_values[deleted_key]
    assert set([key async for key in hamt.keys()]) == set(expected_values)
    for key, expected_value in expected_values.items():
        assert await hamt.get(key) == expected_value
