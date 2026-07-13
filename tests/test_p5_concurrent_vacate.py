"""Regression coverage for concurrent HAMT cache vacating.

Vacating must flush independent sibling subtrees concurrently without violating
the required children-before-parents ordering; the root must still be saved last.
"""

import asyncio
from typing import NamedTuple

import pytest
import pytest_asyncio

from py_hamt import InMemoryCAS
from py_hamt.hamt import HAMT
from py_hamt.store_httpx import ContentAddressedStore

NUMBER_OF_KEYS = 300
GOLDEN_ROOT_HEX = "1e205c7ba99421c5013ad193acef6f7ee70140051bdbf32d4018d7351b8e3b9f3765"


class DelayedInMemoryCAS(InMemoryCAS):
    """An InMemoryCAS that exposes concurrent save calls without changing IDs."""

    def __init__(self) -> None:
        super().__init__()
        self.in_flight = 0
        self.max_in_flight = 0

    def reset_concurrency_counters(self) -> None:
        self.in_flight = 0
        self.max_in_flight = 0

    async def save(
        self,
        data: bytes,
        codec: ContentAddressedStore.CodecInput,
    ) -> bytes:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            await asyncio.sleep(0.01)
            return await super().save(data, codec)
        finally:
            self.in_flight -= 1


class VacateResult(NamedTuple):
    cas: DelayedInMemoryCAS
    root_node_id: bytes
    max_in_flight: int


def key_value(index: int) -> tuple[str, bytes]:
    return f"key-{index}", f"value-{index}".encode()


@pytest_asyncio.fixture(scope="module", loop_scope="module")
async def vacate_result() -> VacateResult:
    cas = DelayedInMemoryCAS()
    hamt = await HAMT.build(
        cas=cas,
        max_bucket_size=1,
        values_are_bytes=True,
    )
    for index in range(NUMBER_OF_KEYS):
        key, value = key_value(index)
        await hamt.set(key, value)

    cas.reset_concurrency_counters()
    await hamt.cache_vacate()

    return VacateResult(
        cas=cas,
        root_node_id=bytes(hamt.root_node_id),
        max_in_flight=cas.max_in_flight,
    )


@pytest.mark.asyncio
async def test_vacate_golden_root_and_readable(vacate_result: VacateResult) -> None:
    assert vacate_result.root_node_id.hex() == GOLDEN_ROOT_HEX

    read_hamt = await HAMT.build(
        cas=vacate_result.cas,
        root_node_id=vacate_result.root_node_id,
        read_only=True,
        values_are_bytes=True,
    )
    for index in range(NUMBER_OF_KEYS):
        key, value = key_value(index)
        assert await read_hamt.get(key) == value


def test_vacate_flushes_siblings_concurrently(vacate_result: VacateResult) -> None:
    assert vacate_result.max_in_flight > 1, (
        "vacate() should overlap CAS saves for independent sibling subtrees; "
        f"observed max in-flight saves: {vacate_result.max_in_flight}"
    )
