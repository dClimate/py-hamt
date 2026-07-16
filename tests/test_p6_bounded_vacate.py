import asyncio
from typing import cast

import pytest

from py_hamt import HAMT, ContentAddressedStore, InMemoryCAS
from py_hamt.hamt import InMemoryTreeStore


class InstrumentedCAS(InMemoryCAS):
    def __init__(self) -> None:
        super().__init__()
        self.in_flight_saves = 0
        self.max_concurrent_saves = 0

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> bytes:
        self.in_flight_saves += 1
        self.max_concurrent_saves = max(self.max_concurrent_saves, self.in_flight_saves)
        try:
            await asyncio.sleep(0.002)
            return await super().save(data, codec)
        finally:
            self.in_flight_saves -= 1

    def reset_save_concurrency(self) -> None:
        self.in_flight_saves = 0
        self.max_concurrent_saves = 0


class FailingCAS(InstrumentedCAS):
    def __init__(self, fail_on_save_number: int) -> None:
        super().__init__()
        self.fail_on_save_number = fail_on_save_number
        self.armed = False
        self.armed_save_calls = 0

    def arm(self) -> None:
        self.armed = True
        self.armed_save_calls = 0

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> bytes:
        if self.armed:
            self.armed_save_calls += 1
            if self.armed_save_calls == self.fail_on_save_number:
                # Let the rest of the unbounded wave start before this task fails.
                await asyncio.sleep(0)
                raise ConnectionError("simulated CAS save failure")

        self.in_flight_saves += 1
        self.max_concurrent_saves = max(self.max_concurrent_saves, self.in_flight_saves)
        try:
            await asyncio.sleep(0.01)
            return await InMemoryCAS.save(self, data, codec)
        finally:
            self.in_flight_saves -= 1


async def build_wide_hamt(cas: InMemoryCAS) -> HAMT:
    hamt = await HAMT.build(cas=cas)
    for i in range(2000):
        await hamt.set(str(i), b"v")
    return hamt


@pytest.mark.asyncio
async def test_vacate_save_waves_are_bounded() -> None:
    cas = InstrumentedCAS()
    hamt = await build_wide_hamt(cas)
    cas.reset_save_concurrency()

    async with hamt.lock:
        await cast(InMemoryTreeStore, hamt.node_store).vacate()

    assert cas.max_concurrent_saves <= 16, "vacate save waves must be bounded"


@pytest.mark.asyncio
async def test_failed_vacate_leaves_no_orphan_save_tasks() -> None:
    cas = FailingCAS(fail_on_save_number=5)
    hamt = await build_wide_hamt(cas)
    cas.arm()

    with pytest.raises(ConnectionError):
        await hamt.cache_vacate()

    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert pending == [], f"orphaned vacate save tasks survived the failure: {pending}"
