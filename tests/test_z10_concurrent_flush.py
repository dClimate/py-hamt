import asyncio
import json

import pytest
import zarr
from multiformats import CID
from testing_utils import CIDInMemoryCAS

from py_hamt import ContentAddressedStore, ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [12, 2],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [2, 2]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0,
        "codecs": [{"name": "bytes"}],
    },
    separators=(",", ":"),
).encode()


class InstrumentedCAS(CIDInMemoryCAS):
    def __init__(self) -> None:
        super().__init__()
        self.in_flight_saves = 0
        self.max_concurrent_saves = 0

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        self.in_flight_saves += 1
        self.max_concurrent_saves = max(self.max_concurrent_saves, self.in_flight_saves)
        try:
            await asyncio.sleep(0.02)
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
        self.save_calls = 0

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        self.save_calls += 1
        if self.save_calls == self.fail_on_save_number:
            await asyncio.sleep(0.005)
            raise ConnectionError("simulated CAS save failure")
        return await super().save(data, codec)


def buf(data: bytes) -> zarr.core.buffer.Buffer:
    return PROTOTYPE.buffer.from_bytes(data)


@pytest.mark.asyncio
async def test_dirty_shards_flush_concurrently() -> None:
    cas = InstrumentedCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
    )
    await store.set("myarr/zarr.json", buf(ARRAY_METADATA))
    for row in range(6):
        await store.set(f"myarr/c/{row}/0", buf(bytes([row])))

    cas.reset_save_concurrency()
    await store.flush()

    assert cas.max_concurrent_saves >= 2, "dirty shards must flush concurrently"


@pytest.mark.asyncio
async def test_failed_flush_leaves_no_orphan_tasks_and_is_retryable() -> None:
    cas = FailingCAS(fail_on_save_number=0)
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
    )
    await store.set("myarr/zarr.json", buf(ARRAY_METADATA))
    for row in range(6):
        await store.set(f"myarr/c/{row}/0", buf(bytes([row])))
    cas.save_calls = 0
    cas.fail_on_save_number = 3

    with pytest.raises(ConnectionError):
        await store.flush()

    # A failed flush must not leave sibling flush tasks running in the
    # background, where they would mutate store state without the write lock.
    pending = [
        task
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task() and not task.done()
    ]
    assert pending == [], f"orphaned flush tasks survived the failure: {pending}"

    cas.fail_on_save_number = 0
    root_cid = await store.flush()

    reopened = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)
    for row in range(6):
        chunk = await reopened.get(f"myarr/c/{row}/0", PROTOTYPE)
        assert chunk is not None
        assert chunk.to_bytes() == bytes([row])
