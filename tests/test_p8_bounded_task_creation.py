import asyncio
import contextvars
import json
from collections.abc import Coroutine
from typing import Any, TypeVar, cast

import pytest
import zarr
from multiformats import CID
from testing_utils import CIDInMemoryCAS

from py_hamt import HAMT, ContentAddressedStore, InMemoryCAS, ShardedZarrStore
from py_hamt.hamt import _VACATE_CONCURRENCY, InMemoryTreeStore
from py_hamt.sharded_zarr_store import _FLUSH_CONCURRENCY

_T = TypeVar("_T")
_HAMT_KEY_COUNT = 300
_SHARD_COUNT = 40
_SAVE_DELAY_SECONDS = 0.01

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [_SHARD_COUNT * 2, 2],
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


class TaskCreationTracker:
    """Count tasks created during one flush until each task finishes."""

    def __init__(self) -> None:
        self.live_tasks: set[asyncio.Task[Any]] = set()
        self.peak_live_tasks = 0

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        loop = asyncio.get_running_loop()
        original_create_task = loop.create_task

        def tracked_create_task(
            coro: Coroutine[Any, Any, _T],
            *,
            name: str | None = None,
            context: contextvars.Context | None = None,
        ) -> asyncio.Task[_T]:
            task = original_create_task(coro, name=name, context=context)
            untyped_task = cast(asyncio.Task[Any], task)
            self.live_tasks.add(untyped_task)
            self.peak_live_tasks = max(
                self.peak_live_tasks,
                len(self.live_tasks),
            )
            task.add_done_callback(self.live_tasks.discard)
            return task

        monkeypatch.setattr(loop, "create_task", tracked_create_task)

    def pending_tasks(self) -> list[asyncio.Task[Any]]:
        return [task for task in self.live_tasks if not task.done()]


class DelayedBytesCAS(InMemoryCAS):
    def __init__(self, fail_on_save: int | None = None) -> None:
        super().__init__()
        self.fail_on_save = fail_on_save
        self.armed = False
        self.started_saves = 0
        self.finished_saves = 0
        self.in_flight_saves = 0

    def arm(self) -> None:
        self.armed = True
        self.started_saves = 0
        self.finished_saves = 0
        self.in_flight_saves = 0

    async def save(
        self,
        data: bytes,
        codec: ContentAddressedStore.CodecInput,
    ) -> bytes:
        if not self.armed:
            return await super().save(data, codec)

        self.started_saves += 1
        save_number = self.started_saves
        self.in_flight_saves += 1
        try:
            if save_number == self.fail_on_save:
                await asyncio.sleep(0)
                raise ConnectionError("simulated HAMT save failure")
            await asyncio.sleep(_SAVE_DELAY_SECONDS)
            result = await super().save(data, codec)
            self.finished_saves += 1
            return result
        finally:
            self.in_flight_saves -= 1


class DelayedCIDCAS(CIDInMemoryCAS):
    def __init__(self, fail_on_save: int | None = None) -> None:
        super().__init__()
        self.fail_on_save = fail_on_save
        self.armed = False
        self.started_saves = 0
        self.finished_saves = 0
        self.in_flight_saves = 0

    def arm(self) -> None:
        self.armed = True
        self.started_saves = 0
        self.finished_saves = 0
        self.in_flight_saves = 0

    async def save(
        self,
        data: bytes,
        codec: ContentAddressedStore.CodecInput,
    ) -> CID:
        if not self.armed:
            return await super().save(data, codec)

        self.started_saves += 1
        save_number = self.started_saves
        self.in_flight_saves += 1
        try:
            if save_number == self.fail_on_save:
                await asyncio.sleep(0)
                raise ConnectionError("simulated shard save failure")
            await asyncio.sleep(_SAVE_DELAY_SECONDS)
            result = await super().save(data, codec)
            self.finished_saves += 1
            return result
        finally:
            self.in_flight_saves -= 1


async def build_wide_hamt(cas: DelayedBytesCAS) -> HAMT:
    hamt = await HAMT.build(
        cas=cas,
        max_bucket_size=1,
        values_are_bytes=True,
    )
    for index in range(_HAMT_KEY_COUNT):
        await hamt.set(f"key-{index}", b"value")
    return hamt


async def build_dirty_sharded_store(cas: DelayedCIDCAS) -> ShardedZarrStore:
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
    )
    await store.set("array/zarr.json", PROTOTYPE.buffer.from_bytes(ARRAY_METADATA))
    for shard_index in range(_SHARD_COUNT):
        await store.set(
            f"array/c/{shard_index}/0",
            PROTOTYPE.buffer.from_bytes(bytes([shard_index])),
        )
    return store


@pytest.mark.asyncio
async def test_vacate_bounds_simultaneously_live_save_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = DelayedBytesCAS()
    hamt = await build_wide_hamt(cas)
    node_store = cast(InMemoryTreeStore, hamt.node_store)
    buffered_node_count = len(node_store.buffer)
    assert buffered_node_count > _VACATE_CONCURRENCY
    cas.arm()
    tracker = TaskCreationTracker()

    with monkeypatch.context() as scoped_monkeypatch:
        tracker.install(scoped_monkeypatch)
        await hamt.cache_vacate()

    assert tracker.peak_live_tasks <= _VACATE_CONCURRENCY, (
        "vacate must not create the entire save wave as pending tasks: "
        f"peak={tracker.peak_live_tasks}, limit={_VACATE_CONCURRENCY}, "
        f"buffered_nodes={buffered_node_count}"
    )


@pytest.mark.asyncio
async def test_shard_flush_bounds_simultaneously_live_save_tasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = DelayedCIDCAS()
    store = await build_dirty_sharded_store(cas)
    cas.arm()
    tracker = TaskCreationTracker()

    with monkeypatch.context() as scoped_monkeypatch:
        tracker.install(scoped_monkeypatch)
        await store.flush()

    assert tracker.peak_live_tasks <= _FLUSH_CONCURRENCY, (
        "shard flush must not create every dirty shard as a pending task: "
        f"peak={tracker.peak_live_tasks}, limit={_FLUSH_CONCURRENCY}, "
        f"dirty_shards={_SHARD_COUNT}"
    )


@pytest.mark.asyncio
async def test_failed_vacate_cancels_bounded_work_without_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = DelayedBytesCAS(fail_on_save=3)
    hamt = await build_wide_hamt(cas)
    total_nodes = len(cast(InMemoryTreeStore, hamt.node_store).buffer)
    cas.arm()
    tracker = TaskCreationTracker()

    with monkeypatch.context() as scoped_monkeypatch:
        tracker.install(scoped_monkeypatch)
        with pytest.raises(ConnectionError, match="simulated HAMT save failure"):
            await hamt.cache_vacate()

    assert cas.started_saves < total_nodes, "failure must stop unscheduled node saves"
    assert cas.in_flight_saves == 0
    assert tracker.pending_tasks() == [], "vacate save tasks survived the failure"


@pytest.mark.asyncio
async def test_failed_shard_flush_cancels_bounded_work_without_orphans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = DelayedCIDCAS(fail_on_save=3)
    store = await build_dirty_sharded_store(cas)
    cas.arm()
    tracker = TaskCreationTracker()

    with monkeypatch.context() as scoped_monkeypatch:
        tracker.install(scoped_monkeypatch)
        with pytest.raises(ConnectionError, match="simulated shard save failure"):
            await store.flush()

    assert cas.started_saves < _SHARD_COUNT, "failure must stop unscheduled shard saves"
    assert cas.in_flight_saves == 0
    assert tracker.pending_tasks() == [], "shard save tasks survived the failure"
