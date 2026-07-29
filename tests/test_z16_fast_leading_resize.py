import json
from unittest.mock import AsyncMock

import numpy as np
import pytest
import xarray as xr
import zarr.core.buffer
from multiformats import CID
from testing_utils import CIDInMemoryCAS

from py_hamt.sharded_zarr_store import (
    SHARDED_ZARR_V2,
    ArrayIndex,
    ShardedZarrStore,
)


def test_fast_leading_resize_requires_strict_append_only_growth() -> None:
    index = ArrayIndex.new(
        array_path="temperature",
        array_shape=(2, 4, 6),
        chunk_shape=(1, 2, 3),
        chunks_per_shard=4,
    )

    assert ShardedZarrStore._can_fast_resize_leading_dimension(index, (3, 4, 6))
    assert ShardedZarrStore._can_fast_resize_leading_dimension(index, (2, 4, 6))
    assert not ShardedZarrStore._can_fast_resize_leading_dimension(index, (1, 4, 6))
    assert not ShardedZarrStore._can_fast_resize_leading_dimension(index, (3, 5, 6))
    assert not ShardedZarrStore._can_fast_resize_leading_dimension(index, (3, 4))


@pytest.mark.asyncio
async def test_v1_leading_growth_skips_snapshot_and_preserves_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = CIDInMemoryCAS()
    prototype = zarr.core.buffer.default_buffer_prototype()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 2),
        chunk_shape=(1, 1),
        chunks_per_shard=3,
    )
    await store.set("temperature/c/0/0", prototype.buffer.from_bytes(b"old-0-0"))
    await store.set("temperature/c/1/1", prototype.buffer.from_bytes(b"old-1-1"))
    root_cid = await store.flush()

    writable = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=root_cid,
    )
    old_shard_cids = list(writable.array_indices[""].shard_cids)

    snapshot = AsyncMock(
        side_effect=AssertionError("leading-dimension growth must not snapshot shards")
    )
    monkeypatch.setattr(writable, "_snapshot_shards_for_resize", snapshot)

    await writable.resize_store((3, 2))

    snapshot.assert_not_awaited()
    resized_index = writable.array_indices[""]
    assert resized_index.array_shape == (3, 2)
    assert resized_index.shard_cids[: len(old_shard_cids)] == old_shard_cids
    assert writable._root_obj["chunks"]["array_shape"] == [3, 2]
    assert writable._root_obj["chunks"]["shard_cids"] == resized_index.shard_cids

    await writable.set("temperature/c/2/0", prototype.buffer.from_bytes(b"new-2-0"))
    resized_root_cid = await writable.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=resized_root_cid,
    )

    old_chunk = await reopened.get("temperature/c/0/0", prototype)
    appended_chunk = await reopened.get("temperature/c/2/0", prototype)
    assert old_chunk is not None
    assert appended_chunk is not None
    assert old_chunk.to_bytes() == b"old-0-0"
    assert appended_chunk.to_bytes() == b"new-2-0"


@pytest.mark.asyncio
async def test_v2_leading_growth_is_fast_and_array_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = CIDInMemoryCAS()
    prototype = zarr.core.buffer.default_buffer_prototype()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=3,
        manifest_version=SHARDED_ZARR_V2,
    )
    for array_path in ("temperature", "humidity"):
        metadata = {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [2, 2],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [1, 1]},
            },
        }
        await store.set(
            f"{array_path}/zarr.json",
            prototype.buffer.from_bytes(json.dumps(metadata).encode()),
        )
        await store.set(
            f"{array_path}/c/0/0",
            prototype.buffer.from_bytes(array_path.encode()),
        )
    root_cid = await store.flush()

    writable = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=root_cid,
    )
    humidity_manifest = writable.array_indices["humidity"].to_manifest()

    snapshot = AsyncMock(
        side_effect=AssertionError("leading-dimension growth must not snapshot shards")
    )
    monkeypatch.setattr(writable, "_snapshot_shards_for_resize", snapshot)

    await writable.resize_store((3, 2), array_path="temperature")

    snapshot.assert_not_awaited()
    assert writable.array_indices["temperature"].array_shape == (3, 2)
    assert writable.array_indices["humidity"].to_manifest() == humidity_manifest

    await writable.set("temperature/c/2/1", prototype.buffer.from_bytes(b"appended"))
    resized_root_cid = await writable.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=resized_root_cid,
    )

    old_chunk = await reopened.get("temperature/c/0/0", prototype)
    appended_chunk = await reopened.get("temperature/c/2/1", prototype)
    assert old_chunk is not None
    assert appended_chunk is not None
    assert old_chunk.to_bytes() == b"temperature"
    assert appended_chunk.to_bytes() == b"appended"
    assert reopened.array_indices["temperature"].array_shape == (3, 2)
    assert reopened.array_indices["humidity"].array_shape == (2, 2)


@pytest.mark.asyncio
async def test_v1_xarray_append_persists_visible_shape_without_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = CIDInMemoryCAS()
    initial = xr.Dataset(
        {"temperature": (("time", "x"), np.arange(4).reshape(2, 2))},
        coords={"time": np.arange(2), "x": np.arange(2)},
    ).chunk({"time": 1, "x": 1})
    appended = xr.Dataset(
        {"temperature": (("time", "x"), np.arange(4, 8).reshape(2, 2))},
        coords={"time": np.arange(2, 4), "x": np.arange(2)},
    ).chunk({"time": 1, "x": 1})
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 2),
        chunk_shape=(1, 1),
        chunks_per_shard=3,
    )
    initial.to_zarr(store=store, mode="w")
    root_cid = await store.flush()
    writable = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=root_cid,
    )

    snapshot = AsyncMock(
        side_effect=AssertionError("leading-dimension growth must not snapshot shards")
    )
    monkeypatch.setattr(writable, "_snapshot_shards_for_resize", snapshot)

    appended.to_zarr(store=writable, append_dim="time")

    snapshot.assert_not_awaited()
    resized_root_cid = await writable.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=resized_root_cid,
    )
    actual = xr.open_zarr(store=reopened).compute()
    expected = xr.concat([initial, appended], dim="time").compute()

    assert actual.sizes == {"time": 4, "x": 2}
    xr.testing.assert_identical(actual, expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("new_shape", [(2, 3), (1, 2)])
async def test_non_leading_change_and_shrink_use_general_resize(
    new_shape: tuple[int, int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 2),
        chunk_shape=(1, 1),
        chunks_per_shard=3,
    )
    snapshot_calls = 0
    original_snapshot = store._snapshot_shards_for_resize

    async def count_snapshot(
        array_index: ArrayIndex,
    ) -> dict[int, list[CID | None]]:
        nonlocal snapshot_calls
        snapshot_calls += 1
        return await original_snapshot(array_index)

    monkeypatch.setattr(store, "_snapshot_shards_for_resize", count_snapshot)

    await store.resize_store(new_shape)

    assert snapshot_calls == 1
    assert store.array_indices[""].array_shape == new_shape
