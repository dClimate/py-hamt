import asyncio
import json
import warnings
from collections.abc import AsyncIterator

import dag_cbor
import numpy as np
import pytest
import xarray as xr
import zarr.core.buffer
from dag_cbor.ipld import IPLDKind
from hypothesis import given, settings
from hypothesis import strategies as st
from multiformats import CID, multihash
from zarr.abc.store import RangeByteRequest

from py_hamt import HAMT
from py_hamt.hamt_to_sharded_converter import (
    _is_zarr_chunk_key,
    _normalize_zarr_chunk_key,
    convert_hamt_to_sharded,
)
from py_hamt.sharded_zarr_store import (
    SHARDED_ZARR_V1,
    SHARDED_ZARR_V2,
    ArrayIndex,
    ShardedZarrStore,
    ShardedZarrV1DeprecationWarning,
)
from py_hamt.store_httpx import ContentAddressedStore
from py_hamt.zarr_hamt_store import ZarrHAMTStore


class LocalCIDCAS(ContentAddressedStore):
    """Small CID-addressed in-memory CAS for ShardedZarrStore tests."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}

    @staticmethod
    def _key(cid: IPLDKind) -> str:
        decoded = CID.decode(cid) if isinstance(cid, str) else cid
        if not isinstance(decoded, CID):
            raise TypeError(f"Expected CID, got {type(cid).__name__}")
        return str(decoded.encode("base32"))

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        digest = multihash.digest(data, "sha2-256")
        cid = CID("base32", 1, codec, digest)
        self.store[self._key(cid)] = data
        return cid

    async def load(
        self,
        cid: IPLDKind,
        offset: int | None = None,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes:
        data = self.store[self._key(cid)]
        if offset is not None:
            if length is None:
                return data[offset:]
            return data[offset : offset + length]
        if suffix is not None:
            return data[-suffix:]
        return data


def _pyramid_level(data: np.ndarray, *, coord_offset: int = 0) -> xr.Dataset:
    return xr.Dataset(
        {"FPAR": (("time", "y", "x"), data)},
        coords={
            "time": coord_offset + np.arange(data.shape[0]),
            "y": coord_offset + np.arange(data.shape[1]),
            "x": coord_offset + np.arange(data.shape[2]),
        },
    )


@pytest.mark.asyncio
async def test_v2_grouped_pyramid_arrays_are_path_aware() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )

    level_0 = _pyramid_level(np.arange(16).reshape(2, 2, 4)).chunk({
        "time": 1,
        "y": 1,
        "x": 2,
    })
    level_1 = _pyramid_level(
        np.arange(8).reshape(2, 2, 2) + 100, coord_offset=1_000
    ).chunk({"time": 1, "y": 2, "x": 1})
    level_2 = _pyramid_level(
        np.arange(4).reshape(2, 1, 2) + 200, coord_offset=2_000
    ).chunk({"time": 1, "y": 1, "x": 1})

    level_0.to_zarr(store=store, group="0", mode="w", zarr_format=3)
    level_1.to_zarr(store=store, group="1", mode="a", zarr_format=3)
    level_2.to_zarr(store=store, group="2", mode="a", zarr_format=3)

    assert store._root_obj["manifest_version"] == SHARDED_ZARR_V2
    assert store.array_indices["0/FPAR"].array_shape == (2, 2, 4)
    assert store.array_indices["1/FPAR"].chunk_shape == (1, 2, 1)
    assert store.array_indices["2/FPAR"].array_shape == (2, 1, 2)

    assert await store.exists("0/FPAR/c/0/0/0")
    assert await store.exists("1/FPAR/c/0/0/0")
    assert await store.exists("2/FPAR/c/0/0/0")
    expected_levels = {"0": level_0, "1": level_1, "2": level_2}
    for group, level in expected_levels.items():
        for coord_name in ("x", "y", "time"):
            coord_path = f"{group}/{coord_name}"
            assert store.array_indices[coord_path].array_shape == (
                level.sizes[coord_name],
            )
            assert await store.exists(f"{coord_path}/c/0")
    assert not await store.exists("x/c/0")
    assert not await store.exists("y/c/0")
    assert not await store.exists("time/c/0")

    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)

    xr.testing.assert_identical(
        level_0, xr.open_zarr(store=read_store, group="0").compute()
    )
    xr.testing.assert_identical(
        level_1, xr.open_zarr(store=read_store, group="1").compute()
    )
    xr.testing.assert_identical(
        level_2, xr.open_zarr(store=read_store, group="2").compute()
    )
    with pytest.raises(ValueError, match="group='0'"):
        xr.open_zarr(store=read_store)

    with pytest.raises(ValueError, match="group='0'"):
        _ = {entry async for entry in read_store.list_dir("")}
    level_entries = {entry async for entry in read_store.list_dir("0")}
    assert {"FPAR", "x", "y", "time", "zarr.json"}.issubset(level_entries)
    array_entries = {entry async for entry in read_store.list_dir("0/FPAR")}
    assert {"zarr.json", "c"}.issubset(array_entries)

    prefix_keys = {key async for key in read_store.list_prefix("0/FPAR/")}
    assert "0/FPAR/zarr.json" in prefix_keys
    assert "0/FPAR/c/0/0/0" in prefix_keys

    proto = zarr.core.buffer.default_buffer_prototype()
    full_chunk = await read_store.get("0/FPAR/c/0/0/0", proto)
    assert full_chunk is not None
    partial_chunk = await read_store.get(
        "0/FPAR/c/0/0/0", proto, RangeByteRequest(start=0, end=5)
    )
    assert partial_chunk is not None
    assert partial_chunk.to_bytes() == full_chunk.to_bytes()[:5]

    isolation_store = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=root_cid
    )
    for coord_name in ("x", "y", "time"):
        await isolation_store.delete(f"0/{coord_name}/c/0")
    xr.testing.assert_identical(
        level_1, xr.open_zarr(store=isolation_store, group="1").compute()
    )

    write_store = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=root_cid
    )
    await write_store.delete("1/FPAR/c/0/0/0")
    assert not await write_store.exists("1/FPAR/c/0/0/0")
    assert await write_store.exists("0/FPAR/c/0/0/0")


@pytest.mark.asyncio
async def test_v2_write_requires_explicit_group() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    ds = _pyramid_level(np.arange(4).reshape(1, 2, 2)).chunk({
        "time": 1,
        "y": 1,
        "x": 1,
    })

    with pytest.raises(ValueError, match="group='0'"):
        ds.to_zarr(store=store, mode="w", zarr_format=3)


@pytest.mark.asyncio
async def test_v2_single_group_root_read_defaults_to_only_group() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    level_0 = _pyramid_level(np.arange(4).reshape(1, 2, 2)).chunk({
        "time": 1,
        "y": 1,
        "x": 1,
    })

    level_0.to_zarr(store=store, group="0", mode="w", zarr_format=3)
    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)

    xr.testing.assert_identical(level_0, xr.open_zarr(store=read_store).compute())
    root_entries = {entry async for entry in read_store.list_dir("")}
    assert {"FPAR", "x", "y", "time", "zarr.json"}.issubset(root_entries)
    array_entries = {entry async for entry in read_store.list_dir("FPAR")}
    assert {"zarr.json", "c"}.issubset(array_entries)


@pytest.mark.asyncio
async def test_v2_resize_is_array_local() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    metadata_a = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2, 2],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1]}},
    }
    metadata_b = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}},
    }

    await store.set(
        "0/FPAR/zarr.json",
        proto.buffer.from_bytes(json.dumps(metadata_a).encode()),
    )
    await store.set(
        "1/FPAR/zarr.json",
        proto.buffer.from_bytes(json.dumps(metadata_b).encode()),
    )
    await store.set("0/FPAR/c/0/0", proto.buffer.from_bytes(b"array-a"))
    await store.set("1/FPAR/c/0", proto.buffer.from_bytes(b"array-b"))

    await store.resize_store((3, 2), array_path="0/FPAR")

    assert store.array_indices["0/FPAR"].array_shape == (3, 2)
    assert store.array_indices["1/FPAR"].array_shape == (4,)
    assert await store.exists("0/FPAR/c/0/0")
    assert await store.exists("1/FPAR/c/0")

    await store.resize_variable("0/FPAR", (4, 2))
    assert store.array_indices["0/FPAR"].array_shape == (4, 2)


@pytest.mark.asyncio
async def test_resize_clears_retained_and_dropped_shard_entries() -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [6],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1]}},
    }
    v2_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=4,
        manifest_version=SHARDED_ZARR_V2,
    )
    await v2_store.set(
        "a/zarr.json", proto.buffer.from_bytes(json.dumps(metadata).encode())
    )
    for idx in range(6):
        await v2_store.set(f"a/c/{idx}", proto.buffer.from_bytes(f"v2-{idx}".encode()))

    await v2_store.resize_store((2,), array_path="a")
    await v2_store.resize_store((6,), array_path="a")

    assert await v2_store.exists("a/c/0")
    assert await v2_store.exists("a/c/1")
    assert not await v2_store.exists("a/c/2")
    assert not await v2_store.exists("a/c/5")

    v1_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(6,),
        chunk_shape=(1,),
        chunks_per_shard=4,
    )
    for idx in range(6):
        await v1_store.set(f"a/c/{idx}", proto.buffer.from_bytes(f"v1-{idx}".encode()))

    await v1_store.resize_store((2,))
    await v1_store.resize_store((6,))

    assert await v1_store.exists("a/c/0")
    assert await v1_store.exists("a/c/1")
    assert not await v1_store.exists("a/c/2")
    assert not await v1_store.exists("a/c/5")


@pytest.mark.asyncio
async def test_resize_preserves_multidimensional_chunk_coordinates() -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()
    metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2, 3],
        "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [1, 1]}},
    }
    v2_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    await v2_store.set(
        "a/zarr.json", proto.buffer.from_bytes(json.dumps(metadata).encode())
    )
    for y_idx in range(2):
        for x_idx in range(3):
            await v2_store.set(
                f"a/c/{y_idx}/{x_idx}",
                proto.buffer.from_bytes(f"v2-{y_idx}-{x_idx}".encode()),
            )

    await v2_store.resize_store((3, 2), array_path="a")

    retained_v2 = await v2_store.get("a/c/1/0", proto)
    assert retained_v2 is not None
    assert retained_v2.to_bytes() == b"v2-1-0"
    assert not await v2_store.exists("a/c/0/2")
    assert not await v2_store.exists("a/c/1/2")
    assert not await v2_store.exists("a/c/2/0")

    v1_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 3),
        chunk_shape=(1, 1),
        chunks_per_shard=2,
    )
    for y_idx in range(2):
        for x_idx in range(3):
            await v1_store.set(
                f"a/c/{y_idx}/{x_idx}",
                proto.buffer.from_bytes(f"v1-{y_idx}-{x_idx}".encode()),
            )

    await v1_store.resize_store((3, 2))

    retained_v1 = await v1_store.get("a/c/1/0", proto)
    assert retained_v1 is not None
    assert retained_v1.to_bytes() == b"v1-1-0"
    assert not await v1_store.exists("a/c/0/2")
    assert not await v1_store.exists("a/c/1/2")
    assert not await v1_store.exists("a/c/2/0")


@pytest.mark.asyncio
async def test_v2_resize_blocks_interleaved_chunk_mutators(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()

    async def paused_resize_store() -> tuple[
        ShardedZarrStore,
        asyncio.Task[None],
        asyncio.Event,
    ]:
        store = await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            chunks_per_shard=2,
            manifest_version=SHARDED_ZARR_V2,
        )
        await store.set(
            "a/zarr.json",
            proto.buffer.from_bytes(
                json.dumps({
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2, 3],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1, 1]},
                    },
                }).encode()
            ),
        )
        await store.set("a/c/1/0", proto.buffer.from_bytes(b"original"))

        snapshot_done = asyncio.Event()
        finish_resize = asyncio.Event()
        original_snapshot = store._snapshot_shards_for_resize

        async def gated_snapshot(
            array_index: ArrayIndex,
        ) -> dict[int, list[CID | None]]:
            snapshot = await original_snapshot(array_index)
            snapshot_done.set()
            await finish_resize.wait()
            return snapshot

        monkeypatch.setattr(store, "_snapshot_shards_for_resize", gated_snapshot)
        resize_task = asyncio.create_task(store.resize_store((3, 2), array_path="a"))
        await snapshot_done.wait()
        return store, resize_task, finish_resize

    store, resize_task, finish_resize = await paused_resize_store()
    write_task = asyncio.create_task(
        store.set("a/c/1/0", proto.buffer.from_bytes(b"interleaved"))
    )
    await asyncio.sleep(0)

    assert not write_task.done()
    finish_resize.set()
    await resize_task
    await write_task

    chunk = await store.get("a/c/1/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"interleaved"

    pointer_cid = await cas.save(b"pointer", codec="raw")
    store, resize_task, finish_resize = await paused_resize_store()
    pointer_task = asyncio.create_task(store.set_pointer("a/c/1/0", str(pointer_cid)))
    await asyncio.sleep(0)

    assert not pointer_task.done()
    finish_resize.set()
    await resize_task
    await pointer_task

    pointer_chunk = await store.get("a/c/1/0", proto)
    assert pointer_chunk is not None
    assert pointer_chunk.to_bytes() == b"pointer"

    store, resize_task, finish_resize = await paused_resize_store()
    delete_task = asyncio.create_task(store.delete("a/c/1/0"))
    await asyncio.sleep(0)

    assert not delete_task.done()
    finish_resize.set()
    await resize_task
    await delete_task

    assert not await store.exists("a/c/1/0")


@pytest.mark.asyncio
async def test_v2_resize_waits_for_in_flight_chunk_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2, 3],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1]},
                },
            }).encode()
        ),
    )
    await store.set("a/c/1/0", proto.buffer.from_bytes(b"original"))

    save_started = asyncio.Event()
    finish_save = asyncio.Event()
    original_save = cas.save

    async def gated_save(data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        if data == b"late-set":
            save_started.set()
            await finish_save.wait()
        return await original_save(data, codec)

    monkeypatch.setattr(cas, "save", gated_save)

    set_task = asyncio.create_task(
        store.set("a/c/1/0", proto.buffer.from_bytes(b"late-set"))
    )
    await save_started.wait()
    resize_task = asyncio.create_task(store.resize_store((3, 2), array_path="a"))
    await asyncio.sleep(0)

    assert not resize_task.done()
    finish_save.set()
    await set_task
    await resize_task

    chunk = await store.get("a/c/1/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"late-set"


@pytest.mark.asyncio
async def test_v2_flush_waits_for_in_flight_resize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2, 3],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1]},
                },
            }).encode()
        ),
    )
    for y_idx in range(2):
        for x_idx in range(3):
            await store.set(
                f"a/c/{y_idx}/{x_idx}",
                proto.buffer.from_bytes(f"{y_idx},{x_idx}".encode()),
            )

    replace_started = asyncio.Event()
    finish_replace = asyncio.Event()
    original_replace = store._replace_shards_after_resize

    async def gated_replace(
        array_index: ArrayIndex,
        old_num_shards: int,
        old_shard_cids: list[CID | None],
        old_shards_by_index: dict[int, list[CID | None]],
        new_shards_by_index: dict[int, list[CID | None]],
    ) -> None:
        replace_started.set()
        await finish_replace.wait()
        await original_replace(
            array_index,
            old_num_shards,
            old_shard_cids,
            old_shards_by_index,
            new_shards_by_index,
        )

    monkeypatch.setattr(store, "_replace_shards_after_resize", gated_replace)

    resize_task = asyncio.create_task(store.resize_store((3, 2), array_path="a"))
    await replace_started.wait()
    flush_task = asyncio.create_task(store.flush())
    await asyncio.sleep(0)

    assert not flush_task.done()
    finish_replace.set()
    await resize_task
    root_cid = await flush_task

    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)
    chunk = await read_store.get("a/c/1/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"1,0"


@pytest.mark.asyncio
async def test_set_pointer_read_only_and_metadata_cache_invalidation() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    first_cid = await cas.save(b"first", codec="raw")
    second_cid = await cas.save(b"second", codec="raw")

    await store.set_pointer("attrs", str(first_cid))
    first = await store.get("attrs", proto)
    assert first is not None
    assert first.to_bytes() == b"first"

    await store.set_pointer("attrs", str(second_cid))
    second = await store.get("attrs", proto)
    assert second is not None
    assert second.to_bytes() == b"second"

    with pytest.raises(PermissionError, match="read-only"):
        await store.with_read_only(True).set_pointer("attrs", str(first_cid))


@pytest.mark.asyncio
async def test_set_pointer_metadata_registration_is_atomic() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    old_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1]},
        },
    }
    new_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [2]},
        },
    }

    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(json.dumps(old_metadata).encode()),
    )
    old_pointer = store._root_obj["metadata"]["a/zarr.json"]
    new_pointer = await cas.save(json.dumps(new_metadata).encode(), codec="raw")

    with pytest.raises(ValueError, match="Cannot change chunk_shape"):
        await store.set_pointer("a/zarr.json", str(new_pointer))

    assert store._root_obj["metadata"]["a/zarr.json"] == old_pointer
    assert store.array_indices["a"].chunk_shape == (1,)
    current_metadata = await store.get("a/zarr.json", proto)
    assert current_metadata is not None
    assert json.loads(current_metadata.to_bytes()) == old_metadata


@pytest.mark.asyncio
async def test_v2_rejects_chunk_shape_change_for_existing_index() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1]},
                },
            }).encode()
        ),
    )

    updated_index = store._register_or_update_array_index(
        array_path="a", array_shape=(3,), chunk_shape=(1,)
    )
    assert updated_index.array_shape == (3,)

    with pytest.raises(ValueError, match="Cannot change chunk_shape"):
        await store.set(
            "a/zarr.json",
            proto.buffer.from_bytes(
                json.dumps({
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [2]},
                    },
                }).encode()
            ),
        )
    with pytest.raises(ValueError, match="Cannot change chunk_shape"):
        store._register_or_update_array_index(
            array_path="a", array_shape=(2,), chunk_shape=(2,)
        )


@pytest.mark.asyncio
async def test_v2_delete_dir_prunes_array_indices_and_allows_overwrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    first_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1]},
        },
    }
    second_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [2]},
        },
    }

    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(json.dumps(first_metadata).encode()),
    )
    await store.set("a/c/0", proto.buffer.from_bytes(b"old"))
    await store.set(
        "b/zarr.json",
        proto.buffer.from_bytes(json.dumps(first_metadata).encode()),
    )
    await store.set("b/c/0", proto.buffer.from_bytes(b"kept"))
    pending_load = asyncio.Event()
    store._pending_shard_loads[store._cache_key("a", 0)] = pending_load
    assert "a" in {entry async for entry in store.list_dir("")}

    async def fail_list_prefix(prefix: str) -> AsyncIterator[str]:
        raise AssertionError(f"delete_dir should not scan list_prefix({prefix!r})")
        if False:
            yield prefix

    monkeypatch.setattr(store, "list_prefix", fail_list_prefix)

    await store.delete_dir("a")

    assert pending_load.is_set()
    assert "a" not in store.array_indices
    assert store._primary_array_path == "b"
    assert "a" not in store._root_obj["arrays"]
    assert "a" not in {entry async for entry in store.list_dir("")}
    assert not await store.exists("a/c/0")
    assert await store.exists("b/c/0")

    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(json.dumps(second_metadata).encode()),
    )
    await store.set("a/c/0", proto.buffer.from_bytes(b"new"))

    assert store.array_indices["a"].array_shape == (4,)
    assert store.array_indices["a"].chunk_shape == (2,)
    chunk = await store.get("a/c/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"new"


@pytest.mark.asyncio
async def test_v2_delete_dir_empty_and_clear_remove_arrays() -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()

    async def populated_store() -> ShardedZarrStore:
        store = await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            chunks_per_shard=1,
            manifest_version=SHARDED_ZARR_V2,
        )
        for array_path in ("a", "b"):
            await store.set(
                f"{array_path}/zarr.json",
                proto.buffer.from_bytes(
                    json.dumps({
                        "zarr_format": 3,
                        "node_type": "array",
                        "shape": [1],
                        "chunk_grid": {
                            "name": "regular",
                            "configuration": {"chunk_shape": [1]},
                        },
                    }).encode()
                ),
            )
            await store.set(
                f"{array_path}/c/0", proto.buffer.from_bytes(array_path.encode())
            )
        return store

    store = await populated_store()
    read_only_store = store.with_read_only(True)
    with pytest.raises(PermissionError, match="read-only"):
        await read_only_store.delete_dir("a")
    with pytest.raises(PermissionError, match="read-only"):
        await read_only_store.clear()

    pending_load = asyncio.Event()
    store._pending_shard_loads[store._cache_key("a", 0)] = pending_load
    await store.delete_dir("")
    assert pending_load.is_set()
    assert store.array_indices == {}
    assert store._root_obj["metadata"] == {}
    assert store._root_obj["arrays"] == {}
    assert [key async for key in store.list()] == []

    store = await populated_store()
    await store.clear()
    assert store.array_indices == {}
    assert store._root_obj["metadata"] == {}
    assert store._root_obj["arrays"] == {}
    assert [key async for key in store.list()] == []


@pytest.mark.asyncio
async def test_v2_to_zarr_mode_w_can_replace_group_geometry() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    first = xr.Dataset({"value": ("x", np.arange(2))}).chunk({"x": 1})
    second = xr.Dataset({"value": ("x", np.arange(4) + 10)}).chunk({"x": 2})

    first.to_zarr(store=store, group="a", mode="w", zarr_format=3)
    assert store.array_indices["a/value"].chunk_shape == (1,)

    second.to_zarr(store=store, group="a", mode="w", zarr_format=3)

    assert store.array_indices["a/value"].array_shape == (4,)
    assert store.array_indices["a/value"].chunk_shape == (2,)
    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)
    xr.testing.assert_identical(
        second, xr.open_zarr(store=read_store, group="a").compute()
    )


@pytest.mark.asyncio
async def test_v1_migrate_to_v2_reuses_shards() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 2),
        chunk_shape=(1, 1),
        chunks_per_shard=2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await store.set("FPAR/c/0/0", proto.buffer.from_bytes(b"legacy"))

    migrated_cid = await store.migrate_v1_to_v2("0/FPAR")
    migrated_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=migrated_cid
    )

    assert migrated_store._root_obj["manifest_version"] == SHARDED_ZARR_V2
    assert "0/FPAR" in migrated_store.array_indices
    migrated_chunk = await migrated_store.get("0/FPAR/c/0/0", proto)
    assert migrated_chunk is not None
    assert migrated_chunk.to_bytes() == b"legacy"
    default_group_chunk = await migrated_store.get("FPAR/c/0/0", proto)
    assert default_group_chunk is not None
    assert default_group_chunk.to_bytes() == b"legacy"


@pytest.mark.asyncio
async def test_v1_migrate_to_v2_rewrites_metadata_for_group_open() -> None:
    cas = LocalCIDCAS()
    data = np.arange(4).reshape(1, 2, 2)
    source = xr.Dataset(
        {"FPAR": (("time", "lat", "lon"), data)},
        coords={
            "time": np.arange(1),
            "lat": np.arange(2),
            "lon": np.arange(2),
        },
    ).chunk({"time": 1, "lat": 1, "lon": 1})
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1, 2, 2),
        chunk_shape=(1, 1, 1),
        chunks_per_shard=2,
    )

    source.to_zarr(store=store, mode="w", zarr_format=3)
    migrated_cid = await store.migrate_v1_to_v2("0/FPAR")
    migrated_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=migrated_cid
    )

    assert "0/zarr.json" in migrated_store._root_obj["metadata"]
    assert "0/FPAR/zarr.json" in migrated_store._root_obj["metadata"]
    assert "FPAR/zarr.json" not in migrated_store._root_obj["metadata"]
    xr.testing.assert_identical(
        source, xr.open_zarr(store=migrated_store, group="0").compute()
    )
    assert (
        ShardedZarrStore._rewrite_v1_metadata_key_for_migration(
            "zarr.json", "", "0/FPAR"
        )
        == "0/FPAR/zarr.json"
    )

    v2_array_metadata_cid = await cas.save(
        json.dumps({
            "zarr_format": 2,
            "shape": [1],
            "chunks": [1],
            "dtype": "|u1",
            "compressor": None,
            "fill_value": 0,
            "order": "C",
            "filters": None,
        }).encode(),
        codec="raw",
    )
    metadata = {"0/root/.zarray": v2_array_metadata_cid}
    await migrated_store._add_missing_group_metadata(metadata, "0/root")
    assert ".zgroup" in metadata
    assert "0/.zgroup" in metadata


@pytest.mark.asyncio
async def test_v1_migrate_to_v2_preserves_existing_group_metadata() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1,),
        chunk_shape=(1,),
        chunks_per_shard=1,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    root_group_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"root": "keep"},
    }
    level_group_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"level": "keep"},
    }
    array_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [1],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1]},
        },
    }

    await store.set(
        "zarr.json",
        proto.buffer.from_bytes(json.dumps(root_group_metadata).encode()),
    )
    await store.set(
        "0/zarr.json",
        proto.buffer.from_bytes(json.dumps(level_group_metadata).encode()),
    )
    await store.set(
        "FPAR/zarr.json",
        proto.buffer.from_bytes(json.dumps(array_metadata).encode()),
    )
    await store.set("FPAR/c/0", proto.buffer.from_bytes(b"chunk"))

    migrated_cid = await store.migrate_v1_to_v2("0/FPAR")
    migrated_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=migrated_cid
    )
    root_metadata = await cas.load(migrated_store._root_obj["metadata"]["zarr.json"])
    level_metadata = await migrated_store.get("0/zarr.json", proto)

    assert json.loads(root_metadata) == root_group_metadata
    assert level_metadata is not None
    assert json.loads(level_metadata.to_bytes()) == level_group_metadata


@pytest.mark.asyncio
async def test_read_only_store_rejects_v1_migration() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1,),
        chunk_shape=(1,),
        chunks_per_shard=1,
    )
    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)

    with pytest.raises(PermissionError, match="read-only"):
        await read_store.migrate_v1_to_v2("0/root")


@pytest.mark.asyncio
async def test_v1_root_chunk_keys_remain_metadata_compatible() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1,),
        chunk_shape=(1,),
        chunks_per_shard=1,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await store.set("c/0", proto.buffer.from_bytes(b"root-array-chunk"))
    root_cid = await store.flush()

    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)
    assert await read_store.exists("c/0")
    chunk = await read_store.get("c/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"root-array-chunk"


@pytest.mark.asyncio
async def test_v1_root_metadata_chunks_migrate_to_primary_path() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1,),
        chunk_shape=(1,),
        chunks_per_shard=1,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await store.set(
        "zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [1],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1]},
                },
            }).encode()
        ),
    )
    await store.set("c/0", proto.buffer.from_bytes(b"root-array-chunk"))

    migrated_cid = await store.migrate_v1_to_v2("0/root")
    migrated_store = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=migrated_cid
    )

    assert await migrated_store.exists("0/root/c/0")
    chunk = await migrated_store.get("0/root/c/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"root-array-chunk"
    await migrated_store.delete("0/root/c/0")
    assert not await migrated_store.exists("0/root/c/0")


@pytest.mark.asyncio
async def test_v1_c_named_group_chunks_use_shard_index() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1, 1, 1),
        chunk_shape=(1, 1, 1),
        chunks_per_shard=1,
    )
    proto = zarr.core.buffer.default_buffer_prototype()

    await store.set("c/FPAR/c/0/0/0", proto.buffer.from_bytes(b"v1-c-group"))
    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)

    chunk = await read_store.get("c/FPAR/c/0/0/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"v1-c-group"


@pytest.mark.asyncio
async def test_migrated_v1_coordinate_chunks_remain_readable() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2,),
        chunk_shape=(1,),
        chunks_per_shard=1,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await store.set(
        "lat/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [2]},
                },
            }).encode()
        ),
    )
    await store.set("lat/c/0", proto.buffer.from_bytes(b"coordinate-chunk"))

    migrated_cid = await store.migrate_v1_to_v2("0/FPAR")
    migrated_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=migrated_cid
    )

    assert await migrated_store.exists("0/lat/c/0")
    chunk = await migrated_store.get("0/lat/c/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"coordinate-chunk"
    partial = await migrated_store.get(
        "0/lat/c/0", proto, RangeByteRequest(start=0, end=10)
    )
    assert partial is not None
    assert partial.to_bytes() == b"coordinate"

    migrated_write_store = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=migrated_cid
    )
    await migrated_write_store.delete("0/lat/c/0")
    assert not await migrated_write_store.exists("0/lat/c/0")


@pytest.mark.asyncio
async def test_empty_v2_root_reopen_retains_default_sharding_config() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=3,
        manifest_version=SHARDED_ZARR_V2,
    )
    root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(cas=cas, read_only=False, root_cid=root_cid)
    proto = zarr.core.buffer.default_buffer_prototype()

    await reopened.set(
        "a/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1]},
                },
            }).encode()
        ),
    )
    await reopened.set("a/c/0", proto.buffer.from_bytes(b"chunk"))

    assert reopened.array_indices["a"].chunks_per_shard == 3
    assert await reopened.exists("a/c/0")


@pytest.mark.asyncio
async def test_v2_c_named_arrays_groups_and_metadata_suffixes() -> None:
    cas = LocalCIDCAS()
    proto = zarr.core.buffer.default_buffer_prototype()
    array_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    await array_store.set(
        "c/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [1],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1]},
                },
            }).encode()
        ),
    )
    await array_store.set("c/c/0", proto.buffer.from_bytes(b"top-level-c"))
    top_level_chunk = await array_store.get("c/c/0", proto)
    assert top_level_chunk is not None
    assert top_level_chunk.to_bytes() == b"top-level-c"

    group_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    await group_store.set("c/.zattrs", proto.buffer.from_bytes(b"{}"))
    attrs = await group_store.get("c/.zattrs", proto)
    assert attrs is not None
    assert attrs.to_bytes() == b"{}"
    await group_store.set(
        "c/FPAR/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [1, 1, 1],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1, 1]},
                },
            }).encode()
        ),
    )
    await group_store.set("c/FPAR/c/0/0/0", proto.buffer.from_bytes(b"group-c"))
    group_chunk = await group_store.get("c/FPAR/c/0/0/0", proto)
    assert group_chunk is not None
    assert group_chunk.to_bytes() == b"group-c"


@pytest.mark.asyncio
async def test_v2_list_dir_can_walk_explicit_chunk_prefixes() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await store.set(
        "a/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [2, 1],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1]},
                },
            }).encode()
        ),
    )
    await store.set("a/c/1/0", proto.buffer.from_bytes(b"chunk"))
    await store.set(
        "b/zarr.json",
        proto.buffer.from_bytes(
            json.dumps({
                "zarr_format": 3,
                "node_type": "array",
                "shape": [1, 1],
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 1]},
                },
            }).encode()
        ),
    )
    await store.set("b/c/0/0", proto.buffer.from_bytes(b"other-chunk"))

    assert {entry async for entry in store.list_dir("a/c")} == {"1"}
    assert {entry async for entry in store.list_dir("a/c/1")} == {"0"}
    assert {entry async for entry in store.list_dir("a/c/1/0")} == set()


@pytest.mark.asyncio
async def test_v2_root_array_chunks_and_defensive_validation_paths() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
        array_shape=(1,),
        chunk_shape=(1,),
    )
    proto = zarr.core.buffer.default_buffer_prototype()

    await store.set("c/0", proto.buffer.from_bytes(b"root-v2"))
    assert await store.exists("c/0")
    chunk = await store.get("c/0", proto)
    assert chunk is not None
    assert chunk.to_bytes() == b"root-v2"

    with pytest.raises(ValueError, match="Shard index 2 out of bounds"):
        await store._load_or_initialize_shard_cache(2, "")

    root_obj = {
        "manifest_version": SHARDED_ZARR_V2,
        "store_type": "py_hamt.sharded_zarr",
        "zarr_format": 3,
        "sharding_config": "invalid",
        "metadata": {},
        "arrays": {},
    }
    root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    opened = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=str(root_cid.encode("base32"))
    )
    assert opened._default_chunks_per_shard is None

    empty_index_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
        array_shape=(1,),
        chunk_shape=(1,),
    )
    assert {key async for key in empty_index_store.list()} == set()


@pytest.mark.asyncio
async def test_failed_shard_validation_clears_pending_load() -> None:
    cas = LocalCIDCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
        array_shape=(1,),
        chunk_shape=(1,),
    )
    bad_shard_cid = await cas.save(dag_cbor.encode([123]), codec="dag-cbor")
    store.array_indices[""].shard_cids[0] = bad_shard_cid

    with pytest.raises(TypeError, match="non-CID"):
        await store._load_or_initialize_shard_cache(0, "")
    assert ("", 0) not in store._pending_shard_loads


@pytest.mark.asyncio
async def test_graft_rejects_offsets_that_alias_invalid_coordinates() -> None:
    cas = LocalCIDCAS()
    source = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1, 1),
        chunk_shape=(1, 1),
        chunks_per_shard=4,
    )
    target = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 2),
        chunk_shape=(1, 1),
        chunks_per_shard=4,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await source.set("temp/c/0/0", proto.buffer.from_bytes(b"source"))
    source_cid = await source.flush()

    with pytest.raises(ValueError, match="chunk_offset"):
        await target.graft_store(source_cid, chunk_offset=(0,))
    with pytest.raises(ValueError, match="out of bounds"):
        await target.graft_store(source_cid, chunk_offset=(0, 2))
    assert not await target.exists("temp/c/1/0")


def test_converter_chunk_key_classifier_keeps_c_named_metadata_first() -> None:
    assert not _is_zarr_chunk_key("c/zarr.json")
    assert not _is_zarr_chunk_key("0/c/zarr.json")
    assert _is_zarr_chunk_key("0/c/c/0")
    assert _is_zarr_chunk_key("0.0")
    assert _is_zarr_chunk_key("0/FPAR/0.0")
    assert not _is_zarr_chunk_key("plain-metadata")
    assert not _is_zarr_chunk_key("not.a.chunk")
    assert (
        _normalize_zarr_chunk_key("a/0", {"a": ArrayIndex.new("a", (2,), (1,), 1)})
        == "a/c/0"
    )
    assert (
        _normalize_zarr_chunk_key(
            "c/FPAR/0", {"c/FPAR": ArrayIndex.new("c/FPAR", (2,), (1,), 1)}
        )
        == "c/FPAR/c/0"
    )
    assert (
        _normalize_zarr_chunk_key(
            "c/FPAR/c/0", {"c/FPAR": ArrayIndex.new("c/FPAR", (2,), (1,), 1)}
        )
        == "c/FPAR/c/0"
    )
    assert (
        _normalize_zarr_chunk_key("0", {"": ArrayIndex.new("", (2,), (1,), 1)}) == "c/0"
    )
    assert (
        _normalize_zarr_chunk_key("b/0", {"a": ArrayIndex.new("a", (2,), (1,), 1)})
        is None
    )
    assert (
        _normalize_zarr_chunk_key("a/0/0", {"a": ArrayIndex.new("a", (2,), (1,), 1)})
        is None
    )


@pytest.mark.asyncio
async def test_converter_discovers_grouped_arrays() -> None:
    cas = LocalCIDCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    source_store = ZarrHAMTStore(hamt)

    level_0 = _pyramid_level(np.arange(4).reshape(1, 2, 2)).chunk({
        "time": 1,
        "y": 1,
        "x": 1,
    })
    level_1 = _pyramid_level(np.arange(2).reshape(1, 1, 2) + 10).chunk({
        "time": 1,
        "y": 1,
        "x": 1,
    })
    level_0.to_zarr(
        store=source_store,
        group="0",
        mode="w",
        zarr_format=3,
        consolidated=False,
    )
    level_1.to_zarr(
        store=source_store,
        group="1",
        mode="a",
        zarr_format=3,
        consolidated=False,
    )
    await hamt.make_read_only()

    sharded_root = await convert_hamt_to_sharded(cas, str(hamt.root_node_id), 2)
    converted_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=sharded_root
    )

    assert converted_store._root_obj["manifest_version"] == SHARDED_ZARR_V2
    assert {"0/FPAR", "1/FPAR"}.issubset(converted_store.array_indices)
    xr.testing.assert_identical(
        level_0,
        xr.open_zarr(store=converted_store, group="0", consolidated=False).compute(),
    )
    xr.testing.assert_identical(
        level_1,
        xr.open_zarr(store=converted_store, group="1", consolidated=False).compute(),
    )


@pytest.mark.asyncio
async def test_converter_translates_classic_zarr_v2_chunks() -> None:
    cas = LocalCIDCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    root_metadata = {
        "zarr_format": 2,
        "shape": [2, 2],
        "chunks": [1, 1],
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }
    variable_metadata = {
        "zarr_format": 2,
        "shape": [2],
        "chunks": [1],
        "dtype": "|u1",
        "compressor": None,
        "fill_value": 0,
        "order": "C",
        "filters": None,
    }

    await hamt.set(".zarray", json.dumps(root_metadata).encode())
    await hamt.set("0.0", b"root-chunk")
    await hamt.set("var/.zarray", json.dumps(variable_metadata).encode())
    await hamt.set("var/0", b"var-zero")
    await hamt.make_read_only()

    sharded_root = await convert_hamt_to_sharded(cas, str(hamt.root_node_id), 2)
    converted_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=sharded_root
    )
    proto = zarr.core.buffer.default_buffer_prototype()

    root_chunk = await converted_store.get("c/0/0", proto)
    assert root_chunk is not None
    assert root_chunk.to_bytes() == b"root-chunk"
    assert await converted_store.exists("0.0")
    variable_chunk = await converted_store.get("var/c/0", proto)
    assert variable_chunk is not None
    assert variable_chunk.to_bytes() == b"var-zero"
    assert await converted_store.exists("var/0")
    assert "0.0" not in converted_store._root_obj["metadata"]
    assert "var/0" not in converted_store._root_obj["metadata"]


@pytest.mark.asyncio
async def test_converter_round_trips_zarr_v2_hamt_source() -> None:
    cas = LocalCIDCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    source_store = ZarrHAMTStore(hamt)
    source = xr.Dataset({
        "var": (("x", "y"), np.arange(4, dtype=np.uint8).reshape(2, 2))
    }).chunk({"x": 1, "y": 1})

    source.to_zarr(
        store=source_store,
        mode="w",
        zarr_format=2,
        consolidated=False,
    )
    await hamt.make_read_only()

    sharded_root = await convert_hamt_to_sharded(cas, str(hamt.root_node_id), 2)
    converted_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=sharded_root
    )

    assert "zarr.json" not in converted_store._root_obj["metadata"]
    assert ".zgroup" in converted_store._root_obj["metadata"]
    assert await converted_store.exists("var/0.1")
    xr.testing.assert_identical(
        source,
        xr.open_zarr(store=converted_store, consolidated=False).compute(),
    )


@pytest.mark.asyncio
async def test_converter_rejects_unclassified_keys() -> None:
    cas = LocalCIDCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    await hamt.set("not-a-zarr-key", b"unknown")
    await hamt.make_read_only()

    with pytest.raises(ValueError, match="Cannot classify Zarr key"):
        await convert_hamt_to_sharded(cas, str(hamt.root_node_id), 2)


@given(
    data=st.data(),
    rank=st.integers(min_value=1, max_value=3),
    chunks_per_shard=st.integers(min_value=1, max_value=5),
    path_segments=st.lists(
        st.text(alphabet="abc012", min_size=1, max_size=4),
        min_size=0,
        max_size=3,
    ),
)
@settings(max_examples=25)
def test_v2_chunk_key_linear_mapping_invariant(
    data: st.DataObject,
    rank: int,
    chunks_per_shard: int,
    path_segments: list[str],
) -> None:
    chunks_per_dim = tuple(
        data.draw(st.integers(min_value=1, max_value=4), label=f"chunks_dim_{idx}")
        for idx in range(rank)
    )
    chunk_shape = tuple(
        data.draw(st.integers(min_value=1, max_value=3), label=f"chunk_shape_{idx}")
        for idx in range(rank)
    )
    coords = tuple(
        data.draw(
            st.integers(min_value=0, max_value=chunks_per_dim[idx] - 1),
            label=f"coord_{idx}",
        )
        for idx in range(rank)
    )
    array_shape = tuple(
        chunks * chunk
        for chunks, chunk in zip(chunks_per_dim, chunk_shape, strict=True)
    )
    array_path = "/".join(path_segments)
    array_index = ArrayIndex.new(
        array_path,
        array_shape,
        chunk_shape,
        chunks_per_shard,
    )
    store = ShardedZarrStore(cas=LocalCIDCAS(), read_only=False)
    store._manifest_version = SHARDED_ZARR_V2
    store.array_indices = {array_index.array_path: array_index}

    chunk_key = ShardedZarrStore._format_chunk_key(array_path, coords)
    parsed = store._parse_chunk_key(chunk_key)

    assert parsed is not None
    assert parsed.array_path == array_index.array_path
    assert parsed.coords == coords
    linear_index = store._get_linear_chunk_index_for_index(coords, array_index)
    shard_idx, index_in_shard = store._get_shard_info_for_index(
        linear_index, array_index
    )
    assert shard_idx * chunks_per_shard + index_in_shard == linear_index
    assert (
        ShardedZarrStore._coords_from_linear_index(
            linear_index, array_index.chunks_per_dim
        )
        == coords
    )


def test_array_index_validation_paths() -> None:
    with pytest.raises(ValueError, match="Inconsistent number of shards"):
        ArrayIndex(
            array_path="a",
            array_shape=(4,),
            chunk_shape=(1,),
            chunks_per_shard=2,
            shard_cids=[None],
        )
    with pytest.raises(ValueError, match="chunks_per_shard"):
        ArrayIndex(
            array_path="a",
            array_shape=(1,),
            chunk_shape=(1,),
            chunks_per_shard=0,
            shard_cids=[],
        )
    with pytest.raises(ValueError, match="same rank"):
        ArrayIndex(
            array_path="a",
            array_shape=(1, 1),
            chunk_shape=(1,),
            chunks_per_shard=1,
            shard_cids=[],
        )
    with pytest.raises(ValueError, match="row-major"):
        ArrayIndex(
            array_path="a",
            array_shape=(1,),
            chunk_shape=(1,),
            chunks_per_shard=1,
            shard_cids=[None],
            order="F",
        )
    with pytest.raises(ValueError, match="shard_cids is not a list"):
        ArrayIndex.from_manifest(
            "a",
            {
                "array_shape": [1],
                "chunk_shape": [1],
                "sharding_config": {"chunks_per_shard": 1},
                "shard_cids": "bad",
            },
        )
    with pytest.raises(ValueError, match="sharding_config"):
        ArrayIndex.from_manifest(
            "a",
            {
                "array_shape": [1],
                "chunk_shape": [1],
                "sharding_config": "bad",
                "shard_cids": [None],
            },
        )

    index = ArrayIndex.new("a", (4,), (1,), 2)
    with pytest.raises(ValueError, match="same number of dimensions"):
        index.resize((1, 1))
    index.resize((1,))
    assert index.num_shards == 1
    assert len(index.shard_cids) == 1


@pytest.mark.asyncio
async def test_v1_create_and_open_emit_deprecation_warning() -> None:
    cas = LocalCIDCAS()

    with pytest.warns(
        ShardedZarrV1DeprecationWarning,
        match="sharded_zarr_v1 is deprecated",
    ) as create_warnings:
        store = await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            array_shape=(1,),
            chunk_shape=(1,),
            chunks_per_shard=1,
        )

    assert store._root_obj["manifest_version"] == SHARDED_ZARR_V1
    assert "group='0'" in str(create_warnings[0].message)

    root_cid = await store.flush()
    with pytest.warns(
        ShardedZarrV1DeprecationWarning,
        match="Prefer sharded_zarr_v2",
    ):
        reopened = await ShardedZarrStore.open(
            cas=cas, read_only=True, root_cid=root_cid
        )
    assert reopened._manifest_version == SHARDED_ZARR_V1


@pytest.mark.asyncio
async def test_v2_create_and_open_do_not_emit_v1_deprecation_warning() -> None:
    cas = LocalCIDCAS()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        store = await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            chunks_per_shard=1,
            manifest_version=SHARDED_ZARR_V2,
        )
        root_cid = await store.flush()
        await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)

    assert not any(
        issubclass(warning.category, ShardedZarrV1DeprecationWarning)
        for warning in caught_warnings
    )


@pytest.mark.asyncio
async def test_v2_validation_paths() -> None:
    cas = LocalCIDCAS()
    with pytest.raises(ValueError, match="Incompatible manifest version"):
        await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            chunks_per_shard=1,
            manifest_version="bad",
        )
    with pytest.raises(ValueError, match="both be provided"):
        await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            chunks_per_shard=1,
            manifest_version=SHARDED_ZARR_V2,
            array_shape=(1,),
        )
    with pytest.raises(ValueError, match="must be provided"):
        await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            chunks_per_shard=1,
            manifest_version="sharded_zarr_v1",
        )

    seeded_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
        array_shape=(1,),
        chunk_shape=(1,),
        primary_array_path="seed",
    )
    assert seeded_store.array_indices["seed"].array_shape == (1,)

    assert ShardedZarrStore._array_path_from_metadata_key("a/.zarray") == "a"
    assert ShardedZarrStore._format_chunk_key("", (0,)) == "c/0"

    empty_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    assert empty_store._root_obj["sharding_config"]["chunks_per_shard"] == 1
    with pytest.raises(RuntimeError, match="not properly initialized"):
        await empty_store.resize_store((1,))
    with pytest.raises(ValueError, match="non-empty"):
        await seeded_store.migrate_v1_to_v2("")
    with pytest.raises(ValueError, match="Only sharded_zarr_v1"):
        await seeded_store.migrate_v1_to_v2("seed")

    assert ShardedZarrStore._array_path_from_metadata_key("attrs") is None
    assert ShardedZarrStore._decode_metadata_json(b"not-json") is None
    assert ShardedZarrStore._decode_metadata_json(b"[]") is None
    assert ShardedZarrStore._extract_array_metadata({}) is None
    assert ShardedZarrStore._extract_array_metadata({"shape": [1]}) is None

    await empty_store._register_array_metadata_from_bytes(
        "attrs", json.dumps({"shape": [1]}).encode()
    )
    await empty_store._register_array_metadata_from_bytes("a/zarr.json", b"not-json")
    await empty_store._register_array_metadata_from_bytes(
        "a/zarr.json", json.dumps({"shape": [1]}).encode()
    )
    assert empty_store.array_indices == {}
    await empty_store._clear_v2_unlocked()
    await empty_store._prune_v2_array_indices_for_prefix("missing")
    with pytest.raises(RuntimeError, match="chunks_per_shard"):
        empty_store._default_chunks_per_shard = None
        empty_store._register_or_update_array_index(
            array_path="a", array_shape=(1,), chunk_shape=(1,)
        )

    await empty_store.delete("missing/c/0")
    empty_store._root_obj["metadata"]["missing"] = None
    empty_store._root_obj["metadata"]["missing/"] = None
    assert {entry async for entry in empty_store.list_dir("missing")} == set()

    empty_source_cid = await empty_store.flush()
    await seeded_store.graft_store(empty_source_cid, chunk_offset=(0,))

    v1_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(1,),
        chunk_shape=(1,),
        chunks_per_shard=1,
    )
    proto = zarr.core.buffer.default_buffer_prototype()
    await v1_store.set("zarr.json", proto.buffer.from_bytes(b"{}"))
    await v1_store.delete_dir("")
    assert "zarr.json" not in v1_store._root_obj["metadata"]
    await v1_store.set("zarr.json", proto.buffer.from_bytes(b"{}"))
    await v1_store.clear()
    assert "zarr.json" not in v1_store._root_obj["metadata"]
    await v1_store._clear_v2_unlocked()
    await v1_store._prune_v2_array_indices_for_prefix("")
    with pytest.raises(ValueError, match="invalid literal"):
        await v1_store.get("temp/c/invalid", proto)
    with pytest.raises(ValueError, match="invalid literal"):
        await v1_store.set("temp/c/invalid", proto.buffer.from_bytes(b"invalid"))
    invalid_v1_pointer = await cas.save(b"invalid", codec="raw")
    with pytest.raises(ValueError, match="invalid literal"):
        await v1_store.set_pointer("temp/c/invalid", str(invalid_v1_pointer))

    classic_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=1,
        manifest_version=SHARDED_ZARR_V2,
    )
    array_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2],
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1]},
        },
    }
    await classic_store.set(
        "zarr.json",
        proto.buffer.from_bytes(json.dumps(array_metadata).encode()),
    )
    await classic_store.set(
        "c/FPAR/zarr.json",
        proto.buffer.from_bytes(json.dumps(array_metadata).encode()),
    )
    await classic_store.set("c/0", proto.buffer.from_bytes(b"root"))
    await classic_store.set("c/FPAR/c/0", proto.buffer.from_bytes(b"group"))

    assert await classic_store.exists("0")
    assert await classic_store.exists("c/FPAR/0")
    assert not await classic_store.exists("0.0")
    assert not await classic_store.exists("0/0")
    assert not await classic_store.exists("c/FPAR/not-a-chunk")
    assert await classic_store.get("c/FPAR/not-a-chunk", proto) is None
    await classic_store.delete("c/FPAR/not-a-chunk")
    invalid_pointer = await cas.save(b"ignored", codec="raw")
    await classic_store.set_pointer("c/FPAR/not-a-chunk", str(invalid_pointer))
    assert not await classic_store.exists("c/FPAR/not-a-chunk")
    await classic_store.set("c/FPAR/not-a-chunk", proto.buffer.from_bytes(b"ignored"))
    assert not await classic_store.exists("c/FPAR/not-a-chunk")
    assert not await classic_store.exists("missing.0")


@pytest.mark.asyncio
async def test_v2_invalid_root_and_shard_validation() -> None:
    cas = LocalCIDCAS()

    invalid_v2_roots = [
        {"manifest_version": SHARDED_ZARR_V2, "metadata": [], "arrays": {}},
        {
            "manifest_version": SHARDED_ZARR_V2,
            "metadata": {},
            "arrays": {
                "bad": {
                    "array_shape": [1],
                    "chunk_shape": [1],
                    "sharding_config": {"chunks_per_shard": 1},
                    "shard_cids": "bad",
                }
            },
        },
        {
            "manifest_version": SHARDED_ZARR_V2,
            "metadata": {},
            "arrays": {
                "bad": {
                    "array_shape": [2],
                    "chunk_shape": [1],
                    "sharding_config": {"chunks_per_shard": 1},
                    "shard_cids": [None],
                }
            },
        },
    ]
    for root_obj in invalid_v2_roots:
        root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
        with pytest.raises(ValueError):
            await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=str(root_cid))

    store = ShardedZarrStore(cas=cas, read_only=True)
    store._root_obj = {
        "manifest_version": SHARDED_ZARR_V2,
        "metadata": {},
        "arrays": {1: {}},
    }
    with pytest.raises(ValueError, match="arrays must map"):
        store._load_v2_root()

    bad_shard_cid = await cas.save(dag_cbor.encode([1]), codec="dag-cbor")
    root_obj = {
        "manifest_version": SHARDED_ZARR_V2,
        "metadata": {},
        "arrays": {
            "a": {
                "array_shape": [1],
                "chunk_shape": [1],
                "sharding_config": {"chunks_per_shard": 1},
                "shard_cids": [bad_shard_cid],
            }
        },
    }
    root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=str(root_cid))
    with pytest.raises(TypeError, match="non-CID"):
        await store.get("a/c/0", zarr.core.buffer.default_buffer_prototype())

    short_shard_cid = await cas.save(dag_cbor.encode([None]), codec="dag-cbor")
    root_obj = {
        "manifest_version": SHARDED_ZARR_V2,
        "metadata": {},
        "arrays": {
            "a": {
                "array_shape": [2],
                "chunk_shape": [1],
                "sharding_config": {"chunks_per_shard": 2},
                "shard_cids": [short_shard_cid],
            }
        },
    }
    root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=str(root_cid))
    with pytest.raises(ValueError, match="expected 2"):
        await store.get("a/c/0", zarr.core.buffer.default_buffer_prototype())

    stale_tail_cid = await cas.save(b"stale", codec="raw")
    assert ShardedZarrStore._remap_shards_for_resize(
        {0: [None, stale_tail_cid]},
        old_chunks_per_dim=(1,),
        old_total_chunks=1,
        new_array_index=ArrayIndex.new("a", (1,), (1,), 2),
    ) == {0: [None, None]}

    extra_chunk_cid = await cas.save(b"extra", codec="raw")
    sparse_shard_cid = await cas.save(
        dag_cbor.encode([None, extra_chunk_cid]), codec="dag-cbor"
    )
    root_obj = {
        "manifest_version": SHARDED_ZARR_V2,
        "metadata": {},
        "arrays": {
            "a": {
                "array_shape": [1],
                "chunk_shape": [1],
                "sharding_config": {"chunks_per_shard": 2},
                "shard_cids": [sparse_shard_cid],
            }
        },
    }
    root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=str(root_cid))
    assert {key async for key in store.list_prefix("a/c/")} == set()
