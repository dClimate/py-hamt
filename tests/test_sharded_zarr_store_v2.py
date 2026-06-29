import json

import dag_cbor
import numpy as np
import pytest
import xarray as xr
import zarr.core.buffer
from dag_cbor.ipld import IPLDKind
from multiformats import CID, multihash
from zarr.abc.store import RangeByteRequest

from py_hamt import HAMT
from py_hamt.hamt_to_sharded_converter import (
    _is_zarr_chunk_key,
    convert_hamt_to_sharded,
)
from py_hamt.sharded_zarr_store import SHARDED_ZARR_V2, ArrayIndex, ShardedZarrStore
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
        id: IPLDKind,
        offset: int | None = None,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes:
        data = self.store[self._key(id)]
        if offset is not None:
            if length is None:
                return data[offset:]
            return data[offset : offset + length]
        if suffix is not None:
            return data[-suffix:]
        return data


def _pyramid_level(data: np.ndarray) -> xr.Dataset:
    return xr.Dataset(
        {"FPAR": (("time", "y", "x"), data)},
        coords={
            "time": np.arange(data.shape[0]),
            "y": np.arange(data.shape[1]),
            "x": np.arange(data.shape[2]),
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

    level_0 = _pyramid_level(np.arange(16).reshape(2, 2, 4)).chunk(
        {"time": 1, "y": 1, "x": 2}
    )
    level_1 = _pyramid_level(np.arange(8).reshape(2, 2, 2) + 100).chunk(
        {"time": 1, "y": 2, "x": 1}
    )
    level_2 = _pyramid_level(np.arange(4).reshape(2, 1, 2) + 200).chunk(
        {"time": 1, "y": 1, "x": 1}
    )

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
    assert await store.exists("0/x/c/0")
    assert await store.exists("0/y/c/0")
    assert await store.exists("0/time/c/0")

    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(
        cas=cas, read_only=True, root_cid=root_cid
    )

    xr.testing.assert_identical(
        level_0, xr.open_zarr(store=read_store, group="0").compute()
    )
    xr.testing.assert_identical(
        level_1, xr.open_zarr(store=read_store, group="1").compute()
    )
    xr.testing.assert_identical(
        level_2, xr.open_zarr(store=read_store, group="2").compute()
    )

    root_entries = {entry async for entry in read_store.list_dir("")}
    assert {"0", "1", "2", "zarr.json"}.issubset(root_entries)
    level_entries = {entry async for entry in read_store.list_dir("0")}
    assert {"FPAR", "x", "y", "time", "zarr.json"}.issubset(level_entries)

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

    write_store = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=root_cid
    )
    await write_store.delete("1/FPAR/c/0/0/0")
    assert not await write_store.exists("1/FPAR/c/0/0/0")
    assert await write_store.exists("0/FPAR/c/0/0/0")


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
    assert await migrated_store.get("FPAR/c/0/0", proto) is None


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
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [1],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1]},
                    },
                }
            ).encode()
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
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [2]},
                    },
                }
            ).encode()
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
    reopened = await ShardedZarrStore.open(
        cas=cas, read_only=False, root_cid=root_cid
    )
    proto = zarr.core.buffer.default_buffer_prototype()

    await reopened.set(
        "a/zarr.json",
        proto.buffer.from_bytes(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1]},
                    },
                }
            ).encode()
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
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [1],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1]},
                    },
                }
            ).encode()
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
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [1, 1, 1],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1, 1, 1]},
                    },
                }
            ).encode()
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
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [2, 1],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1, 1]},
                    },
                }
            ).encode()
        ),
    )
    await store.set("a/c/1/0", proto.buffer.from_bytes(b"chunk"))
    await store.set(
        "b/zarr.json",
        proto.buffer.from_bytes(
            json.dumps(
                {
                    "zarr_format": 3,
                    "node_type": "array",
                    "shape": [1, 1],
                    "chunk_grid": {
                        "name": "regular",
                        "configuration": {"chunk_shape": [1, 1]},
                    },
                }
            ).encode()
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


@pytest.mark.asyncio
async def test_converter_discovers_grouped_arrays() -> None:
    cas = LocalCIDCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    source_store = ZarrHAMTStore(hamt)

    level_0 = _pyramid_level(np.arange(4).reshape(1, 2, 2)).chunk(
        {"time": 1, "y": 1, "x": 1}
    )
    level_1 = _pyramid_level(np.arange(2).reshape(1, 1, 2) + 10).chunk(
        {"time": 1, "y": 1, "x": 1}
    )
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
    store._root_obj = {"manifest_version": SHARDED_ZARR_V2, "metadata": {}, "arrays": {1: {}}}
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
