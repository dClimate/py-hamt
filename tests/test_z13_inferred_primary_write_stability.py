import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import dag_cbor
import pytest
import zarr
from dag_cbor.ipld import IPLDKind
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
TEMP_CHUNKS = {
    "temp/c/0/0": b"temp-00",
    "temp/c/0/1": b"temp-01",
    "temp/c/1/0": b"temp-10",
    "temp/c/1/1": b"temp-11",
}
PRECIP_KEY = "precip/c/0/0"
PRECIP_CHUNK = b"precip-00"
FOREIGN_SAME_RANK_KEY = "precip/c/0/0"
FOREIGN_SAME_RANK_CHUNK = b"precip-same-rank-00"


def array_metadata(*, shape: tuple[int, ...], chunk_shape: tuple[int, ...]) -> bytes:
    return json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": list(shape),
            "data_type": "uint8",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(chunk_shape)},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
            "attributes": {},
        },
        separators=(",", ":"),
    ).encode()


def buf(data: bytes) -> zarr.core.buffer.Buffer:
    return PROTOTYPE.buffer.from_bytes(data)


class FailingShardLoadCAS(CIDInMemoryCAS):
    """Fail one selected load after a legacy root has been opened."""

    def __init__(self) -> None:
        super().__init__()
        self.failing_cid: IPLDKind | None = None

    async def load(
        self,
        identifier: IPLDKind,
        offset: int | None = None,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes:
        if self.failing_cid is not None and str(identifier) == str(self.failing_cid):
            raise RuntimeError("simulated shard load failure")
        return await super().load(
            identifier,
            offset=offset,
            length=length,
            suffix=suffix,
        )


async def decoded_root(cas: CIDInMemoryCAS, root_cid: str) -> dict[str, Any]:
    root_obj = dag_cbor.decode(await cas.load(root_cid))
    assert isinstance(root_obj, dict)
    return cast(dict[str, Any], root_obj)


async def v1_recorded_and_legacy_roots() -> tuple[CIDInMemoryCAS, str, str]:
    """Build equivalent V1 roots with and without the recorded primary field."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set(
        "temp/zarr.json",
        buf(array_metadata(shape=(4, 4), chunk_shape=(2, 2))),
    )
    # This is metadata for another array, but its geometry excludes it from
    # legacy-primary inference. Thus "temp" remains the unique candidate.
    await store.set(
        "precip/zarr.json",
        buf(array_metadata(shape=(4, 4), chunk_shape=(1, 1))),
    )
    for key, data in TEMP_CHUNKS.items():
        await store.set(key, buf(data))
    recorded_root_cid = await store.flush()

    root_obj = await decoded_root(cas, recorded_root_cid)
    chunk_info = root_obj["chunks"]
    assert isinstance(chunk_info, dict)
    assert chunk_info["primary_array_path"] == "temp"
    chunk_info.pop("primary_array_path")
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    return cas, recorded_root_cid, str(legacy_root_cid)


async def read_bytes(store: ShardedZarrStore, key: str) -> bytes | None:
    value = await store.get(key, PROTOTYPE)
    return None if value is None else value.to_bytes()


async def read_temp_chunks(store: ShardedZarrStore) -> dict[str, bytes | None]:
    return {key: await read_bytes(store, key) for key in TEMP_CHUNKS}


@dataclass(frozen=True)
class NonPrimaryWriteOutcome:
    live_primary: str | None
    persisted_primary: str | None
    precip_in_metadata: bool
    temp_chunks: Mapping[str, bytes | None]
    precip_chunk: bytes | None


async def exercise_non_primary_write(
    cas: CIDInMemoryCAS, root_cid: str
) -> NonPrimaryWriteOutcome:
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=root_cid,
    )
    assert store._primary_array_path == "temp"

    await store.set(PRECIP_KEY, buf(PRECIP_CHUNK))
    live_primary = store._primary_array_path
    flushed_root_cid = await store.flush()
    root_obj = await decoded_root(cas, flushed_root_cid)
    chunk_info = root_obj["chunks"]
    metadata = root_obj["metadata"]
    assert isinstance(chunk_info, dict)
    assert isinstance(metadata, dict)
    persisted_primary = chunk_info.get("primary_array_path")
    assert persisted_primary is None or isinstance(persisted_primary, str)

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )
    return NonPrimaryWriteOutcome(
        live_primary=live_primary,
        persisted_primary=persisted_primary,
        precip_in_metadata=PRECIP_KEY in metadata,
        temp_chunks=await read_temp_chunks(reopened),
        precip_chunk=await read_bytes(reopened, PRECIP_KEY),
    )


async def assert_inferred_primary_write_is_accepted(
    cas: CIDInMemoryCAS, legacy_root_cid: str
) -> None:
    """A writable open records the unambiguous inferred primary and keeps chunks.

    Recording makes routing deterministic: a later reopen reads from the sealed
    primary instead of re-inferring, so a second same-geometry array can no
    longer make the reopen inference ambiguous and misroute a chunk.
    """
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=legacy_root_cid,
    )
    assert store._primary_array_path == "temp"

    replacement = b"updated-temp-11"
    await store.set("temp/c/1/1", buf(replacement))
    assert store._primary_array_path == "temp"
    flushed_root_cid = await store.flush()
    root_obj = await decoded_root(cas, flushed_root_cid)
    chunk_info = root_obj["chunks"]
    assert isinstance(chunk_info, dict)
    assert chunk_info["primary_array_path"] == "temp"

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )
    expected_chunks = {**TEMP_CHUNKS, "temp/c/1/1": replacement}
    assert await read_temp_chunks(reopened) == expected_chunks


@pytest.mark.asyncio
async def test_non_primary_write_matches_recorded_primary_semantics() -> None:
    cas, recorded_root_cid, legacy_root_cid = await v1_recorded_and_legacy_roots()

    # A writable open records the unambiguous inferred primary, so a legacy root
    # behaves identically to one that already had the primary recorded.
    await assert_inferred_primary_write_is_accepted(cas, legacy_root_cid)

    expected = NonPrimaryWriteOutcome(
        live_primary="temp",
        persisted_primary="temp",
        precip_in_metadata=True,
        temp_chunks=TEMP_CHUNKS,
        precip_chunk=PRECIP_CHUNK,
    )
    recorded = await exercise_non_primary_write(cas, recorded_root_cid)
    inferred = await exercise_non_primary_write(cas, legacy_root_cid)
    assert recorded == expected
    assert inferred == expected


@pytest.mark.asyncio
async def test_non_primary_write_keeps_every_temp_chunk_readable_after_reopen() -> None:
    cas, _, legacy_root_cid = await v1_recorded_and_legacy_roots()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=legacy_root_cid,
    )
    assert store._primary_array_path == "temp"

    await store.set(PRECIP_KEY, buf(PRECIP_CHUNK))
    flushed_root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )

    assert await read_temp_chunks(reopened) == TEMP_CHUNKS


@pytest.mark.asyncio
async def test_recorded_empty_primary_routes_named_chunk_to_metadata() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    root_chunk_key = "/c/0/0"
    original_root_chunk = b"root-00"
    await store.set(root_chunk_key, buf(original_root_chunk))
    root_cid = await store.flush()

    initial_root = await decoded_root(cas, root_cid)
    initial_chunk_info = initial_root["chunks"]
    assert isinstance(initial_chunk_info, dict)
    assert initial_chunk_info["primary_array_path"] == ""

    reopened_for_write = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=root_cid,
    )
    named_chunk_key = "named/c/0/0"
    await reopened_for_write.set(named_chunk_key, buf(b"named-00"))
    mutated_root_cid = await reopened_for_write.flush()

    mutated_root = await decoded_root(cas, mutated_root_cid)
    mutated_chunk_info = mutated_root["chunks"]
    mutated_metadata = mutated_root["metadata"]
    assert isinstance(mutated_chunk_info, dict)
    assert isinstance(mutated_metadata, dict)

    reopened_for_read = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=mutated_root_cid,
    )
    assert await read_bytes(reopened_for_read, root_chunk_key) == original_root_chunk
    assert named_chunk_key in mutated_metadata
    assert mutated_chunk_info["primary_array_path"] == ""


@pytest.mark.asyncio
async def test_recorded_empty_primary_listing_round_trips_through_get() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    root_chunk_key = "/c/0/0"
    canonical_root_chunk_key = "c/0/0"
    root_chunk = b"root-00"
    await store.set(root_chunk_key, buf(root_chunk))
    root_cid = await store.flush()

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    listed = [key async for key in reopened.list()]

    assert listed == [canonical_root_chunk_key]
    listed_value = await reopened.get(listed[0], PROTOTYPE)
    assert listed_value is not None
    assert listed_value.to_bytes() == root_chunk

    root_obj = await decoded_root(cas, root_cid)
    chunk_info = root_obj["chunks"]
    assert isinstance(chunk_info, dict)
    chunk_info.pop("primary_array_path")
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")
    legacy_reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root_cid),
    )
    legacy_listed = [key async for key in legacy_reopened.list()]
    assert legacy_listed == [root_chunk_key]
    legacy_value = await legacy_reopened.get(legacy_listed[0], PROTOTYPE)
    assert legacy_value is not None
    assert legacy_value.to_bytes() == root_chunk


@pytest.mark.asyncio
async def test_failed_first_primary_write_does_not_persist_its_path() -> None:
    cas = FailingShardLoadCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    original_key = "temp/c/0/0"
    original_chunk = b"original-temp"
    await store.set(original_key, buf(original_chunk))
    recorded_root_cid = await store.flush()

    root_obj = await decoded_root(cas, recorded_root_cid)
    chunk_info = root_obj["chunks"]
    assert isinstance(chunk_info, dict)
    shard_cid = chunk_info["shard_cids"][0]
    chunk_info.pop("primary_array_path")
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=str(legacy_root_cid),
    )
    cas.failing_cid = shard_cid
    with pytest.raises(RuntimeError, match="simulated shard load failure"):
        await reopened.set("wrong/c/0/0", buf(b"failed-write"))

    assert "primary_array_path" not in reopened._root_obj["chunks"]
    assert reopened._primary_array_path == ""

    cas.failing_cid = None
    flushed_root_cid = await reopened.flush()
    persisted_root = await decoded_root(cas, flushed_root_cid)
    assert "primary_array_path" not in persisted_root["chunks"]

    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )
    assert await read_bytes(read_store, original_key) == original_chunk


@pytest.mark.asyncio
async def test_same_geometry_secondary_write_is_routed_deterministically() -> None:
    """A second same-geometry array must not corrupt the primary across a reopen.

    Because the writable open records the inferred primary, the reopen routes
    from that recorded value instead of re-inferring. Without recording, the
    second array's metadata made the reopen inference ambiguous, so its
    metadata-stored chunk re-parsed against the shared shard index and read the
    primary's slot (``temp-00``) instead of its own bytes. Recording makes the
    secondary chunk read back its own value.
    """
    cas, _, legacy_root_cid = await v1_recorded_and_legacy_roots()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=legacy_root_cid,
    )
    assert store._primary_array_path == "temp"

    await store.set(
        "precip2/zarr.json",
        buf(array_metadata(shape=(4, 4), chunk_shape=(2, 2))),
    )
    secondary_key = "precip2/c/0/0"
    secondary_chunk = b"precip2-00"
    await store.set(secondary_key, buf(secondary_chunk))

    assert store._primary_array_path == "temp"
    assert await read_temp_chunks(store) == TEMP_CHUNKS
    flushed_root_cid = await store.flush()
    root_obj = await decoded_root(cas, flushed_root_cid)
    chunk_info = root_obj["chunks"]
    metadata = root_obj["metadata"]
    assert isinstance(chunk_info, dict)
    assert isinstance(metadata, dict)
    assert secondary_key in metadata
    # The inferred primary is now recorded, so the reopen is unambiguous.
    assert chunk_info["primary_array_path"] == "temp"

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )
    assert await read_temp_chunks(reopened) == TEMP_CHUNKS
    # The corruption fix: the secondary reads its OWN bytes, not temp's slot.
    assert await read_bytes(reopened, secondary_key) == secondary_chunk


@pytest.mark.asyncio
async def test_metadata_routed_foreign_chunk_lifecycle_preserves_primary() -> None:
    cas, _, legacy_root_cid = await v1_recorded_and_legacy_roots()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=legacy_root_cid,
    )
    assert store._primary_array_path == "temp"

    await store.set(PRECIP_KEY, buf(PRECIP_CHUNK))
    assert await store.exists(PRECIP_KEY)
    assert await read_bytes(store, PRECIP_KEY) == PRECIP_CHUNK
    assert await read_temp_chunks(store) == TEMP_CHUNKS

    await store.delete(PRECIP_KEY)
    assert not await store.exists(PRECIP_KEY)
    assert await read_bytes(store, PRECIP_KEY) is None
    assert await read_temp_chunks(store) == TEMP_CHUNKS

    flushed_root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )
    assert not await reopened.exists(PRECIP_KEY)
    assert await read_bytes(reopened, PRECIP_KEY) is None
    assert await read_temp_chunks(reopened) == TEMP_CHUNKS


async def v1_root_primary_legacy_root() -> tuple[CIDInMemoryCAS, str]:
    """A legacy V1 root whose primary infers to the root array itself (path "").

    The root array's own metadata makes "" the unique inference candidate. No
    ``primary_array_path`` is ever sealed (no primary chunk is written), so the
    flushed root is legacy-shaped and reopening re-infers "".
    """
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("zarr.json", buf(array_metadata(shape=(4, 4), chunk_shape=(2, 2))))
    root_cid = await store.flush()

    root_obj = await decoded_root(cas, root_cid)
    chunk_info = root_obj["chunks"]
    assert isinstance(chunk_info, dict)
    assert "primary_array_path" not in chunk_info
    return cas, root_cid


@pytest.mark.asyncio
async def test_inferred_root_primary_is_recorded_and_not_rebound() -> None:
    """A foreign same-rank chunk must not rebind an inferred root primary ("").

    Inference sets the primary to the root array (""). A writable open records
    that unambiguous inference — even the empty-string path — so the foreign
    write routes to metadata and the reopen reads from the recorded primary
    rather than re-inferring. This keeps the root shard slot intact and the
    foreign chunk readable from its own metadata entry.
    """
    cas, legacy_root_cid = await v1_root_primary_legacy_root()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=legacy_root_cid,
    )
    assert store._primary_array_path == ""
    assert store._primary_inferred is True

    await store.set(FOREIGN_SAME_RANK_KEY, buf(FOREIGN_SAME_RANK_CHUNK))

    # The inferred root primary is not rebound in memory and is recorded to disk.
    assert store._primary_array_path == ""
    flushed_root_cid = await store.flush()
    root_obj = await decoded_root(cas, flushed_root_cid)
    chunk_info = root_obj["chunks"]
    metadata = root_obj["metadata"]
    assert isinstance(chunk_info, dict)
    assert isinstance(metadata, dict)
    assert chunk_info["primary_array_path"] == ""
    assert FOREIGN_SAME_RANK_KEY in metadata

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=flushed_root_cid,
    )
    assert reopened._primary_array_path == ""
    # The reopen routes from the recorded primary; it does not re-infer.
    assert reopened._primary_inferred is False
    assert await read_bytes(reopened, FOREIGN_SAME_RANK_KEY) == FOREIGN_SAME_RANK_CHUNK


@pytest.mark.asyncio
async def test_unrecorded_empty_primary_list_dir_is_consistent() -> None:
    """list_dir("c") must surface chunks for an unrecorded empty-primary root.

    Such a root emits its shard tree with a leading slash ("/c/...") to keep it
    distinct from legacy metadata keys, but list_dir prefixes are slash-stripped.
    list_dir("") advertises "c", so descending into list_dir("c") must return the
    chunk components rather than an empty listing.
    """
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("/c/0/0", buf(b"root-00"))
    root_cid = await store.flush()

    # Drop the sealed primary to model a legacy, unrecorded-primary root.
    root_obj = await decoded_root(cas, root_cid)
    chunk_info = root_obj["chunks"]
    assert isinstance(chunk_info, dict)
    chunk_info.pop("primary_array_path")
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root_cid),
    )
    assert reopened._primary_array_path == ""
    top_level = {key async for key in reopened.list_dir("")}
    assert "c" in top_level
    assert [key async for key in reopened.list_dir("c")] == ["0"]
