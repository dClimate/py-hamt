import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import dag_cbor
import pytest
import zarr
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
    """Writing the inferred primary may seal it and must preserve its chunks."""
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

    # Guard the complementary case first: a write under the inferred primary
    # itself is accepted, may seal "temp", and survives a reopen.
    await assert_inferred_primary_write_is_accepted(cas, legacy_root_cid)

    recorded = await exercise_non_primary_write(cas, recorded_root_cid)
    assert recorded == NonPrimaryWriteOutcome(
        live_primary="temp",
        persisted_primary="temp",
        precip_in_metadata=True,
        temp_chunks=TEMP_CHUNKS,
        precip_chunk=PRECIP_CHUNK,
    )

    inferred = await exercise_non_primary_write(cas, legacy_root_cid)
    violations: list[str] = []
    if inferred.live_primary != recorded.live_primary:
        violations.append(
            f"in-memory primary changed to {inferred.live_primary!r}, expected 'temp'"
        )
    assert inferred.persisted_primary in (None, "temp")
    if inferred.precip_in_metadata != recorded.precip_in_metadata:
        violations.append("precip write did not use recorded-primary metadata routing")
    if inferred.temp_chunks != recorded.temp_chunks:
        violations.append(f"temp chunks changed after reopen: {inferred.temp_chunks!r}")
    if inferred.precip_chunk != recorded.precip_chunk:
        violations.append(
            f"precip bytes differ: {inferred.precip_chunk!r} != {recorded.precip_chunk!r}"
        )
    assert not violations, "\n".join(violations)


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
async def test_same_geometry_secondary_write_stays_in_metadata() -> None:
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
    assert chunk_info.get("primary_array_path") in (None, "temp")


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
