import json
from typing import Any, cast

import dag_cbor
import pytest
import zarr
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4, 4],
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
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {},
    },
    separators=(",", ":"),
).encode()
CHUNKLESS_ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4, 4],
    },
    separators=(",", ":"),
).encode()


def buf(data: bytes) -> zarr.core.buffer.Buffer:
    return PROTOTYPE.buffer.from_bytes(data)


def decode_root(data: bytes) -> dict[str, Any]:
    root_obj = dag_cbor.decode(data)
    assert isinstance(root_obj, dict)
    return cast(dict[str, Any], root_obj)


@pytest.mark.asyncio
async def test_v1_legacy_root_infers_primary_array_path_for_listing() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )

    chunk = await reopened.get("temp/c/0/0", PROTOTYPE)
    assert chunk is not None
    assert chunk.to_bytes() == b"chunk"

    listed = [key async for key in reopened.list()]
    assert "temp/c/0/0" in listed and "c/0/0" not in listed
    assert [key async for key in reopened.list_prefix("temp/c/")] == ["temp/c/0/0"]


@pytest.mark.asyncio
async def test_inference_rejects_lone_chunkless_candidate() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    malformed_cid = await cas.save(CHUNKLESS_ARRAY_METADATA, codec="raw")
    root_obj["metadata"] = {"malformed/zarr.json": malformed_cid}
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path in (None, "")


@pytest.mark.asyncio
async def test_inference_ignores_chunkless_candidate_beside_valid_array() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    root_obj["metadata"]["malformed/zarr.json"] = await cas.save(
        CHUNKLESS_ARRAY_METADATA,
        codec="raw",
    )
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path == "temp"


class FlakyMetadataCAS(CIDInMemoryCAS):
    """Fails loads of one specific CID to simulate a partial CAS outage."""

    def __init__(self) -> None:
        super().__init__()
        self.failing_cid: object = None

    async def load(self, id, offset=None, length=None, suffix=None):  # type: ignore[override]
        if self.failing_cid is not None and str(id) == str(self.failing_cid):
            raise ConnectionError("simulated partial CAS outage")
        return await super().load(id, offset=offset, length=length, suffix=suffix)


@pytest.mark.asyncio
async def test_inference_aborts_when_a_candidate_cannot_be_loaded() -> None:
    """If any candidate's metadata cannot be loaded, the true primary might be
    the unreadable one — inference must abort (keep the legacy '' default)
    rather than confidently adopt a surviving same-shape candidate."""
    cas = FlakyMetadataCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("precip/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    # Make temp's metadata unreadable: only "precip" would survive the scan.
    metadata = root_obj["metadata"]
    cas.failing_cid = metadata["temp/zarr.json"]

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path in (None, ""), (
        "a partial CAS outage must not rebind shard data under a surviving "
        f"candidate, got {reopened._primary_array_path!r}"
    )


@pytest.mark.asyncio
async def test_inference_skips_non_candidates_and_dedupes_dual_format() -> None:
    """The scan must skip coordinate arrays, undecodable blobs, shape and
    chunk-shape mismatches, non-string keys, and count an array once even
    when it registers both zarr.json and v2 .zarray metadata."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"chunk"))
    root = await store.flush()

    def meta(payload: dict) -> bytes:
        return json.dumps(payload).encode()

    root_obj = decode_root(await cas.load(root))
    root_obj["chunks"].pop("primary_array_path")
    metadata = root_obj["metadata"]
    metadata["lat/zarr.json"] = await cas.save(meta({"shape": [4, 4]}), codec="raw")
    metadata["garbage/zarr.json"] = await cas.save(b"not-json", codec="raw")
    metadata["wrongshape/zarr.json"] = await cas.save(
        meta({"shape": [8, 8]}), codec="raw"
    )
    metadata["wrongchunks/zarr.json"] = await cas.save(
        meta({
            "shape": [4, 4],
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [4, 4]},
            },
        }),
        codec="raw",
    )
    # v2-format dual registration of the same array: "chunks" key, same grid.
    metadata["temp/.zarray"] = await cas.save(
        meta({"shape": [4, 4], "chunks": [2, 2]}), codec="raw"
    )
    legacy_root = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root),
    )
    assert reopened._primary_array_path == "temp"
    listed = [key async for key in reopened.list()]
    assert "temp/c/0/0" in listed


@pytest.mark.asyncio
async def test_inference_handles_malformed_metadata_maps() -> None:
    """Direct-call coverage for defensive branches dag-cbor cannot produce:
    a non-mapping metadata value and a non-string metadata key."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    store._root_obj["chunks"].pop("primary_array_path", None)

    store._root_obj["metadata"] = None
    await store._infer_v1_legacy_primary_array_path()
    assert store._primary_array_path in (None, "")

    cid = await cas.save(b"payload", codec="raw")
    store._root_obj["metadata"] = {1: cid}
    await store._infer_v1_legacy_primary_array_path()
    assert store._primary_array_path in (None, "")
