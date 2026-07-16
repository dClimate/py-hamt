import json

import pytest
import zarr
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
COORD_ARRAY_METADATA = json.dumps(
    {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [4],
        "data_type": "uint8",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [2]},
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


async def new_v1_store(cas: CIDInMemoryCAS) -> ShardedZarrStore:
    return await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )


async def assert_value(store: ShardedZarrStore, key: str, expected: bytes) -> None:
    value = await store.get(key, PROTOTYPE)
    assert value is not None
    assert value.to_bytes() == expected


@pytest.mark.asyncio
async def test_v1_coordinate_chunk_before_primary_chunk_round_trips() -> None:
    cas = CIDInMemoryCAS()
    store = await new_v1_store(cas)

    await store.set("y/zarr.json", buf(COORD_ARRAY_METADATA))
    await store.set("y/c/0", buf(b"coord-chunk"))
    await assert_value(store, "y/c/0", b"coord-chunk")

    await store.set("temp/c/0/0", buf(b"primary"))
    await assert_value(store, "temp/c/0/0", b"primary")

    root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    await assert_value(reopened, "y/c/0", b"coord-chunk")


@pytest.mark.asyncio
async def test_v1_coordinate_chunk_after_primary_chunk_round_trips() -> None:
    cas = CIDInMemoryCAS()
    store = await new_v1_store(cas)

    await store.set("temp/c/0/0", buf(b"primary"))
    await assert_value(store, "temp/c/0/0", b"primary")

    await store.set("y/zarr.json", buf(COORD_ARRAY_METADATA))
    await store.set("y/c/0", buf(b"coord-chunk"))
    await assert_value(store, "y/c/0", b"coord-chunk")

    root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    await assert_value(reopened, "y/c/0", b"coord-chunk")


@pytest.mark.asyncio
async def test_v1_wrong_rank_key_under_recorded_primary_fails_loud() -> None:
    """A malformed (wrong-rank) key under the recorded primary path must keep
    failing coordinate validation instead of being silently reclassified as
    metadata — masking it would let a store opened with a mismatched
    array_shape divert every chunk into root metadata."""
    cas = CIDInMemoryCAS()
    store = await new_v1_store(cas)

    await store.set("temp/zarr.json", buf(COORD_ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"primary"))  # records "temp" as primary

    with pytest.raises((IndexError, RuntimeError)):
        await store.set("temp/c/0", buf(b"malformed"))
