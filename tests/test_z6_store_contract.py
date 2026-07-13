import json

import pytest
import zarr
from testing_utils import CIDInMemoryCAS
from zarr.abc.store import (
    OffsetByteRequest,
    RangeByteRequest,
    SuffixByteRequest,
)

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


def buf(data: bytes) -> zarr.core.buffer.Buffer:
    return PROTOTYPE.buffer.from_bytes(data)


async def populated_store() -> ShardedZarrStore:
    store = await ShardedZarrStore.open(
        cas=CIDInMemoryCAS(),
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=2,
    )
    await store.set("temp/zarr.json", buf(ARRAY_METADATA))
    await store.set("temp/c/0/0", buf(b"zero-zero"))
    await store.set("temp/c/1/1", buf(b"one-one"))
    return store


@pytest.mark.asyncio
async def test_get_metadata_byte_range_range() -> None:
    store = await populated_store()

    result = await store.get(
        "temp/zarr.json",
        PROTOTYPE,
        byte_range=RangeByteRequest(start=0, end=5),
    )

    assert result is not None
    assert result.to_bytes() == ARRAY_METADATA[0:5]


@pytest.mark.asyncio
async def test_get_metadata_byte_range_offset() -> None:
    store = await populated_store()

    result = await store.get(
        "temp/zarr.json",
        PROTOTYPE,
        byte_range=OffsetByteRequest(offset=5),
    )

    assert result is not None
    assert result.to_bytes() == ARRAY_METADATA[5:]


@pytest.mark.asyncio
async def test_get_metadata_byte_range_suffix() -> None:
    store = await populated_store()

    result = await store.get(
        "temp/zarr.json",
        PROTOTYPE,
        byte_range=SuffixByteRequest(suffix=5),
    )

    assert result is not None
    assert result.to_bytes() == ARRAY_METADATA[-5:]


@pytest.mark.asyncio
async def test_list_yields_chunk_keys() -> None:
    store = await populated_store()

    keys = {key async for key in store.list()}

    assert keys == {"temp/zarr.json", "temp/c/0/0", "temp/c/1/1"}


@pytest.mark.asyncio
async def test_list_prefix_yields_chunk_keys() -> None:
    store = await populated_store()

    keys = {key async for key in store.list_prefix("temp/c/")}

    assert keys == {"temp/c/0/0", "temp/c/1/1"}


@pytest.mark.asyncio
async def test_list_dir_named_prefix() -> None:
    store = await populated_store()

    children = {child async for child in store.list_dir("temp")}

    assert children == {"zarr.json", "c"}


@pytest.mark.asyncio
async def test_root_level_chunk_key_no_silent_clobber() -> None:
    store = await populated_store()

    try:
        await store.set("c/0/0", buf(b"ROOT-CHUNK"))
    except Exception:
        # Rejecting root-level chunks explicitly is an acceptable contract.
        pass
    else:
        named = await store.get("temp/c/0/0", PROTOTYPE)
        assert named is not None and named.to_bytes() == b"zero-zero", (
            "root-level chunk write silently clobbered named array chunk"
        )
        rooted = await store.get("c/0/0", PROTOTYPE)
        assert rooted is not None and rooted.to_bytes() == b"ROOT-CHUNK"
