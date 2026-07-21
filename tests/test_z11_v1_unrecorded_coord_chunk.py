import json

import pytest
import zarr
from hypothesis import given, settings
from hypothesis import strategies as st
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


def array_metadata(rank: int) -> bytes:
    return json.dumps(
        {
            "zarr_format": 3,
            "node_type": "array",
            "shape": [4] * rank,
            "data_type": "uint8",
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": [2] * rank},
            },
            "chunk_key_encoding": {
                "name": "default",
                "configuration": {"separator": "/"},
            },
            "fill_value": 0,
            "codecs": [{"name": "bytes"}],
            "attributes": {},
        },
        separators=(",", ":"),
    ).encode()


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


@pytest.mark.asyncio
@given(
    data=st.data(),
    coordinate_path=st.sampled_from((
        "lat",
        "lon",
        "time",
        "group/latitude",
        "forecast_reference_time",
    )),
    coordinate_rank=st.integers(min_value=1, max_value=3),
    coordinate_payload=st.binary(max_size=64),
    first_primary_payload=st.binary(max_size=64),
    second_primary_payload=st.binary(max_size=64),
    invalid_primary_rank=st.sampled_from((1, 3)),
)
@settings(max_examples=25, deadline=None)
async def test_v1_unrecorded_coordinate_chunk_ordering_property(
    data: st.DataObject,
    coordinate_path: str,
    coordinate_rank: int,
    coordinate_payload: bytes,
    first_primary_payload: bytes,
    second_primary_payload: bytes,
    invalid_primary_rank: int,
) -> None:
    """Coordinate chunks remain metadata-backed around primary shard writes."""
    cas = CIDInMemoryCAS()
    store = await new_v1_store(cas)
    coordinate_key = f"{coordinate_path}/c/" + "/".join(
        "0" for _ in range(coordinate_rank)
    )
    primary_values = {
        "temp/c/0/0": first_primary_payload,
        "temp/c/0/1": second_primary_payload,
    }
    actions = data.draw(
        st.permutations((
            "coordinate_metadata",
            "coordinate_chunk",
            "primary_metadata",
            "first_primary_chunk",
            "second_primary_chunk",
        )),
        label="write_order",
    )
    written_values: dict[str, bytes] = {}

    for action in actions:
        if action == "coordinate_metadata":
            await store.set(
                f"{coordinate_path}/zarr.json",
                buf(array_metadata(coordinate_rank)),
            )
        elif action == "coordinate_chunk":
            await store.set(coordinate_key, buf(coordinate_payload))
            written_values[coordinate_key] = coordinate_payload
        elif action == "primary_metadata":
            await store.set("temp/zarr.json", buf(array_metadata(2)))
        elif action == "first_primary_chunk":
            key = "temp/c/0/0"
            await store.set(key, buf(primary_values[key]))
            written_values[key] = primary_values[key]
        else:
            key = "temp/c/0/1"
            await store.set(key, buf(primary_values[key]))
            written_values[key] = primary_values[key]

        for key, expected in written_values.items():
            await assert_value(store, key, expected)

    invalid_primary_key = "temp/c/" + "/".join("0" for _ in range(invalid_primary_rank))
    with pytest.raises((IndexError, RuntimeError)):
        await store.set(invalid_primary_key, buf(b"malformed"))

    root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    await assert_value(reopened, coordinate_key, coordinate_payload)
    for key, expected in primary_values.items():
        await assert_value(reopened, key, expected)
