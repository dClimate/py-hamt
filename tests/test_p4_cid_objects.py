import pytest
import zarr.core.buffer
from multiformats import CID
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
METADATA = b'{"shape":[2,2],"node_type":"array"}'


async def _new_store(cas: CIDInMemoryCAS) -> ShardedZarrStore:
    return await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 2),
        chunk_shape=(1, 1),
        chunks_per_shard=2,
    )


@pytest.mark.asyncio
async def test_warm_gets_and_sets_keep_cids_as_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cas = CIDInMemoryCAS()
    store = await _new_store(cas)
    initial_values = {
        "zarr.json": METADATA,
        "c/0/0": b"chunk-00",
        "c/0/1": b"chunk-01",
        "c/1/0": b"chunk-10",
    }
    for key, value in initial_values.items():
        await store.set(key, PROTOTYPE.buffer.from_bytes(value))

    warm_keys = ("c/0/0", "c/0/1", "c/1/0")
    for key in warm_keys:
        assert await store.get(key, PROTOTYPE) is not None

    counters = {"str": 0, "decode": 0}
    original_decode = CID.decode
    original_str = CID.__str__

    def counted_decode(value: str | bytes) -> CID:
        counters["decode"] += 1
        return original_decode(value)

    def counted_str(cid: CID) -> str:
        counters["str"] += 1
        return original_str(cid)

    monkeypatch.setattr(CID, "decode", staticmethod(counted_decode))
    monkeypatch.setattr(CID, "__str__", counted_str)

    counters.update(str=0, decode=0)
    for key in warm_keys:
        result = await store.get(key, PROTOTYPE)
        assert result is not None
        assert result.to_bytes() == initial_values[key]

    warm_get_counters = counters.copy()
    counters.update(str=0, decode=0)
    for key, value in {
        "c/1/0": b"updated-chunk-10",
        "c/1/1": b"chunk-11",
    }.items():
        await store.set(key, PROTOTYPE.buffer.from_bytes(value))
    set_counters = counters.copy()

    assert warm_get_counters == {"str": 0, "decode": 0}, (
        "warm gets must pass CID objects directly; observed "
        f"{warm_get_counters['str']} CID.__str__ calls and "
        f"{warm_get_counters['decode']} CID.decode calls"
    )
    assert set_counters["decode"] == 0, (
        "sets must keep saved CIDs as objects instead of decoding their strings; "
        f"observed {set_counters['decode']} CID.decode calls"
    )


@pytest.mark.asyncio
async def test_cid_object_optimization_preserves_root_and_values() -> None:
    cas = CIDInMemoryCAS()
    store = await _new_store(cas)
    values = {
        "zarr.json": METADATA,
        "c/0/0": b"golden-00",
        "c/0/1": b"golden-01",
        "c/1/0": b"golden-10",
        "c/1/1": b"golden-11",
    }
    for key, value in values.items():
        await store.set(key, PROTOTYPE.buffer.from_bytes(value))

    root_cid = await store.flush()

    # The golden root includes the versioned per-array manifest layout.
    assert root_cid == "bafyr4ibiddqekdsvo4oraxn5lfhqoq7cw477gcadnhymxu6vqmjcrp55jm"
    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    for key, expected in values.items():
        result = await read_store.get(key, PROTOTYPE)
        assert result is not None
        assert result.to_bytes() == expected
