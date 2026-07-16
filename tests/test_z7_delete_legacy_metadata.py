import json

import dag_cbor
import pytest
import zarr
from multiformats import CID
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
CHUNK_KEY = "myarr/c/0/0"
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
        "codecs": [{"name": "bytes"}],
    },
    separators=(",", ":"),
).encode()


def buf(data: bytes) -> zarr.core.buffer.Buffer:
    return PROTOTYPE.buffer.from_bytes(data)


async def _base_store() -> tuple[CIDInMemoryCAS, str]:
    """A flushed v2 store with one registered array and one shard chunk."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=4,
    )
    await store.set("myarr/zarr.json", buf(ARRAY_METADATA))
    await store.set(CHUNK_KEY, buf(b"current"))
    root_cid = await store.flush()
    return cas, root_cid


async def _decoded_root(cas: CIDInMemoryCAS, root_cid: str) -> dict:
    root_obj = dag_cbor.decode(await cas.load(root_cid))
    assert isinstance(root_obj, dict)
    return root_obj


async def dual_representation_store() -> tuple[CIDInMemoryCAS, ShardedZarrStore]:
    """A writable store whose chunk key exists BOTH as a populated shard slot
    and as a stale legacy entry in root metadata, mimicking a legacy root that
    was later overwritten by an older py-hamt without metadata cleanup."""
    cas, root_cid = await _base_store()

    root_obj = await _decoded_root(cas, root_cid)
    stale_cid = await cas.save(b"stale-legacy", codec="raw")
    metadata = root_obj["metadata"]
    assert isinstance(metadata, dict)
    metadata[CHUNK_KEY] = stale_cid
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=str(legacy_root_cid),
    )
    # Preconditions: both representations exist, and the shard slot wins reads.
    assert CHUNK_KEY in store._root_obj["metadata"]
    current = await store.get(CHUNK_KEY, PROTOTYPE)
    assert current is not None
    assert current.to_bytes() == b"current"
    return cas, store


async def shard_empty_legacy_store() -> tuple[CIDInMemoryCAS, ShardedZarrStore]:
    """A writable store whose chunk key exists ONLY as a legacy metadata entry
    (shard slot empty), the layout served by the legacy fallback."""
    cas, root_cid = await _base_store()

    root_obj = await _decoded_root(cas, root_cid)
    arrays = root_obj["arrays"]
    assert isinstance(arrays, dict)
    shard_cids = arrays["myarr"]["shard_cids"]
    shard_cid = shard_cids[0]
    assert isinstance(shard_cid, CID)
    shard_entries = dag_cbor.decode(await cas.load(shard_cid))
    assert isinstance(shard_entries, list)
    original_chunk_cid = shard_entries[0]
    assert isinstance(original_chunk_cid, CID)

    shard_cids[0] = None
    metadata = root_obj["metadata"]
    assert isinstance(metadata, dict)
    metadata[CHUNK_KEY] = original_chunk_cid
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=str(legacy_root_cid),
    )
    original = await store.get(CHUNK_KEY, PROTOTYPE)
    assert original is not None
    assert original.to_bytes() == b"current"
    return cas, store


@pytest.mark.asyncio
async def test_overwrite_removes_legacy_metadata_entry() -> None:
    cas, store = await shard_empty_legacy_store()

    await store.set(CHUNK_KEY, buf(b"overwritten"))

    new_root_cid = await store.flush()
    root_obj = await _decoded_root(cas, new_root_cid)
    metadata = root_obj["metadata"]
    assert isinstance(metadata, dict)
    assert CHUNK_KEY not in metadata, (
        "overwriting a legacy chunk must not pin the superseded CID in metadata"
    )
    overwritten = await store.get(CHUNK_KEY, PROTOTYPE)
    assert overwritten is not None
    assert overwritten.to_bytes() == b"overwritten"


@pytest.mark.asyncio
async def test_delete_removes_legacy_metadata_when_shard_slot_exists() -> None:
    _, store = await dual_representation_store()

    await store.delete(CHUNK_KEY)

    deleted = await store.get(CHUNK_KEY, PROTOTYPE)
    assert deleted is None, deleted.to_bytes() if deleted is not None else None


@pytest.mark.asyncio
async def test_delete_persists_legacy_metadata_removal_after_reopen() -> None:
    cas, store = await dual_representation_store()

    await store.delete(CHUNK_KEY)
    new_root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=new_root_cid,
    )

    deleted = await reopened.get(CHUNK_KEY, PROTOTYPE)
    assert deleted is None, deleted.to_bytes() if deleted is not None else None
