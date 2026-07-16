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


async def overwritten_legacy_store() -> tuple[CIDInMemoryCAS, ShardedZarrStore]:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=4,
    )
    await store.set("myarr/zarr.json", buf(ARRAY_METADATA))
    await store.set(CHUNK_KEY, buf(b"original"))
    root_cid = await store.flush()

    root_obj = dag_cbor.decode(await cas.load(root_cid))
    assert isinstance(root_obj, dict)
    arrays = root_obj["arrays"]
    assert isinstance(arrays, dict)
    array_manifest = arrays["myarr"]
    assert isinstance(array_manifest, dict)
    shard_cids = array_manifest["shard_cids"]
    assert isinstance(shard_cids, list)
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

    legacy_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        root_cid=str(legacy_root_cid),
    )
    original = await legacy_store.get(CHUNK_KEY, PROTOTYPE)
    assert original is not None
    assert original.to_bytes() == b"original"

    await legacy_store.set(CHUNK_KEY, buf(b"overwritten"))
    overwritten = await legacy_store.get(CHUNK_KEY, PROTOTYPE)
    assert overwritten is not None
    assert overwritten.to_bytes() == b"overwritten"
    return cas, legacy_store


@pytest.mark.asyncio
async def test_overwrite_removes_legacy_metadata_entry() -> None:
    cas, store = await overwritten_legacy_store()

    new_root_cid = await store.flush()
    root_obj = dag_cbor.decode(await cas.load(new_root_cid))
    assert isinstance(root_obj, dict)
    metadata = root_obj["metadata"]
    assert isinstance(metadata, dict)
    assert CHUNK_KEY not in metadata, (
        "overwriting a legacy chunk must not pin the superseded CID in metadata"
    )


@pytest.mark.asyncio
async def test_delete_removes_legacy_metadata_when_shard_slot_exists() -> None:
    _, store = await overwritten_legacy_store()

    await store.delete(CHUNK_KEY)

    deleted = await store.get(CHUNK_KEY, PROTOTYPE)
    assert deleted is None, deleted.to_bytes() if deleted is not None else None


@pytest.mark.asyncio
async def test_delete_persists_legacy_metadata_removal_after_reopen() -> None:
    cas, store = await overwritten_legacy_store()

    await store.delete(CHUNK_KEY)
    new_root_cid = await store.flush()
    reopened = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=new_root_cid,
    )

    deleted = await reopened.get(CHUNK_KEY, PROTOTYPE)
    assert deleted is None, deleted.to_bytes() if deleted is not None else None
