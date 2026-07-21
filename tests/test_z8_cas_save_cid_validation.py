import json

import pytest
import zarr
from dag_cbor.ipld import IPLDKind
from testing_utils import CIDInMemoryCAS

from py_hamt import ContentAddressedStore, ShardedZarrStore
from py_hamt.sharded_zarr_store import SHARDED_ZARR_V2

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


class ToggleRawBytesCAS(CIDInMemoryCAS):
    def __init__(self) -> None:
        super().__init__()
        self.return_raw_bytes = False

    async def save(
        self, data: bytes, codec: ContentAddressedStore.CodecInput
    ) -> IPLDKind:  # type: ignore[override]
        cid = await super().save(data, codec)
        if self.return_raw_bytes and codec == "raw":
            return self._hash_algorithm.digest(data, size=32)
        return cid


@pytest.mark.asyncio
async def test_set_rejects_non_cid_from_cas_save() -> None:
    cas = ToggleRawBytesCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=2,
        manifest_version=SHARDED_ZARR_V2,
    )
    await store.set("temp/zarr.json", PROTOTYPE.buffer.from_bytes(ARRAY_METADATA))

    cas.return_raw_bytes = True

    with pytest.raises((TypeError, RuntimeError), match="CID"):
        await store.set("temp/c/0/0", PROTOTYPE.buffer.from_bytes(b"chunk data"))
