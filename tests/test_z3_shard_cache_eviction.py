"""Regression tests for shard eviction while a shard is in active use."""

import pytest
import zarr.core.buffer
from dag_cbor.ipld import IPLDKind
from multiformats import CID, multihash

from py_hamt import ContentAddressedStore, ShardedZarrStore


def _normalize_cid(identifier: IPLDKind) -> str:
    """Normalize a CID object or string to a stable base32 key."""
    cid = CID.decode(identifier) if isinstance(identifier, str) else identifier
    if not isinstance(cid, CID):
        raise TypeError(
            f"Expected a CID or CID string, got {type(identifier).__name__}"
        )
    return cid.set(base="base32").encode("base32")


class CIDInMemoryCAS(ContentAddressedStore):
    """Fully offline CAS whose ``save`` method returns real CIDs."""

    def __init__(self) -> None:
        self.store: dict[str, bytes] = {}
        self._hash_algorithm = multihash.get("blake3")

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        digest = self._hash_algorithm.digest(data, size=32)
        cid = CID("base32", 1, codec, digest)
        self.store[_normalize_cid(cid)] = data
        return cid

    async def load(
        self,
        identifier: IPLDKind,
        offset: int | None = None,
        length: int | None = None,
        suffix: int | None = None,
    ) -> bytes:
        data = self.store[_normalize_cid(identifier)]
        if offset is not None:
            if length is not None:
                return data[offset : offset + length]
            return data[offset:]
        if suffix is not None:
            return data[-suffix:]
        return data


@pytest.mark.asyncio
async def test_read_never_raises_under_cache_pressure() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(8,),
        chunk_shape=(1,),
        chunks_per_shard=1,
        max_cache_memory_bytes=1,
    )
    pointer = str(await cas.save(b"payload", codec="raw"))
    prototype = zarr.core.buffer.default_buffer_prototype()

    await store.set_pointer("temp/c/0", pointer)
    await store.set_pointer("temp/c/1", pointer)
    assert await store.get("temp/c/2", prototype) is None
    assert await store.get("temp/c/2", prototype) is None

    chunk_zero = await store.get("temp/c/0", prototype)
    assert chunk_zero is not None
    assert chunk_zero.to_bytes() == b"payload"
    assert await store.get("temp/c/3", prototype) is None


@pytest.mark.asyncio
async def test_no_lost_acknowledged_writes_under_cache_pressure() -> None:
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(16,),
        chunk_shape=(1,),
        chunks_per_shard=1,
        max_cache_memory_bytes=1,
    )
    prototype = zarr.core.buffer.default_buffer_prototype()
    expected_payloads = [f"payload-{index}".encode() for index in range(16)]

    for index, payload in enumerate(expected_payloads):
        pointer = str(await cas.save(payload, codec="raw"))
        key = f"temp/c/{index}"
        await store.set_pointer(key, pointer)
        written_chunk = await store.get(key, prototype)
        assert written_chunk is not None
        assert written_chunk.to_bytes() == payload

    root_cid = await store.flush()
    persisted_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
        max_cache_memory_bytes=1,
    )

    for index, expected_payload in enumerate(expected_payloads):
        persisted_chunk = await persisted_store.get(f"temp/c/{index}", prototype)
        assert persisted_chunk is not None
        assert persisted_chunk.to_bytes() == expected_payload
