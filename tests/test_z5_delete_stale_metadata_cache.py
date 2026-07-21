import pytest
import zarr.core.buffer

from py_hamt import (
    HAMT,
    InMemoryCAS,
    SimpleEncryptedZarrHAMTStore,
    ZarrHAMTStore,
)

PROTOTYPE = zarr.core.buffer.default_buffer_prototype()
METADATA_KEY = "temp/zarr.json"
METADATA_PAYLOAD = b'{"shape": [10]}'


async def _assert_deleted_metadata_is_not_cached(store: ZarrHAMTStore) -> None:
    await store.set(METADATA_KEY, PROTOTYPE.buffer.from_bytes(METADATA_PAYLOAD))

    cached_value = await store.get(METADATA_KEY, PROTOTYPE)
    assert cached_value is not None
    assert cached_value.to_bytes() == METADATA_PAYLOAD

    await store.delete(METADATA_KEY)

    assert not await store.exists(METADATA_KEY)
    assert METADATA_KEY not in [key async for key in store.list()]

    deleted_value = await store.get(METADATA_KEY, PROTOTYPE)
    stale_value = None if deleted_value is None else deleted_value.to_bytes()
    assert deleted_value is None, (
        f"get() returned stale metadata after delete: {stale_value!r}"
    )


@pytest.mark.asyncio
async def test_plain_delete_purges_metadata_read_cache() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS(), values_are_bytes=True)
    store = ZarrHAMTStore(hamt, read_only=False)

    await _assert_deleted_metadata_is_not_cached(store)


@pytest.mark.asyncio
async def test_encrypted_delete_purges_metadata_read_cache() -> None:
    hamt = await HAMT.build(cas=InMemoryCAS(), values_are_bytes=True)
    store = SimpleEncryptedZarrHAMTStore(
        hamt,
        read_only=False,
        encryption_key=b"\x01" * 32,
        header=b"z5-delete-cache-test",
    )

    await _assert_deleted_metadata_is_not_cached(store)
