from collections.abc import AsyncIterator, Iterable
from typing import cast
import zarr.abc.store
import zarr.core.buffer
from zarr.core.common import BytesLike
from typing import Optional
import asyncio

from py_hamt.hamt import HAMT

class ZarrHAMTStore(zarr.abc.store.Store):
    """
    Write and read Zarr v3s with a HAMT.

    #### A note about using the same `ZarrHAMTStore` for writing and then reading again
    If you write a Zarr to a HAMT, and then change it to read only mode, it's best to reinitialize a new ZarrHAMTStore with the proper read only setting. This is because this class, to err on the safe side, will not touch its super class's settings.

    #### Sample Code
    ```python
    # Write
    ds: xarray.Dataset = # ...
    cas: ContentAddressedStore = # ...
    hamt: HAMT = # ... make sure values_are_bytes is True and read_only is False to enable writes
    zhs = ZarrHAMTStore(hamt, False)
    xarray.to_zarr(store=zhs)
    await hamt.make_read_only()
    root_node_id = hamt.root_node_id
    print(root_node_id)

    # Read
    hamt_read = HAMT(cas=cas, root_node_id=root_node_id, read_only=True, values_are_bytes=True)
    zhs_read = ZarrHAMTStore(hamt, True)
    ds_read = xarray.open_zarr(store=zhs_read)
    print(ds_read)
    xarray.testing.assert_identical(ds, ds_read)
    ```
    """

    def __init__(self, hamt: HAMT, read_only: bool = False) -> None:
        """
        ### `hamt` and `read_only`
        You need to make sure the following two things are true:

        1. The HAMT is in the same read only mode that you are passing into the Zarr store. This means that `hamt.read_only == read_only`. This is because making a HAMT read only automatically requires async operations, but `__init__` cannot be async.
        2. The HAMT has `hamt.values_are_bytes == True`. This improves efficiency with Zarr v3 operations.

        ##### A note about the zarr chunk separator, "/" vs "."
        While Zarr v2 used periods by default, Zarr v3 uses forward slashes, and that is assumed here as well.

        #### Metadata Read Cache
        `ZarrHAMTStore` has an internal read cache for metadata. In practice metadata "zarr.json" files are very very frequently and duplicately requested compared to all other keys, and there are significant speed improvements gotten by implementing this cache. In terms of memory management, in practice this cache does not need an eviction step since "zarr.json" files are much smaller than the memory requirement of the zarr data.
        """
        super().__init__(read_only=read_only)

        assert hamt.read_only == read_only
        assert hamt.values_are_bytes
        self.hamt: HAMT = hamt
        """
        The internal HAMT.
        Once done with write operations, the hamt can be set to read only mode as usual to get your root node ID.
        """

        self.metadata_read_cache: dict[str, bytes] = {}
        """@private"""

    def _map_byte_request(
        self, byte_range: Optional[zarr.abc.store.ByteRequest]
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
        """Helper to map Zarr ByteRequest to offset, length, suffix."""
        offset: Optional[int] = None
        length: Optional[int] = None
        suffix: Optional[int] = None

        if byte_range:
            if isinstance(byte_range, zarr.abc.store.RangeByteRequest):
                offset = byte_range.start
                length = byte_range.end - byte_range.start
                if length < 0:
                    raise ValueError("End must be >= start for RangeByteRequest")
            elif isinstance(byte_range, zarr.abc.store.OffsetByteRequest):
                offset = byte_range.offset
            elif isinstance(byte_range, zarr.abc.store.SuffixByteRequest):
                suffix = byte_range.suffix
            else:
                raise TypeError(f"Unsupported ByteRequest type: {type(byte_range)}")

        return offset, length, suffix

    @property
    def read_only(self) -> bool:
        """@private"""
        return self.hamt.read_only

    def __eq__(self, other: object) -> bool:
        """@private"""
        if not isinstance(other, ZarrHAMTStore):
            return False
        return self.hamt.root_node_id == other.hamt.root_node_id

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: zarr.abc.store.ByteRequest | None = None,
    ) -> zarr.core.buffer.Buffer | None:
        """@private"""
        try:
            val: bytes
            # do len check to avoid indexing into overly short strings, 3.12 does not throw errors but we dont know if other versions will
            is_metadata: bool = (
                len(key) >= 9 and key[-9:] == "zarr.json"
            )  # if path ends with zarr.json

            if is_metadata and byte_range is None and key in self.metadata_read_cache:
                val = self.metadata_read_cache[key]
            else:
                offset, length, suffix = self._map_byte_request(byte_range)
                val = cast(
                    bytes,
                    await self.hamt.get(
                        key, offset=offset, length=length, suffix=suffix
                    ),
                )  # We know values received will always be bytes since we only store bytes in the HAMT
                if is_metadata and byte_range is None:
                    self.metadata_read_cache[key] = val

            return prototype.buffer.from_bytes(val)
        except KeyError:
            # Sometimes zarr queries keys that don't exist anymore, just return nothing on those cases
            return None
        except Exception as e:
            print(f"Error getting key '{key}' with range {byte_range}: {e}")
            raise

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[tuple[str, zarr.abc.store.ByteRequest | None]],
    ) -> list[zarr.core.buffer.Buffer | None]:
        """
        Retrieves multiple keys or byte ranges concurrently using asyncio.gather.
        """
        tasks = [self.get(key, prototype, byte_range) for key, byte_range in key_ranges]
        results = await asyncio.gather(
            *tasks, return_exceptions=False
        )  # Set return_exceptions=True for debugging
        return results

    async def exists(self, key: str) -> bool:
        """@private"""
        try:
            await self.hamt.get(key)
            return True
        except KeyError:
            return False

    @property
    def supports_writes(self) -> bool:
        """@private"""
        return not self.hamt.read_only

    @property
    def supports_partial_writes(self) -> bool:
        """@private"""
        return False

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        """@private"""
        if key in self.metadata_read_cache:
            self.metadata_read_cache[key] = value.to_bytes()
        await self.hamt.set(key, value.to_bytes())

    async def set_if_not_exists(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        """@private"""
        if not (await self.exists(key)):
            await self.set(key, value)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        """@private"""
        raise NotImplementedError

    @property
    def supports_deletes(self) -> bool:
        """@private"""
        return not self.hamt.read_only

    async def delete(self, key: str) -> None:
        """@private"""
        try:
            await self.hamt.delete(key)
            # In practice these lines never seem to be needed, creating and appending data are the only operations most zarrs actually undergo
            # if key in self.metadata_read_cache:
            #     del self.metadata_read_cache[key]
        # It's fine if the key was not in the HAMT
        # Sometimes zarr v3 calls deletes on keys that don't exist (or have already been deleted) for some reason, probably concurrency issues
        except KeyError:
            return

    @property
    def supports_listing(self) -> bool:
        """@private"""
        return True

    async def list(self) -> AsyncIterator[str]:
        """@private"""
        async for key in self.hamt.keys():
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        """@private"""
        async for key in self.hamt.keys():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        """
        @private
        """
        async for key in self.hamt.keys():
            if key.startswith(prefix):
                suffix: str = key[len(prefix) :]
                first_slash: int = suffix.find("/")
                if first_slash == -1:
                    yield suffix
                else:
                    name: str = suffix[0:first_slash]
                    yield name
