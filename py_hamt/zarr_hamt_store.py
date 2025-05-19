from collections.abc import AsyncIterator, Iterable
import zarr.abc.store
import zarr.core.buffer
from zarr.core.common import BytesLike

from py_hamt.hamt import HAMT


class ZarrHAMTStore(zarr.abc.store.Store):
    """
    Write and read Zarr v3s with a HAMT.

    #### A note about using the same `ZarrHAMTStore` for writing and then reading again
    If you write a Zarr to a HAMT, and then change it to read only mode, it's best to reinitialize a new ZarrHAMTStore with the proper read only setting. This is because this class, to err on the safe side, will not touch its super class's settings.
    """

    def __init__(self, hamt: HAMT, read_only: bool = False) -> None:
        """
        ### `hamt` and `read_only`
        You need to make sure the following two things are true:

        1. The HAMT is in the same read only mode that you are passing into the Zarr store. This means that `hamt.read_only == read_only`. This is because making a HAMT read only automatically requires async operations, but `__init__` cannot be async.
        2. The HAMT has `hamt.values_are_bytes == True`. This improves efficiency with Zarr v3 operations.

        ##### A note about the zarr chunk separator, "/" vs "."
        While Zarr v2 used periods by default, Zarr v3 uses forward slashes, and that is assumed here as well.
        """
        super().__init__(read_only=read_only)

        assert hamt.read_only == read_only
        assert hamt.values_are_bytes
        self.hamt = hamt
        """
        The internal HAMT.
        Once done with write operations, the hamt can be set to read only mode as usual to get your root node ID.
        """

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
            val: bytes = await self.hamt.get(key)  # type: ignore We know values received will always be bytes since we only store bytes in the HAMT
            return prototype.buffer.from_bytes(val)
        except KeyError:
            # Sometimes zarr queries keys that don't exist anymore, just return nothing on those cases
            return

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[tuple[str, zarr.abc.store.ByteRequest | None]],
    ) -> list[zarr.core.buffer.Buffer | None]:
        """@private"""
        raise NotImplementedError

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

    async def list_prefix(self, prefix: str) -> AsyncIterator:
        """@private"""
        async for key in self.hamt.keys():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator:
        """
        @private
        """
        async for key in self.hamt.keys():
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                first_slash = suffix.find("/")
                if first_slash == -1:
                    yield suffix
                else:
                    name = suffix[0:first_slash]
                    yield name
