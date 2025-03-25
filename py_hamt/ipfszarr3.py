from collections.abc import AsyncIterator, Iterable
import zarr.abc.store
import zarr.core.buffer
from zarr.core.common import BytesLike

from py_hamt.hamt import HAMT


class IPFSZarr3(zarr.abc.store.Store):
    """
    While Zarr v2 can use a generic key-value map (MutableMapping) that HAMT already conforms to, Zarr v3s require storage classes to conform to a new abstract class. IPFSZarr3 just wraps over a HAMT to provide this compatibility.

    An example of how to write and read a zarr, using xarray, is provided below.
    # Write and get CID
    ```python
    import xarray as xr
    from py_hamt import IPFSStore, HAMT, IPFSZarr3

    ds = ... # some xarray Dataset
    ipfszarr3 = IPFSZarr3(HAMT(store=IPFSStore()))
    xr.to_zarr(store=ipfszarr3)
    print(ipfszarr3.hamt.root_node_id) # The CID of the root, which is used for reading
    ```

    # Read from CID
    ```python
    import xarray as xr
    from multiformats import CID
    from py_hamt import IPFSStore, HAMT, IPFSZarr3

    cid = CID.decode("...") # the CID for the HAMT root
    ipfszarr3 = IPFSZarr3(HAMT(store=IPFSStore(), root_node_id=cid), read_only=True)
    ds = xr.open_zarr(store=ipfszarr3)
    print(ds)
    ```
    """

    hamt: HAMT
    """The internal HAMT. Safe to read the CID from, if done doing operations."""

    def __init__(self, hamt: HAMT, read_only: bool = False) -> None:
        super().__init__(read_only=read_only)
        self.hamt = hamt
        if read_only:
            self.hamt.make_read_only()
        else:
            self.hamt.enable_write()

    @property
    def read_only(self) -> bool:
        return self.hamt.read_only

    def __eq__(self, val: object) -> bool:
        if not isinstance(val, IPFSZarr3):
            return False
        return self.hamt.root_node_id == val.hamt.root_node_id

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: zarr.abc.store.ByteRequest | None = None,
    ) -> zarr.core.buffer.Buffer | None:
        if key not in self.hamt:
            return
        # We know this value will always be bytes since we only store bytes in the HAMT
        val: bytes = self.hamt[key]  # type: ignore
        return prototype.buffer.from_bytes(val)

        # Hypothetical code for supporting partial writes, but there is not much point since IPFS itself doesn't support partial write and reads
        # Untested! If for some reason this is being uncommented and then used in the future, this needs to be tested
        # subset: bytes
        # match byte_range:
        #     case None:
        #         subset = val
        #     case zarr.abc.store.RangeByteRequest:
        #         subset = val[byte_range.start : byte_range.end]
        #     case zarr.abc.store.OffsetByteRequest:
        #         subset = val[byte_range.offset :]
        #     case zarr.abc.store.SuffixByteRequest:
        #         subset = val[-byte_range.suffix :]

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[tuple[str, zarr.abc.store.ByteRequest | None]],
    ) -> list[zarr.core.buffer.Buffer | None]:
        raise NotImplementedError

    async def exists(self, key: str) -> bool:
        return key in self.hamt

    @property
    def supports_writes(self) -> bool:
        return not self.hamt.read_only

    @property
    def supports_partial_writes(self) -> bool:
        return False

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        self.hamt[key] = value.to_bytes()

    async def set_if_not_exists(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        if key not in self.hamt:
            await self.set(key, value)

    async def set_partial_values(
        self, key_start_values: Iterable[tuple[str, int, BytesLike]]
    ) -> None:
        raise NotImplementedError

    @property
    def supports_deletes(self) -> bool:
        return not self.hamt.read_only

    async def delete(self, key: str) -> None:
        del self.hamt[key]

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        for key in self.hamt:
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator:
        for key in self.hamt:
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator:
        for key in self.hamt:
            if key.startswith(prefix):
                suffix = key[len(prefix) :]
                first_slash = suffix.find("/")
                if first_slash == -1:
                    yield suffix
                else:
                    name = suffix[0:first_slash]
                    yield name
