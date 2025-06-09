from collections.abc import AsyncIterator, Iterable
from typing import cast

import zarr.abc.store
import zarr.core.buffer
from zarr.core.common import BytesLike

from py_hamt.hamt import HAMT


class ZarrHAMTStore(zarr.abc.store.Store):
    """
    Write and read Zarr v3s with a HAMT.

    Read **or** write a Zarr-v3 store whose key/value pairs live inside a
    py-hamt mapping.

    Keys are stored verbatim (``"temp/c/0/0/0"`` → same string in HAMT) and
    the value is the raw byte payload produced by Zarr.  No additional
    framing, compression, or encryption is applied by this class. For a zarr encryption example
    see where metadata is available use the method in https://github.com/dClimate/jupyter-notebooks/blob/main/notebooks/202b%20-%20Encryption%20Example%20(Encryption%20with%20Zarr%20Codecs).ipynb
    For a fully encrypted zarr store, where metadata is not available, please see
    :class:`SimpleEncryptedZarrHAMTStore` but we do not recommend using it.

    #### A note about using the same `ZarrHAMTStore` for writing and then reading again
    If you write a Zarr to a HAMT, and then change it to read only mode, it's best to reinitialize a new ZarrHAMTStore with the proper read only setting. This is because this class, to err on the safe side, will not touch its super class's settings.

    #### Sample Code
    ```python
    # --- Write ---
    ds: xarray.Dataset = # ...
    cas: ContentAddressedStore = # ...
    hamt: HAMT = # ... make sure values_are_bytes is True and read_only is False to enable writes
    hamt = await HAMT.build(cas, values_are_bytes=True)     # write-enabled
    zhs  = ZarrHAMTStore(hamt, read_only=False)
    ds.to_zarr(store=zhs, mode="w", zarr_format=3)
    await hamt.make_read_only() # flush + freeze
    root_node_id = hamt.root_node_id
    print(root_node_id)

     # --- read ---
    hamt_ro = await HAMT.build(
        cas, root_node_id=root_cid, read_only=True, values_are_bytes=True
    )
    zhs_ro  = ZarrHAMTStore(hamt_ro, read_only=True)
    ds_ro = xarray.open_zarr(store=zhs_ro)


    print(ds_ro)
    xarray.testing.assert_identical(ds, ds_ro)
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

            if is_metadata and key in self.metadata_read_cache:
                val = self.metadata_read_cache[key]
            else:
                val = cast(
                    bytes, await self.hamt.get(key)
                )  # We know values received will always be bytes since we only store bytes in the HAMT
                if is_metadata:
                    self.metadata_read_cache[key] = val

            return prototype.buffer.from_bytes(val)
        except KeyError:
            # Sometimes zarr queries keys that don't exist anymore, just return nothing on those cases
            return None

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
        List *immediate* children that live directly under **prefix**.

        This is similar to :py:meth:`list_prefix` but collapses everything
        below the first ``"/"`` after *prefix*.  Each child name is yielded
        **exactly once** in the order of first appearance while scanning the
        HAMT keys.

        Parameters
        ----------
        prefix : str
            Logical directory path.  *Must* end with ``"/"`` for the result to
            make sense (e.g. ``"a/b/"``).

        Yields
        ------
        str
            The name of each direct child (file or sub-directory) of *prefix*.

        Examples
        --------
        With keys ::

            a/b/c/d
            a/b/c/e
            a/b/f
            a/b/g/h/i

        ``await list_dir("a/b/")`` produces ::

            c
            f
            g

        Notes
        -----
        • Internally uses a :class:`set` to deduplicate names; memory grows
            with the number of *unique* children, not the total number of keys.
        • Order is **not** sorted; it reflects the first encounter while
            iterating over :py:meth:`HAMT.keys`.
        """
        seen_names: set[str] = set()
        async for key in self.hamt.keys():
            if key.startswith(prefix):
                suffix: str = key[len(prefix) :]
                first_slash: int = suffix.find("/")
                if first_slash == -1:
                    if suffix not in seen_names:
                        seen_names.add(suffix)
                        yield suffix
                else:
                    name: str = suffix[0:first_slash]
                    if name not in seen_names:
                        seen_names.add(name)
                        yield name
