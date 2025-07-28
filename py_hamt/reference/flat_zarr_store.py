# Functional Flat Store for Zarr using IPFS. This is ultimately the "Best" way to do it
# but only if the metadata is small enough to fit in a single CBOR object.
# This is commented out because we are not using it. However I did not want to delete it
# because it is a good example of how to implement a Zarr store using a flat index

# import asyncio
# import math
# from collections.abc import AsyncIterator, Iterable
# from typing import Optional, cast

# import dag_cbor
# import zarr.abc.store
# import zarr.core.buffer
# from zarr.core.common import BytesLike

# from .store import ContentAddressedStore


# class FlatZarrStore(zarr.abc.store.Store):
#     """
#     Implements the Zarr Store API using a flat, predictable layout for chunk CIDs.

#     This store bypasses the need for a HAMT, offering direct, calculated
#     access to chunk data based on a mathematical formula. It is designed for
#     dense, multi-dimensional arrays where chunk locations are predictable.

#     The store is structured around a single root CBOR object. This root object contains:
#     1.  A dictionary mapping metadata keys (like 'zarr.json') to their CIDs.
#     2.  A single CID pointing to a large, contiguous block of bytes (the "flat index").
#         This flat index is a concatenation of the CIDs of all data chunks.

#     Accessing a chunk involves:
#     1.  Loading the root object (if not cached).
#     2.  Calculating the byte offset of the chunk's CID within the flat index.
#     3.  Fetching that specific CID using a byte-range request on the flat index.
#     4.  Fetching the actual chunk data using the retrieved CID.

#     ### Sample Code
#     ```python
#     import xarray as xr
#     import numpy as np
#     from py_hamt import KuboCAS, FlatZarrStore

#     # --- Write ---
#     ds = xr.Dataset(
#         {"data": (("t", "y", "x"), np.arange(24).reshape(2, 3, 4))},
#     )
#     cas = KuboCAS()
#     # When creating, must provide array shape and chunk shape
#     store_write = await FlatZarrStore.open(
#         cas,
#         read_only=False,
#         array_shape=ds.data.shape,
#         chunk_shape=ds.data.encoding['chunks']
#     )
#     ds.to_zarr(store=store_write, mode="w")
#     root_cid = await store_write.flush() # IMPORTANT: flush to get final root CID
#     print(f"Finished writing. Root CID: {root_cid}")


#     # --- Read ---
#     store_read = await FlatZarrStore.open(cas, read_only=True, root_cid=root_cid)
#     ds_read = xr.open_zarr(store=store_read)
#     print("Read back dataset:")
#     print(ds_read)
#     xr.testing.assert_identical(ds, ds_read)
#     ```
#     """

#     def __init__(
#         self, cas: ContentAddressedStore, read_only: bool, root_cid: Optional[str]
#     ):
#         """Use the async `open()` classmethod to instantiate this class."""
#         super().__init__(read_only=read_only)
#         self.cas = cas
#         self._root_cid = root_cid
#         self._root_obj: Optional[dict] = None
#         self._flat_index_cache: Optional[bytearray] = None
#         self._cid_len: Optional[int] = None
#         self._array_shape: Optional[tuple[int, ...]] = None
#         self._chunk_shape: Optional[tuple[int, ...]] = None
#         self._chunks_per_dim: Optional[tuple[int, ...]] = None
#         self._dirty = False

#     @classmethod
#     async def open(
#         cls,
#         cas: ContentAddressedStore,
#         read_only: bool,
#         root_cid: Optional[str] = None,
#         *,
#         array_shape: Optional[tuple[int, ...]] = None,
#         chunk_shape: Optional[tuple[int, ...]] = None,
#         cid_len: int = 59,  # Default for base32 v1 CIDs like bafy...
#     ) -> "FlatZarrStore":
#         """
#         Asynchronously opens an existing FlatZarrStore or initializes a new one.

#         Args:
#             cas: The Content Addressed Store (e.g., KuboCAS).
#             read_only: If True, the store is in read-only mode.
#             root_cid: The root CID of an existing store to open. Required for read_only.
#             array_shape: The full shape of the Zarr array. Required for a new writeable store.
#             chunk_shape: The shape of a single chunk. Required for a new writeable store.
#             cid_len: The expected byte length of a CID string.
#         """
#         store = cls(cas, read_only, root_cid)
#         if root_cid:
#             await store._load_root_from_cid()
#         elif not read_only:
#             if not all([array_shape, chunk_shape]):
#                 raise ValueError(
#                     "array_shape and chunk_shape must be provided for a new store."
#                 )
#             store._initialize_new_root(array_shape, chunk_shape, cid_len)
#         else:
#             raise ValueError("root_cid must be provided for a read-only store.")
#         return store

#     def _initialize_new_root(
#         self,
#         array_shape: tuple[int, ...],
#         chunk_shape: tuple[int, ...],
#         cid_len: int,
#     ):
#         self._array_shape = array_shape
#         self._chunk_shape = chunk_shape
#         self._cid_len = cid_len
#         self._chunks_per_dim = tuple(
#             math.ceil(a / c) for a, c in zip(array_shape, chunk_shape)
#         )
#         self._root_obj = {
#             "manifest_version": "flat_zarr_v1",
#             "metadata": {},
#             "chunks": {
#                 "cid": None,  # Will be filled on first flush
#                 "array_shape": list(self._array_shape),
#                 "chunk_shape": list(self._chunk_shape),
#                 "cid_byte_length": self._cid_len,
#             },
#         }
#         self._dirty = True

#     async def _load_root_from_cid(self):
#         if not self._root_cid:
#             raise ValueError("Cannot load root without a root_cid.")
#         root_bytes = await self.cas.load(self._root_cid)
#         self._root_obj = dag_cbor.decode(root_bytes)
#         chunk_info = self._root_obj.get("chunks", {})
#         self._array_shape = tuple(chunk_info["array_shape"])
#         self._chunk_shape = tuple(chunk_info["chunk_shape"])
#         self._cid_len = chunk_info["cid_byte_length"]
#         self._chunks_per_dim = tuple(
#             math.ceil(a / c) for a, c in zip(self._array_shape, self._chunk_shape)
#         )

#     def _parse_chunk_key(self, key: str) -> Optional[tuple[int, ...]]:
#         if not self._array_shape or not key.startswith("c/"):
#             return None
#         parts = key.split("/")
#         if len(parts) != len(self._array_shape) + 1:
#             return None
#         try:
#             return tuple(map(int, parts[1:]))
#         except (ValueError, IndexError):
#             return None

#     async def set_partial_values(
#         self, key_start_values: Iterable[tuple[str, int, BytesLike]]
#     ) -> None:
#         """@private"""
#         raise NotImplementedError("Partial writes are not supported by this store.")

#     async def get_partial_values(
#         self,
#         prototype: zarr.core.buffer.BufferPrototype,
#         key_ranges: Iterable[tuple[str, zarr.abc.store.ByteRequest | None]],
#     ) -> list[zarr.core.buffer.Buffer | None]:
#         """
#         Retrieves multiple keys or byte ranges concurrently.
#         """
#         tasks = [self.get(key, prototype, byte_range) for key, byte_range in key_ranges]
#         results = await asyncio.gather(*tasks)
#         return results

#     def __eq__(self, other: object) -> bool:
#         """@private"""
#         if not isinstance(other, FlatZarrStore):
#             return NotImplemented
#         return self._root_cid == other._root_cid

#     def _get_chunk_offset(self, chunk_coords: tuple[int, ...]) -> int:
#         linear_index = 0
#         multiplier = 1
#         for i in reversed(range(len(self._chunks_per_dim))):
#             linear_index += chunk_coords[i] * multiplier
#             multiplier *= self._chunks_per_dim[i]
#         return linear_index * self._cid_len

#     async def flush(self) -> str:
#         """
#         Writes all pending changes (metadata and chunk index) to the CAS
#         and returns the new root CID. This MUST be called after all writes are complete.
#         """
#         if self.read_only or not self._dirty:
#             return self._root_cid

#         if self._flat_index_cache is not None:
#             flat_index_cid_obj = await self.cas.save(
#                 bytes(self._flat_index_cache), codec="raw"
#             )
#             self._root_obj["chunks"]["cid"] = str(flat_index_cid_obj)

#         root_obj_bytes = dag_cbor.encode(self._root_obj)
#         new_root_cid_obj = await self.cas.save(root_obj_bytes, codec="dag-cbor")
#         self._root_cid = str(new_root_cid_obj)
#         self._dirty = False
#         return self._root_cid

#     async def get(
#         self,
#         key: str,
#         prototype: zarr.core.buffer.BufferPrototype,
#         byte_range: zarr.abc.store.ByteRequest | None = None,
#     ) -> zarr.core.buffer.Buffer | None:
#         """@private"""
#         if self._root_obj is None:
#             await self._load_root_from_cid()

#         chunk_coords = self._parse_chunk_key(key)
#         try:
#             # Metadata request
#             if chunk_coords is None:
#                 metadata_cid = self._root_obj["metadata"].get(key)
#                 if metadata_cid is None:
#                     return None
#                 data = await self.cas.load(metadata_cid)
#                 return prototype.buffer.from_bytes(data)

#             # Chunk data request
#             flat_index_cid = self._root_obj["chunks"]["cid"]
#             if flat_index_cid is None:
#                 return None

#             offset = self._get_chunk_offset(chunk_coords)
#             chunk_cid_bytes = await self.cas.load(
#                 flat_index_cid, offset=offset, length=self._cid_len
#             )

#             if all(b == 0 for b in chunk_cid_bytes):
#                 return None  # Chunk doesn't exist

#             chunk_cid = chunk_cid_bytes.decode("ascii")
#             data = await self.cas.load(chunk_cid)
#             return prototype.buffer.from_bytes(data)

#         except (KeyError, IndexError):
#             return None

#     async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
#         """@private"""
#         if self.read_only:
#             raise ValueError("Cannot write to a read-only store.")
#         if self._root_obj is None:
#             raise RuntimeError("Store not initialized for writing.")

#         self._dirty = True
#         raw_bytes = value.to_bytes()
#         value_cid_obj = await self.cas.save(raw_bytes, codec="raw")
#         value_cid = str(value_cid_obj)

#         if len(value_cid) != self._cid_len:
#             raise ValueError(
#                 f"Inconsistent CID length. Expected {self._cid_len}, got {len(value_cid)}"
#             )

#         chunk_coords = self._parse_chunk_key(key)

#         if chunk_coords is None:  # Metadata
#             self._root_obj["metadata"][key] = value_cid
#             return

#         # Chunk Data
#         if self._flat_index_cache is None:
#             num_chunks = math.prod(self._chunks_per_dim)
#             self._flat_index_cache = bytearray(num_chunks * self._cid_len)

#         offset = self._get_chunk_offset(chunk_coords)
#         self._flat_index_cache[offset : offset + self._cid_len] = value_cid.encode(
#             "ascii"
#         )

#     # --- Other required zarr.abc.store methods ---

#     async def exists(self, key: str) -> bool:
#         """@private"""
#         # A more efficient version might check for null bytes in the flat index
#         # but this is functionally correct.

#         # TODO: Optimize this check
#         return True


#         # return (await self.get(key, zarr.core.buffer.Buffer.prototype, None)) is not None

#     @property
#     def supports_writes(self) -> bool:
#         """@private"""
#         return not self.read_only

#     @property
#     def supports_partial_writes(self) -> bool:
#         """@private"""
#         return False  # Each chunk is an immutable object

#     @property
#     def supports_deletes(self) -> bool:
#         """@private"""
#         return not self.read_only

#     async def delete(self, key: str) -> None:
#         if self.read_only:
#             raise ValueError("Cannot delete from a read-only store.")
#         if self._root_obj is None:
#             await self._load_root_from_cid()
#         chunk_coords = self._parse_chunk_key(key)
#         if chunk_coords is None:
#             if key in self._root_obj["metadata"]:
#                 del self._root_obj["metadata"][key]
#                 self._dirty = True
#                 return
#             else:
#                 raise KeyError(f"Metadata key '{key}' not found.")
#         flat_index_cid = self._root_obj["chunks"]["cid"]
#         if self._flat_index_cache is None:
#             if not flat_index_cid:
#                 raise KeyError(f"Chunk key '{key}' not found in non-existent index.")
#             self._flat_index_cache = bytearray(await self.cas.load(flat_index_cid))
#         offset = self._get_chunk_offset(chunk_coords)
#         if all(b == 0 for b in self._flat_index_cache[offset : offset + self._cid_len]):
#             raise KeyError(f"Chunk key '{key}' not found.")
#         self._flat_index_cache[offset : offset + self._cid_len] = bytearray(self._cid_len)
#         self._dirty = True

#     @property
#     def supports_listing(self) -> bool:
#         """@private"""
#         return True

#     async def list(self) -> AsyncIterator[str]:
#         """@private"""
#         if self._root_obj is None:
#             await self._load_root_from_cid()
#         for key in self._root_obj["metadata"]:
#             yield key
#         # Note: Listing all chunk keys without reading the index is non-trivial.
#         # A full implementation might need an efficient way to iterate non-null chunks.
#         # This basic version only lists metadata.

#     async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
#         """@private"""
#         async for key in self.list():
#             if key.startswith(prefix):
#                 yield key

#     async def list_dir(self, prefix: str) -> AsyncIterator[str]:
#         """@private"""
#         # This simplified version only works for the root.
#         if prefix == "":
#             seen = set()
#             async for key in self.list():
#                 name = key.split('/')[0]
#                 if name not in seen:
#                     seen.add(name)
#                     yield name
