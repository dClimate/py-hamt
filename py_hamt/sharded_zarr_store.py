import asyncio
import math
from collections.abc import AsyncIterator, Iterable
from typing import Optional, cast, Dict, List, Set, Tuple

import dag_cbor
import zarr.abc.store
import zarr.core.buffer
from zarr.core.common import BytesLike
from .store import ContentAddressedStore


class ShardedZarrStore(zarr.abc.store.Store):
    """
    Implements the Zarr Store API using a sharded layout for chunk CIDs.

    This store divides the flat index of chunk CIDs into multiple smaller "shards".
    Each shard is a contiguous block of bytes containing CIDs for a subset of chunks.
    This can improve performance for certain access patterns and reduce the size
    of individual index objects stored in the CAS.

    The store's root object contains:
    1.  A dictionary mapping metadata keys (like 'zarr.json') to their CIDs.
    2.  A list of CIDs, where each CID points to a shard of the chunk index.
    3.  Sharding configuration details (e.g., chunks_per_shard).

    Accessing a chunk involves:
    1.  Loading the root object (if not cached).
    2.  Determining the shard index and the offset of the chunk's CID within that shard.
    3.  Fetching the specific shard's CID from the root object.
    4.  Fetching the chunk's CID using a byte-range request on the identified shard.
    5.  Fetching the actual chunk data using the retrieved chunk CID.
    """

    def __init__(
        self,
        cas: ContentAddressedStore,
        read_only: bool,
        root_cid: Optional[str],
    ):
        """Use the async `open()` classmethod to instantiate this class."""
        super().__init__(read_only=read_only)
        self.cas = cas
        self._root_cid = root_cid
        self._root_obj: Optional[dict] = None

        self._shard_data_cache: Dict[int, bytearray] = {} # shard_index -> shard_byte_data
        self._dirty_shards: Set[int] = set() # Set of shard_indices that need flushing
        self._pending_shard_loads: Dict[int, asyncio.Task] = {} # shard_index -> Task loading the full shard

        self._cid_len: Optional[int] = None
        self._array_shape: Optional[Tuple[int, ...]] = None
        self._chunk_shape: Optional[Tuple[int, ...]] = None
        self._chunks_per_dim: Optional[Tuple[int, ...]] = None # Number of chunks in each dimension
        self._chunks_per_shard: Optional[int] = None # How many chunk CIDs per shard
        self._num_shards: Optional[int] = None # Total number of shards
        self._total_chunks: Optional[int] = None # Total number of chunks in the array

        self._dirty_root = False # Indicates if the root object itself (metadata or shard_cids list) changed

    @classmethod
    async def open(
        cls,
        cas: ContentAddressedStore,
        read_only: bool,
        root_cid: Optional[str] = None,
        *,
        array_shape: Optional[Tuple[int, ...]] = None,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        chunks_per_shard: Optional[int] = None,
        cid_len: int = 59,  # Default for base32 v1 CIDs like bafy... (e.g., bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi)
    ) -> "ShardedZarrStore":
        """
        Asynchronously opens an existing ShardedZarrStore or initializes a new one.
        """
        store = cls(cas, read_only, root_cid)
        if root_cid:
            await store._load_root_from_cid()
        elif not read_only:
            if not all([array_shape, chunk_shape, chunks_per_shard is not None]):
                raise ValueError(
                    "array_shape, chunk_shape, and chunks_per_shard must be provided for a new store."
                )
            if not isinstance(chunks_per_shard, int) or chunks_per_shard <= 0:
                raise ValueError("chunks_per_shard must be a positive integer.")
            store._initialize_new_root(array_shape, chunk_shape, chunks_per_shard, cid_len)
        else:
            raise ValueError("root_cid must be provided for a read-only store.")
        return store

    def _initialize_new_root(
        self,
        array_shape: Tuple[int, ...],
        chunk_shape: Tuple[int, ...],
        chunks_per_shard: int,
        cid_len: int,
    ):
        self._array_shape = array_shape
        self._chunk_shape = chunk_shape
        self._cid_len = cid_len
        self._chunks_per_shard = chunks_per_shard
        
        if not all(cs > 0 for cs in chunk_shape):
            raise ValueError("All chunk_shape dimensions must be positive.")
        if not all(asarray_s >= 0 for asarray_s in array_shape): # array_shape dims can be 0
             raise ValueError("All array_shape dimensions must be non-negative.")


        self._chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0 for a, c in zip(array_shape, chunk_shape)
        )
        self._total_chunks = math.prod(self._chunks_per_dim)
        
        if self._total_chunks == 0:
            self._num_shards = 0
        else:
            self._num_shards = math.ceil(self._total_chunks / self._chunks_per_shard)

        self._root_obj = {
            "manifest_version": "sharded_zarr_v1",
            "metadata": {}, # For .json
            "chunks": { # Information about the chunk index itself
                "array_shape": list(self._array_shape), # Original array shape
                "chunk_shape": list(self._chunk_shape), # Original chunk shape
                "cid_byte_length": self._cid_len,
                "sharding_config": {
                    "chunks_per_shard": self._chunks_per_shard,
                },
                "shard_cids": [None] * self._num_shards, # List of CIDs for each shard
            },
        }
        self._dirty_root = True

    async def _load_root_from_cid(self):
        if not self._root_cid:
            raise RuntimeError("Cannot load root without a root_cid.")
        root_bytes = await self.cas.load(self._root_cid)
        self._root_obj = dag_cbor.decode(root_bytes)

        if self._root_obj.get("manifest_version") != "sharded_zarr_v1":
            raise ValueError(f"Incompatible manifest version: {self._root_obj.get('manifest_version')}. Expected 'sharded_zarr_v1'.")   
        chunk_info = self._root_obj["chunks"]
        self._array_shape = tuple(chunk_info["array_shape"])
        self._chunk_shape = tuple(chunk_info["chunk_shape"])
        self._cid_len = chunk_info["cid_byte_length"]
        sharding_cfg = chunk_info.get("sharding_config", {}) # Handle older formats if any planned
        self._chunks_per_shard = sharding_cfg["chunks_per_shard"]

        if not all(cs > 0 for cs in self._chunk_shape):
             raise ValueError("Loaded chunk_shape dimensions must be positive.")

        self._chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0 for a, c in zip(self._array_shape, self._chunk_shape)
        )
        self._total_chunks = math.prod(self._chunks_per_dim)
        
        expected_num_shards = 0
        if self._total_chunks > 0:
            expected_num_shards = math.ceil(self._total_chunks / self._chunks_per_shard)
        self._num_shards = expected_num_shards
        
        if len(chunk_info["shard_cids"]) != self._num_shards:
            raise ValueError(
                f"Inconsistent number of shards. Expected {self._num_shards} from shapes/config, "
                f"found {len(chunk_info['shard_cids'])} in root object's shard_cids list."
            )

    async def _fetch_and_cache_full_shard(self, shard_idx: int, shard_cid: str):
        """
        Fetches the full data for a shard and caches it.
        Manages removal from _pending_shard_loads.
        """
        try:
            shard_data_bytes = await self.cas.load(shard_cid) # Load full shard
            self._shard_data_cache[shard_idx] = bytearray(shard_data_bytes)
        except Exception as e:
            print(e)
            # Handle or log the exception appropriately
            print(f"Warning: Failed to cache full shard {shard_idx} (CID: {shard_cid}): {e}")
            # If it fails, subsequent requests might try again if it's still not in cache.
        finally:
            # Ensure the task is removed from pending list once done (success or failure)
            if shard_idx in self._pending_shard_loads:
                del self._pending_shard_loads[shard_idx]

    def _parse_chunk_key(self, key: str) -> Optional[Tuple[int, ...]]:
        # 1. Exclude .json files immediately (metadata)
        if key.endswith(".json"):
            return None
        excluded_array_prefixes = {"time", "lat", "lon", "latitude", "longitude"}
        
        chunk_marker = "/c/"
        marker_idx = key.rfind(chunk_marker) # Use rfind for robustness
        if marker_idx == -1:
            # Key does not contain "/c/", so it's not a chunk data key
            # in the expected format (e.g., could be .zattrs, .zgroup at various levels).
            return None

        # Extract the part of the key before "/c/", which might represent the array/group path
        # e.g., "temp" from "temp/c/0/0/0"
        # e.g., "group1/lat" from "group1/lat/c/0"
        # e.g., "" if key is "c/0/0/0" (root array)
        path_before_c = key[:marker_idx]

        # Determine the actual array name (the last component of the path before "/c/")
        actual_array_name = ""
        if path_before_c:
            actual_array_name = path_before_c.split('/')[-1]
        
        # 2. If the determined array name is in our exclusion list, return None.
        if actual_array_name in excluded_array_prefixes:
            return None

        # If we've reached here, the key is potentially for a "main" data variable
        # that this store instance is expected to handle via sharding.
        # Now, proceed with the original parsing logic using self._array_shape and
        # self._chunks_per_dim, which should be configured for this main data variable.

        if not self._array_shape or not self._chunks_per_dim:
            # This ShardedZarrStore instance is not properly initialized
            # with the shape/chunking info for the array it's supposed to manage.
            # This might also happen if a key like "some_other_main_array/c/0" is passed
            # but this store instance was configured for "temp".
            return None
        
        # The part after "/c/" contains the chunk coordinates
        coord_part = key[marker_idx + len(chunk_marker):]
        parts = coord_part.split('/')
        
        # Validate dimensionality:
        # The number of coordinate parts must match the dimensionality of the array
        # this store instance is configured for (self._chunks_per_dim).
        if len(parts) != len(self._chunks_per_dim):
            # This key's dimensionality does not match the store's configured array.
            # It's likely for a different array or a malformed key for the current array.
            return None
                
        try:
            coords = tuple(map(int, parts))
            # Validate coordinates against the chunk grid of the store's configured array
            for i, c_coord in enumerate(coords):
                if not (0 <= c_coord < self._chunks_per_dim[i]):
                    return None # Coordinate out of bounds for this array's chunk grid
            return coords
        except (ValueError, IndexError): # If int conversion fails or other issues
            return None

    def _get_linear_chunk_index(self, chunk_coords: Tuple[int, ...]) -> int:
        linear_index = 0
        multiplier = 1
        # Convert N-D chunk coordinates to a flat 1-D index (row-major order)
        for i in reversed(range(len(self._chunks_per_dim))):
            linear_index += chunk_coords[i] * multiplier
            multiplier *= self._chunks_per_dim[i]
        return linear_index

    def _get_shard_info(self, linear_chunk_index: int) -> Tuple[int, int]:
        if self._chunks_per_shard is None or self._chunks_per_shard <= 0:
            raise RuntimeError("Sharding not configured properly: _chunks_per_shard invalid.")
        if linear_chunk_index < 0:
            raise ValueError("Linear chunk index cannot be negative.")

        shard_idx = linear_chunk_index // self._chunks_per_shard
        index_in_shard = linear_chunk_index % self._chunks_per_shard
        return shard_idx, index_in_shard

    async def _load_or_initialize_shard_cache(self, shard_idx: int) -> bytearray:
        if shard_idx in self._shard_data_cache:
            return self._shard_data_cache[shard_idx]

        if shard_idx in self._pending_shard_loads:
            try:
                await self._pending_shard_loads[shard_idx]
                if shard_idx in self._shard_data_cache:
                    return self._shard_data_cache[shard_idx]
                else:
                    pass  # Fall through to normal loading
            except asyncio.CancelledError:
                if shard_idx in self._pending_shard_loads:
                    del self._pending_shard_loads[shard_idx]
                # Fall through to normal loading
            except Exception as e:
                print(f"Warning: Pending shard load for {shard_idx} failed: {e}. Attempting fresh load.")

        if self._root_obj is None:
            raise RuntimeError("Root object not loaded or initialized (_root_obj is None).")
        if not (0 <= shard_idx < self._num_shards if self._num_shards is not None else False):
             raise ValueError(f"Shard index {shard_idx} out of bounds for {self._num_shards} shards.")

        shard_cid = self._root_obj["chunks"]["shard_cids"][shard_idx]
        if shard_cid:
            shard_data_bytes = await self.cas.load(shard_cid)
            self._shard_data_cache[shard_idx] = bytearray(shard_data_bytes)
        else:
            if self._cid_len is None: # Should be set
                raise RuntimeError("Store not initialized: _cid_len is None for shard initialization.")
            # New shard or shard not yet written, initialize with zeros
            shard_size_bytes = self._chunks_per_shard * self._cid_len
            self._shard_data_cache[shard_idx] = bytearray(shard_size_bytes) # Filled with \x00
        return self._shard_data_cache[shard_idx]


    async def set_partial_values(
        self, key_start_values: Iterable[Tuple[str, int, BytesLike]]
    ) -> None:
        raise NotImplementedError("Partial writes are not supported by ShardedZarrStore.")

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[Tuple[str, zarr.abc.store.ByteRequest | None]],
    ) -> List[Optional[zarr.core.buffer.Buffer]]:
        tasks = [self.get(key, prototype, byte_range) for key, byte_range in key_ranges]
        results = await asyncio.gather(*tasks)
        return results # type: ignore

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardedZarrStore):
            return NotImplemented
        # For equality, root CID is primary. Config like chunks_per_shard is part of that root's identity.
        return self._root_cid == other._root_cid

    async def flush(self) -> str:
        if self.read_only:
            if self._root_cid is None: # Read-only store should have been opened with a root_cid
                 raise ValueError("Read-only store has no root CID to return. Was it opened correctly?")
            return self._root_cid

        if self._root_obj is None: # Should be initialized for a writable store
            raise RuntimeError("Store not initialized for writing: _root_obj is None.")

        # Save all dirty shards first, as their CIDs might need to go into the root object
        if self._dirty_shards:
            for shard_idx in sorted(list(self._dirty_shards)):
                if shard_idx not in self._shard_data_cache:
                    # This implies an internal logic error if a shard is dirty but not in cache
                    # However, could happen if cache was cleared externally; robust code might reload/reinit
                    print(f"Warning: Dirty shard {shard_idx} not found in cache. Skipping save for this shard.")
                    continue
                
                shard_data_bytes = bytes(self._shard_data_cache[shard_idx])
                
                # The CAS save method here should return a string CID.
                new_shard_cid = await self.cas.save(shard_data_bytes, codec="raw") # Shards are raw bytes of CIDs
                
                if self._root_obj["chunks"]["shard_cids"][shard_idx] != new_shard_cid:
                    self._root_obj["chunks"]["shard_cids"][shard_idx] = new_shard_cid
                    self._dirty_root = True # Root object changed because a shard_cid in its list changed

            self._dirty_shards.clear()

        if self._dirty_root:
            root_obj_bytes = dag_cbor.encode(self._root_obj)
            new_root_cid = await self.cas.save(root_obj_bytes, codec="dag-cbor")
            self._root_cid = str(new_root_cid) # Ensure it's string
            self._dirty_root = False
        
        if self._root_cid is None: # Should only happen if nothing was dirty AND it was a new store never flushed
             raise RuntimeError("Failed to obtain a root CID after flushing. Store might be empty or unchanged.")
        return self._root_cid


    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: Optional[zarr.abc.store.ByteRequest] = None,
    ) -> Optional[zarr.core.buffer.Buffer]:
        if self._root_obj is None:
            raise RuntimeError("Load the root object first before accessing data.")

        chunk_coords = self._parse_chunk_key(key)
        # Metadata request (e.g., ".json")
        if chunk_coords is None:
            metadata_cid = self._root_obj["metadata"].get(key)
            if metadata_cid is None:
                return None
            # byte_range is not typically applicable to metadata JSON objects themselves
            if byte_range is not None:
                    # Consider if this should be an error or ignored for metadata
                    print(f"Warning: byte_range requested for metadata key '{key}'. Ignoring range.")
            data = await self.cas.load(metadata_cid)
            return prototype.buffer.from_bytes(data)

        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        if not (0 <= shard_idx < len(self._root_obj["chunks"]["shard_cids"])):
                # This case implies linear_chunk_index was out of _total_chunks bounds or bad sharding logic
                return None 

        target_shard_cid = self._root_obj["chunks"]["shard_cids"][shard_idx]
        if target_shard_cid is None: # This shard has no data (all chunks within it are implicitly empty)
            return None

        offset_in_shard_bytes = index_in_shard * self._cid_len
        chunk_cid_bytes: Optional[bytes] = None

        if shard_idx in self._shard_data_cache:
            cached_shard_data = self._shard_data_cache[shard_idx]
            chunk_cid_bytes = bytes(cached_shard_data[offset_in_shard_bytes : offset_in_shard_bytes + self._cid_len])

        if chunk_cid_bytes is None: # Not in cache or cache was invalid
            chunk_cid_bytes = await self.cas.load(
                target_shard_cid, offset=offset_in_shard_bytes, length=self._cid_len
            )
            # After successfully fetching the specific CID bytes,
            # check if we should initiate a background load of the full shard.
            if shard_idx not in self._shard_data_cache and shard_idx not in self._pending_shard_loads:
                self._pending_shard_loads[shard_idx] = asyncio.create_task(
                    self._fetch_and_cache_full_shard(shard_idx, target_shard_cid)
                )

        if all(b == 0 for b in chunk_cid_bytes): # Check for null CID placeholder (e.g. \x00 * cid_len)
            return None  # Chunk doesn't exist or is considered empty

        # Decode CID (assuming ASCII, remove potential null padding)
        chunk_cid_str = chunk_cid_bytes.decode("ascii").rstrip('\x00')
        if not chunk_cid_str: # Empty string after rstrip if all were \x00 (already caught above)
                return None

        # Actual chunk data load using the retrieved chunk_cid_str
        req_offset = byte_range.start if byte_range else None
        req_length = None
        if byte_range:
            if byte_range.end is not None:
                if byte_range.start > byte_range.end: # Zarr allows start == stop for 0 length
                    raise ValueError(f"Byte range start ({byte_range.start}) cannot be greater than end ({byte_range.end})")
                req_length = byte_range.end - byte_range.start

        data = await self.cas.load(chunk_cid_str, offset=req_offset, length=req_length)
        return prototype.buffer.from_bytes(data)


    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        if self.read_only:
            raise ValueError("Cannot write to a read-only store.")
        if self._root_obj is None:
            raise RuntimeError("Store not initialized for writing (root_obj is None). Call open() first.")

        raw_chunk_data_bytes = value.to_bytes()
        # Save the actual chunk data to CAS first, to get its CID
        chunk_data_cid_obj = await self.cas.save(raw_chunk_data_bytes, codec="raw") # Chunks are typically raw bytes
        chunk_data_cid_str = str(chunk_data_cid_obj)
        await self.set_pointer(key, chunk_data_cid_str) # Store the CID in the index

    async def set_pointer(
        self, key: str, pointer: str
    ) -> None:
        # Ensure the CID (as ASCII bytes) fits in the allocated slot, padding with nulls
        chunk_data_cid_ascii_bytes = pointer.encode("ascii")
        if len(chunk_data_cid_ascii_bytes) > self._cid_len:
            raise ValueError(
                f"Encoded CID byte length ({len(chunk_data_cid_ascii_bytes)}) exceeds configured CID length ({self._cid_len}). CID: {pointer}"
            )
        padded_chunk_data_cid_bytes = chunk_data_cid_ascii_bytes.ljust(self._cid_len, b'\0')


        chunk_coords = self._parse_chunk_key(key)

        if chunk_coords is None:  # Metadata key (e.g., ".zarray")
            # For metadata, the 'value' is the metadata content itself, not a CID to it.
            # So, we store the metadata content, get its CID, and put *that* CID in root_obj.
            # This means the `value_cid_str` for metadata should be from `raw_chunk_data_bytes`.
            # This seems to align with FlatZarrStore, where `value_cid` is used for both.
            self._root_obj["metadata"][key] = pointer # Store the string CID of the metadata content
            self._dirty_root = True
            return

        # Chunk Data: `chunk_data_cid_str` is the CID of the data we just saved.
        # Now we need to store this CID string (padded) into the correct shard.
        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        # Ensure the target shard is loaded or initialized in cache
        target_shard_data_cache = await self._load_or_initialize_shard_cache(shard_idx)

        offset_in_shard_bytes = index_in_shard * self._cid_len
        
        # Check if the content is actually changing to avoid unnecessary dirtying (optional optimization)
        # current_bytes_in_shard = target_shard_data_cache[offset_in_shard_bytes : offset_in_shard_bytes + self._cid_len]
        # if current_bytes_in_shard == padded_chunk_data_cid_bytes:
        #     return # No change

        target_shard_data_cache[offset_in_shard_bytes : offset_in_shard_bytes + self._cid_len] = padded_chunk_data_cid_bytes
        self._dirty_shards.add(shard_idx)
        # If this write implies the shard CID in root_obj["chunks"]["shard_cids"] might change
        # (e.g., from None to an actual CID when the shard is first flushed),
        # then _dirty_root will be set by flush().


    async def exists(self, key: str) -> bool:
        if self._root_obj is None:
            raise RuntimeError("Root object not loaded. Call _load_root_from_cid() first.")

        chunk_coords = self._parse_chunk_key(key)
        if chunk_coords is None: # Metadata
            return key in self._root_obj.get("metadata", {})

        try:
            linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
            shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

            if not (self._root_obj and "chunks" in self._root_obj and \
                    0 <= shard_idx < len(self._root_obj["chunks"]["shard_cids"])):
                return False
            
            target_shard_cid = self._root_obj["chunks"]["shard_cids"][shard_idx]
            if target_shard_cid is None: # Shard itself doesn't exist
                return False

            offset_in_shard_bytes = index_in_shard * self._cid_len
            
            # Optimization: Check local shard cache first
            if shard_idx in self._shard_data_cache:
                 cached_shard_data = self._shard_data_cache[shard_idx]
                 # Ensure index_in_shard is valid for this cached data length
                 if offset_in_shard_bytes + self._cid_len <= len(cached_shard_data):
                    chunk_cid_bytes_from_cache = cached_shard_data[offset_in_shard_bytes : offset_in_shard_bytes + self._cid_len]
                    return not all(b == 0 for b in chunk_cid_bytes_from_cache)
                 # else: fall through to CAS load, cache might be out of sync or wrong size (should not happen with correct logic)

            # If not in cache or cache check was inconclusive, fetch from CAS
            chunk_cid_bytes_from_cas = await self.cas.load(
                target_shard_cid, offset=offset_in_shard_bytes, length=self._cid_len
            )
            return not all(b == 0 for b in chunk_cid_bytes_from_cas)
        except Exception: # Broad catch for issues like invalid coords, CAS errors during load etc.
            return False


    @property
    def supports_writes(self) -> bool:
        return not self.read_only

    @property
    def supports_partial_writes(self) -> bool:
        return False # Each chunk CID is written atomically into a shard slot

    @property
    def supports_deletes(self) -> bool:
        return not self.read_only 

    async def delete(self, key: str) -> None:
        if self.read_only:
            raise ValueError("Cannot delete from a read-only store.")
        if self._root_obj is None:
            raise RuntimeError("Store not initialized for deletion (root_obj is None).")

        chunk_coords = self._parse_chunk_key(key)
        if chunk_coords is None: # Metadata
            if key in self._root_obj.get("metadata", {}):
                del self._root_obj["metadata"][key]
                self._dirty_root = True
                return
            else:
                raise KeyError(f"Metadata key '{key}' not found for deletion.")
        

        # Chunk deletion: zero out the CID entry in the shard
        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        if not (0 <= shard_idx < (self._num_shards if self._num_shards is not None else 0)):
             raise KeyError(f"Chunk key '{key}' maps to an invalid shard index {shard_idx}.")

        # Ensure shard data is available for modification (loads from CAS if not in cache, or initializes if new)
        target_shard_data_cache = await self._load_or_initialize_shard_cache(shard_idx)
        
        offset_in_shard_bytes = index_in_shard * self._cid_len
        
        # Check if the entry is already zeroed (meaning it doesn't exist or already deleted)
        is_already_zero = True
        for i in range(self._cid_len):
            if offset_in_shard_bytes + i >= len(target_shard_data_cache) or \
               target_shard_data_cache[offset_in_shard_bytes + i] != 0:
                is_already_zero = False
                break
        
        if is_already_zero:
            raise KeyError(f"Chunk key '{key}' not found or already effectively deleted (CID slot is zeroed).")

        # Zero out the CID entry in the shard cache
        for i in range(self._cid_len):
            target_shard_data_cache[offset_in_shard_bytes + i] = 0
        
        self._dirty_shards.add(shard_idx)
        # If this shard becomes non-None in root_obj due to other writes, flush will handle it.
        # If this deletion makes a previously non-None shard all zeros, the shard itself might
        # eventually be elided if we had shard GC, but its CID remains in root_obj for now.

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        for key in self._root_obj.get("metadata", {}):
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # This simplified version only works for the root directory (prefix == "") of metadata.
        # It lists unique first components of metadata keys.
        if self._root_obj is None:
            raise RuntimeError("Root object not loaded. Call _load_root_from_cid() first.")

        seen: Set[str] = set()
        if prefix == "":
            async for key in self.list(): # Iterates metadata keys
                # e.g., if key is "group1/.zgroup" or "array1/.zarray", first_component is "group1" or "array1"
                # if key is ".zgroup", first_component is ".zgroup"
                first_component = key.split('/', 1)[0]
                if first_component not in seen:
                    seen.add(first_component)
                    yield first_component
        else:
            # For listing subdirectories like "group1/", we'd need to match keys starting with "group1/"
            # and then extract the next component. This is more involved.
            # Zarr spec: list_dir(path) should yield children (both objects and "directories")
            # For simplicity, and consistency with original FlatZarrStore, keeping this minimal.
            # To make it more compliant for prefix="foo/":
            normalized_prefix = prefix if prefix.endswith('/') else prefix + '/'
            async for key in self.list_prefix(normalized_prefix):
               remainder = key[len(normalized_prefix):]
               child = remainder.split('/', 1)[0]
               if child not in seen:
                   seen.add(child)
                   yield child