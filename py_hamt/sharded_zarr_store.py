import asyncio
import math
from collections.abc import AsyncIterator, Iterable
from typing import Coroutine, Dict, List, Optional, Set, Tuple
import json
import itertools
import logging 

import dag_cbor
from multiformats.cid import CID
import zarr.abc.store
import zarr.core.buffer
from zarr.core.common import BytesLike

from .store_httpx import ContentAddressedStore


class ShardedZarrStore(zarr.abc.store.Store):
    """
    Implements the Zarr Store API using a sharded layout for chunk CIDs.

    # CHANGED: Docstring updated to reflect DAG-CBOR format.
    This store divides the flat index of chunk CIDs into multiple "shards".
    Each shard is a DAG-CBOR array where each element is either a CID link
    to a chunk or a null value if the chunk is empty. This structure allows
    for efficient traversal by IPLD-aware systems.

    The store's root object contains:
    1.  A dictionary mapping metadata keys (like 'zarr.json') to their CIDs.
    2.  A list of CIDs, where each CID points to a shard object.
    3.  Sharding configuration details (e.g., chunks_per_shard).
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

        self._resize_lock = asyncio.Lock()
        
        # An event to signal when a resize is in-progress.
        # It starts in the "set" state, allowing all operations to proceed.
        self._resize_complete = asyncio.Event()
        self._resize_complete.set()

        self._shard_data_cache: Dict[
            int, list[Optional[CID]]
        ] = {}
        self._dirty_shards: Set[int] = set()
        self._pending_shard_loads: Dict[
            int, asyncio.Task
        ] = {}

        self._array_shape: Optional[Tuple[int, ...]] = None
        self._chunk_shape: Optional[Tuple[int, ...]] = None
        self._chunks_per_dim: Optional[Tuple[int, ...]] = None
        self._chunks_per_shard: Optional[int] = None
        self._num_shards: Optional[int] = None
        self._total_chunks: Optional[int] = None

        self._dirty_root = False


    def _update_geometry(self):
        """Calculates derived geometric properties from the base shapes."""
        if self._array_shape is None or self._chunk_shape is None or self._chunks_per_shard is None:
            raise RuntimeError("Base shape information is not set.")

        if not all(cs > 0 for cs in self._chunk_shape):
            raise ValueError("All chunk_shape dimensions must be positive.")
        if not all(s >= 0 for s in self._array_shape):
            raise ValueError("All array_shape dimensions must be non-negative.")

        self._chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0 for a, c in zip(self._array_shape, self._chunk_shape)
        )
        self._total_chunks = math.prod(self._chunks_per_dim)

        if self._total_chunks == 0:
            self._num_shards = 0
        else:
            self._num_shards = math.ceil(self._total_chunks / self._chunks_per_shard)

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
        # REMOVED: cid_len is no longer needed.
    ) -> "ShardedZarrStore":
        """
        Asynchronously opens an existing ShardedZarrStore or initializes a new one.
        """
        store = cls(cas, read_only, root_cid)
        if root_cid:
            await store._load_root_from_cid()
        elif not read_only:
            if array_shape is None or chunk_shape is None:
                raise ValueError(
                    "array_shape and chunk_shape must be provided for a new store."
                )

            if not isinstance(chunks_per_shard, int) or chunks_per_shard <= 0:
                raise ValueError("chunks_per_shard must be a positive integer.")

            store._initialize_new_root(
                array_shape, chunk_shape, chunks_per_shard
            )
        else:
            raise ValueError("root_cid must be provided for a read-only store.")
        return store

    def _initialize_new_root(
        self,
        array_shape: Tuple[int, ...],
        chunk_shape: Tuple[int, ...],
        chunks_per_shard: int,
    ):
        self._array_shape = array_shape
        self._chunk_shape = chunk_shape
        self._chunks_per_shard = chunks_per_shard

        self._update_geometry()

        self._root_obj = {
            "manifest_version": "sharded_zarr_v1", # CHANGED: Version reflects new format
            "metadata": {},
            "chunks": {
                "array_shape": list(self._array_shape),
                "chunk_shape": list(self._chunk_shape),
                # REMOVED: cid_byte_length is no longer relevant
                "sharding_config": {
                    "chunks_per_shard": self._chunks_per_shard,
                },
                "shard_cids": [None] * self._num_shards,
            },
        }
        self._dirty_root = True

    async def _load_root_from_cid(self):
        if not self._root_cid:
            raise RuntimeError("Cannot load root without a root_cid.")
        root_bytes = await self.cas.load(self._root_cid)
        self._root_obj = dag_cbor.decode(root_bytes)

        if self._root_obj.get("manifest_version") != "sharded_zarr_v1":
            raise ValueError(
                f"Incompatible manifest version: {self._root_obj.get('manifest_version')}. Expected 'sharded_zarr_v1'."
            )

        chunk_info = self._root_obj["chunks"]
        self._array_shape = tuple(chunk_info["array_shape"])
        self._chunk_shape = tuple(chunk_info["chunk_shape"])
        self._chunks_per_shard = chunk_info["sharding_config"]["chunks_per_shard"]

        self._update_geometry()

        if len(chunk_info["shard_cids"]) != self._num_shards:
            raise ValueError(
                f"Inconsistent number of shards. Expected {self._num_shards}, found {len(chunk_info['shard_cids'])}."
            )

    async def _fetch_and_cache_full_shard(self, shard_idx: int, shard_cid: str):
        # CHANGED: Logic now decodes the shard from DAG-CBOR into a list.
        try:
            shard_data_bytes = await self.cas.load(shard_cid)
            # Decode the CBOR object, which should be a list of CIDs/None
            decoded_shard = dag_cbor.decode(shard_data_bytes)
            if not isinstance(decoded_shard, list):
                raise TypeError(f"Shard {shard_idx} did not decode to a list.")
            self._shard_data_cache[shard_idx] = decoded_shard
        except Exception as e:
            logging.warning(
                f"Failed to fetch or decode shard {shard_idx} (CID: {shard_cid}): {e}"
            )
        finally:
            if shard_idx in self._pending_shard_loads:
                del self._pending_shard_loads[shard_idx]
    
    # ... (Keep _parse_chunk_key, _get_linear_chunk_index, _get_shard_info as they are) ...
    def _parse_chunk_key(self, key: str) -> Optional[Tuple[int, ...]]:
        # 1. Exclude .json files immediately (metadata)
        if key.endswith(".json"):
            return None
        excluded_array_prefixes = {"time", "lat", "lon", "latitude", "longitude"}

        chunk_marker = "/c/"
        marker_idx = key.rfind(chunk_marker)  # Use rfind for robustness
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
            actual_array_name = path_before_c.split("/")[-1]

        # 2. If the determined array name is in our exclusion list, return None.
        if actual_array_name in excluded_array_prefixes:
            return None

        # If we've reached here, the key is potentially for a "main" data variable
        # that this store instance is expected to handle via sharding.
        # Now, proceed with the original parsing logic using self._array_shape and
        # self._chunks_per_dim, which should be configured for this main data variable.

        # print(
        #     f"Parsing chunk key: {key} for array: {actual_array_name} with shape: {self._array_shape} and chunks_per_dim: {self._chunks_per_dim}")

        if not self._array_shape or not self._chunks_per_dim:
            # This ShardedZarrStore instance is not properly initialized
            # with the shape/chunking info for the array it's supposed to manage.
            # This might also happen if a key like "some_other_main_array/c/0" is passed
            # but this store instance was configured for "temp".
            return None

        # The part after "/c/" contains the chunk coordinates
        coord_part = key[marker_idx + len(chunk_marker) :]
        parts = coord_part.split("/")

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
                    return None  # Coordinate out of bounds for this array's chunk grid
            return coords
        except (ValueError, IndexError):  # If int conversion fails or other issues
            return None

    def _get_linear_chunk_index(self, chunk_coords: Tuple[int, ...]) -> int:
        if self._chunks_per_dim is None:
            raise ValueError("Chunks per dimension not set")
        linear_index = 0
        multiplier = 1
        # Convert N-D chunk coordinates to a flat 1-D index (row-major order)
        for i in reversed(range(len(self._chunks_per_dim))):
            linear_index += chunk_coords[i] * multiplier
            multiplier *= self._chunks_per_dim[i]
        return linear_index

    def _get_shard_info(self, linear_chunk_index: int) -> Tuple[int, int]:
        if self._chunks_per_shard is None or self._chunks_per_shard <= 0:
            raise RuntimeError(
                "Sharding not configured properly: _chunks_per_shard invalid."
            )
        if linear_chunk_index < 0:
            raise ValueError("Linear chunk index cannot be negative.")

        shard_idx = linear_chunk_index // self._chunks_per_shard
        index_in_shard = linear_chunk_index % self._chunks_per_shard
        return shard_idx, index_in_shard

    async def _load_or_initialize_shard_cache(self, shard_idx: int) -> list:
        # CHANGED: This method is updated to handle list-based cache and DAG-CBOR decoding.
        if shard_idx in self._shard_data_cache:
            return self._shard_data_cache[shard_idx]

        if shard_idx in self._pending_shard_loads:
            await self._pending_shard_loads[shard_idx]
            if shard_idx in self._shard_data_cache:
                return self._shard_data_cache[shard_idx]

        if self._root_obj is None or self._num_shards is None:
            raise RuntimeError("Root object not loaded or initialized.")
        if not (0 <= shard_idx < self._num_shards):
            raise ValueError(f"Shard index {shard_idx} out of bounds.")

        shard_cid_obj = self._root_obj["chunks"]["shard_cids"][shard_idx]
        if shard_cid_obj:
            # The CID in the root should already be a CID object if loaded correctly.
            shard_cid_str = str(shard_cid_obj)
            await self._fetch_and_cache_full_shard(shard_idx, shard_cid_str)
        else:
            if self._chunks_per_shard is None:
                raise RuntimeError("Store not initialized: _chunks_per_shard is None.")
            # Initialize new shard as a list of Nones
            self._shard_data_cache[shard_idx] = [None] * self._chunks_per_shard
        
        if shard_idx not in self._shard_data_cache:
             raise RuntimeError(f"Failed to load or initialize shard {shard_idx}")
        
        return self._shard_data_cache[shard_idx]

    async def set_partial_values(
        self, key_start_values: Iterable[Tuple[str, int, BytesLike]]
    ) -> None:
        raise NotImplementedError(
            "Partial writes are not supported by ShardedZarrStore."
        )

    async def get_partial_values(
        self,
        prototype: zarr.core.buffer.BufferPrototype,
        key_ranges: Iterable[Tuple[str, zarr.abc.store.ByteRequest | None]],
    ) -> List[Optional[zarr.core.buffer.Buffer]]:
        tasks = [self.get(key, prototype, byte_range) for key, byte_range in key_ranges]
        results = await asyncio.gather(*tasks)
        return results  # type: ignore

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardedZarrStore):
            return NotImplemented
        # For equality, root CID is primary. Config like chunks_per_shard is part of that root's identity.
        return self._root_cid == other._root_cid
    
    async def flush(self) -> str:
        # CHANGED: This method now encodes shards using DAG-CBOR.
        if self.read_only:
            if not self._root_cid:
                raise ValueError("Read-only store has no root CID to return.")
            return self._root_cid

        if self._root_obj is None:
            raise RuntimeError("Store not initialized for writing.")

        if self._dirty_shards:
            for shard_idx in sorted(list(self._dirty_shards)):
                if shard_idx not in self._shard_data_cache:
                    logging.warning(f"Dirty shard {shard_idx} not in cache. Skipping.")
                    continue

                # Get the list of CIDs/Nones from the cache
                shard_data_list = self._shard_data_cache[shard_idx]

                # Encode this list into a DAG-CBOR byte representation
                shard_data_bytes = dag_cbor.encode(shard_data_list)

                # Save the DAG-CBOR block and get its CID
                new_shard_cid_obj = await self.cas.save(
                    shard_data_bytes, codec="dag-cbor" # Use 'dag-cbor' codec
                )

                if self._root_obj["chunks"]["shard_cids"][shard_idx] != new_shard_cid_obj:
                    # Store the CID object directly
                    self._root_obj["chunks"]["shard_cids"][shard_idx] = new_shard_cid_obj
                    self._dirty_root = True

            self._dirty_shards.clear()

        if self._dirty_root:
            # Ensure all metadata CIDs are CID objects for correct encoding
            self._root_obj["metadata"] = {
                k: (CID.decode(v) if isinstance(v, str) else v)
                for k, v in self._root_obj["metadata"].items()
            }
            root_obj_bytes = dag_cbor.encode(self._root_obj)
            new_root_cid = await self.cas.save(root_obj_bytes, codec="dag-cbor")
            self._root_cid = str(new_root_cid)
            self._dirty_root = False

        if self._root_cid is None:
            raise RuntimeError("Failed to obtain a root CID after flushing.")
        return self._root_cid

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: Optional[zarr.abc.store.ByteRequest] = None,
    ) -> Optional[zarr.core.buffer.Buffer]:
        # CHANGED: Logic is simplified to not use byte offsets. It relies on the full-shard cache.
        if self._root_obj is None:
            raise RuntimeError("Load the root object first before accessing data.")
        # print('Getting key', key)

        chunk_coords = self._parse_chunk_key(key)
        # Metadata request
        if chunk_coords is None:
            metadata_cid_obj = self._root_obj["metadata"].get(key)
            if metadata_cid_obj is None:
                return None
            if byte_range is not None:
                logging.warning(f"Byte range request for metadata key '{key}' ignored.")
            data = await self.cas.load(str(metadata_cid_obj))
            return prototype.buffer.from_bytes(data)

        # Chunk data request
        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        if not (0 <= shard_idx < len(self._root_obj["chunks"]["shard_cids"])):
            return None

        # This will load the full shard into cache if it's not already there.
        target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)

        # Get the CID object (or None) from the cached list.
        chunk_cid_obj = target_shard_list[index_in_shard]
        
        if chunk_cid_obj is None:
            return None # Chunk is empty/doesn't exist.

        chunk_cid_str = str(chunk_cid_obj)

        # Actual chunk data load using the retrieved chunk CID
        req_offset = byte_range.start if byte_range else None
        req_length = None
        if byte_range:
            if byte_range.end is not None:
                if (
                    byte_range.start > byte_range.end
                ):  # Zarr allows start == stop for 0 length
                    raise ValueError(
                        f"Byte range start ({byte_range.start}) cannot be greater than end ({byte_range.end})"
                    )
                req_length = byte_range.end - byte_range.start
        data = await self.cas.load(chunk_cid_str, offset=req_offset, length=req_length)
        return prototype.buffer.from_bytes(data)

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        if self.read_only:
            raise ValueError("Cannot write to a read-only store.")
        if self._root_obj is None:
            raise RuntimeError("Store not initialized for writing. Call open() first.")

        await self._resize_complete.wait()

        print("Setting key:", key)

        if key.endswith("zarr.json") and not key.startswith("time/") and not key.startswith(("lat/", "latitude/")) and not key.startswith(("lon/", "longitude/")) and not len(key) == 9:
            metadata_json = json.loads(value.to_bytes().decode("utf-8"))
            print("setting metadata for key:", key, "with value:", metadata_json)
            new_array_shape = metadata_json.get("shape")
            if not new_array_shape:
                raise ValueError("Shape not found in metadata.")
            if tuple(new_array_shape) != self._array_shape:
                async with self._resize_lock:
                    # Double-check after acquiring the lock, in case another task
                    # just finished this exact resize while we were waiting.
                    if tuple(new_array_shape) != self._array_shape:
                        # Block all other tasks until resize is complete.
                        self._resize_complete.clear()
                        try:
                            await self.resize_store(new_shape=tuple(new_array_shape)) 
                        finally:
                            # All waiting tasks will now un-pause and proceed safely.
                            self._resize_complete.set()

        raw_data_bytes = value.to_bytes()
        # Save the data to CAS first to get its CID.
        # Metadata is often saved as 'raw', chunks as well unless compressed.
        data_cid_obj = await self.cas.save(raw_data_bytes, codec="raw")
        await self.set_pointer(key, str(data_cid_obj))

    async def set_pointer(self, key: str, pointer: str) -> None:
        # CHANGED: Logic now updates a list in the cache, not a bytearray.
        if self._root_obj is None:
            raise RuntimeError("Load the root object first before accessing data.")
        
        chunk_coords = self._parse_chunk_key(key)
        
        pointer_cid_obj = CID.decode(pointer) # Convert string to CID object

        if chunk_coords is None:  # Metadata key
            self._root_obj["metadata"][key] = pointer_cid_obj
            self._dirty_root = True
            return

        # Chunk Data: Store the CID object in the correct shard list.
        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)
        
        if target_shard_list[index_in_shard] != pointer_cid_obj:
            target_shard_list[index_in_shard] = pointer_cid_obj
            self._dirty_shards.add(shard_idx)

    # ... (Keep exists method, but simplify it) ...
    async def exists(self, key: str) -> bool:
        # CHANGED: Simplified to use the list-based cache.
        if self._root_obj is None:
            raise RuntimeError("Root object not loaded.")

        chunk_coords = self._parse_chunk_key(key)
        if chunk_coords is None:  # Metadata
            return key in self._root_obj.get("metadata", {})

        try:
            linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
            shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

            if not (0 <= shard_idx < len(self._root_obj["chunks"]["shard_cids"])):
                return False

            shard_cid_obj = self._root_obj["chunks"]["shard_cids"][shard_idx]
            if shard_cid_obj is None:
                return False

            # Load shard if not cached and check the index
            target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)
            return target_shard_list[index_in_shard] is not None
        except Exception:
            return False
            
    # ... (Keep supports_writes, etc. properties) ...
    @property
    def supports_writes(self) -> bool:
        return not self.read_only

    @property
    def supports_partial_writes(self) -> bool:
        return False  # Each chunk CID is written atomically into a shard slot

    @property
    def supports_deletes(self) -> bool:
        return not self.read_only

    async def delete(self, key: str) -> None:
        # CHANGED: Simplified to set list element to None.
        if self.read_only:
            raise ValueError("Cannot delete from a read-only store.")
        if self._root_obj is None:
            raise RuntimeError("Store not initialized for deletion.")
        
        chunk_coords = self._parse_chunk_key(key)
        if chunk_coords is None:  # Metadata
            if self._root_obj["metadata"].pop(key, None):
                self._dirty_root = True
            else:
                raise KeyError(f"Metadata key '{key}' not found.")
            return

        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        if not (0 <= shard_idx < self._num_shards if self._num_shards is not None else 0):
            raise KeyError(f"Chunk key '{key}' is out of bounds.")

        target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)
        
        if target_shard_list[index_in_shard] is not None:
            target_shard_list[index_in_shard] = None
            self._dirty_shards.add(shard_idx)

    # ... (Keep listing methods as they are, they operate on metadata) ...
    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        if self._root_obj is None:
            raise RuntimeError(
                "Root object not loaded. Call _load_root_from_cid() first."
            )
        for key in list(self._root_obj.get("metadata", {})):
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key
    # ... (Keep graft_store, but it needs significant changes) ...
    
    async def graft_store(self, store_to_graft_cid: str, chunk_offset: Tuple[int, ...]):
        # CHANGED: This method is heavily modified to work with the new DAG-CBOR format.
        if self.read_only:
            raise ValueError("Cannot graft onto a read-only store.")
        if self._root_obj is None:
            raise RuntimeError("Main store must be initialized before grafting.")

        print(f"Grafting store {store_to_graft_cid[:10]}... at chunk offset {chunk_offset}")

        store_to_graft = await ShardedZarrStore.open(cas=self.cas, read_only=True, root_cid=store_to_graft_cid)
        if store_to_graft._root_obj is None or store_to_graft._chunks_per_dim is None:
             raise ValueError("Store to graft could not be loaded or is not configured.")
        source_chunk_grid = store_to_graft._chunks_per_dim
        for local_coords in itertools.product(*[range(s) for s in source_chunk_grid]):
            linear_local_index = store_to_graft._get_linear_chunk_index(local_coords)
            local_shard_idx, index_in_local_shard = store_to_graft._get_shard_info(linear_local_index)

            # Load the source shard into its cache
            source_shard_list = await store_to_graft._load_or_initialize_shard_cache(local_shard_idx)
            
            pointer_cid_obj = source_shard_list[index_in_local_shard]
            if pointer_cid_obj is None:
                continue

            # Calculate global coordinates and write to the main store's index
            global_coords = tuple(c_local + c_offset for c_local, c_offset in zip(local_coords, chunk_offset))
            linear_global_index = self._get_linear_chunk_index(global_coords)
            global_shard_idx, index_in_global_shard = self._get_shard_info(linear_global_index)
            
            target_shard_list = await self._load_or_initialize_shard_cache(global_shard_idx)
            
            if target_shard_list[index_in_global_shard] != pointer_cid_obj:
                target_shard_list[index_in_global_shard] = pointer_cid_obj
                self._dirty_shards.add(global_shard_idx)

        print(f"✓ Grafting complete for store {store_to_graft_cid[:10]}...")

    # ... (Keep resizing methods as they mostly affect metadata) ...
    async def resize_store(self, new_shape: Tuple[int, ...]):
        """
        Resizes the store's main shard index to accommodate a new overall array shape.
        This is a metadata-only operation on the store's root object.
        """
        if self.read_only:
            raise ValueError("Cannot resize a read-only store.")
        if self._root_obj is None or self._chunk_shape is None or self._chunks_per_shard is None:
            raise RuntimeError("Store is not properly initialized for resizing.")
        if len(new_shape) != len(self._array_shape):
            raise ValueError("New shape must have the same number of dimensions as the old shape.")

        self._array_shape = tuple(new_shape)
        self._chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0
            for a, c in zip(self._array_shape, self._chunk_shape)
        )
        self._total_chunks = math.prod(self._chunks_per_dim)
        old_num_shards = self._num_shards if self._num_shards is not None else 0
        self._num_shards = math.ceil(self._total_chunks / self._chunks_per_shard) if self._total_chunks > 0 else 0
        self._root_obj["chunks"]["array_shape"] = list(self._array_shape)
        if self._num_shards > old_num_shards:
            self._root_obj["chunks"]["shard_cids"].extend([None] * (self._num_shards - old_num_shards))
        elif self._num_shards < old_num_shards:
            self._root_obj["chunks"]["shard_cids"] = self._root_obj["chunks"]["shard_cids"][:self._num_shards]

        self._dirty_root = True


    async def resize_variable(self, variable_name: str, new_shape: Tuple[int, ...]):
        """
        Resizes the Zarr metadata for a specific variable (e.g., '.json' file).
        This does NOT change the store's main shard index.
        """
        if self.read_only:
            raise ValueError("Cannot resize a read-only store.")
        if self._root_obj is None:
            raise RuntimeError("Store is not properly initialized for resizing.")

        # Zarr v2 uses .json, not zarr.json
        zarr_metadata_key = f"{variable_name}/zarr.json"
        
        old_zarr_metadata_cid = self._root_obj["metadata"].get(zarr_metadata_key)
        if not old_zarr_metadata_cid:
            raise KeyError(f"Cannot find metadata for key '{zarr_metadata_key}' to resize.")

        old_zarr_metadata_bytes = await self.cas.load(old_zarr_metadata_cid)
        zarr_metadata_json = json.loads(old_zarr_metadata_bytes)
        
        zarr_metadata_json["shape"] = list(new_shape)
        
        new_zarr_metadata_bytes = json.dumps(zarr_metadata_json, indent=2).encode('utf-8')
        # Metadata is a raw blob of bytes
        new_zarr_metadata_cid = await self.cas.save(new_zarr_metadata_bytes, codec='raw')
        
        self._root_obj["metadata"][zarr_metadata_key] = str(new_zarr_metadata_cid)
        self._dirty_root = True
        print(f"Resized metadata for variable '{variable_name}'. New shape: {new_shape}")

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        # This simplified version only works for the root directory (prefix == "") of metadata.
        # It lists unique first components of metadata keys.
        if self._root_obj is None:
            raise RuntimeError(
                "Root object not loaded. Call _load_root_from_cid() first."
            )

        seen: Set[str] = set()
        if prefix == "":
            async for key in self.list():  # Iterates metadata keys
                # e.g., if key is "group1/.zgroup" or "array1/.json", first_component is "group1" or "array1"
                # if key is ".zgroup", first_component is ".zgroup"
                first_component = key.split("/", 1)[0]
                if first_component not in seen:
                    seen.add(first_component)
                    yield first_component
        else:
            # For listing subdirectories like "group1/", we'd need to match keys starting with "group1/"
            # and then extract the next component. This is more involved.
            # Zarr spec: list_dir(path) should yield children (both objects and "directories")
            # For simplicity, and consistency with original FlatZarrStore, keeping this minimal.
            # To make it more compliant for prefix="foo/":
            normalized_prefix = prefix if prefix.endswith("/") else prefix + "/"
            async for key in self.list_prefix(normalized_prefix):
                remainder = key[len(normalized_prefix) :]
                child = remainder.split("/", 1)[0]
                if child not in seen:
                    seen.add(child)
                    yield child
