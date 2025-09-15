import asyncio
import itertools
import json
import math
import sys
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator, Iterable
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import dag_cbor
import zarr.abc.store
import zarr.core.buffer
from multiformats.cid import CID
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.common import BytesLike

from .store_httpx import ContentAddressedStore


class MemoryBoundedLRUCache:
    """
    An LRU cache that evicts items when memory usage exceeds a threshold.

    Memory usage is calculated using sys.getsizeof for accurate sizing.
    Dirty shards (those marked for writing) are never evicted until marked clean.
    All operations are thread-safe for async access using an asyncio.Lock.
    """

    def __init__(self, max_memory_bytes: int = 100 * 1024 * 1024):  # 100MB default
        self.max_memory_bytes = max_memory_bytes
        self._cache: OrderedDict[int, List[Optional[CID]]] = OrderedDict()
        self._dirty_shards: Set[int] = set()
        self._shard_sizes: Dict[int, int] = {}  # Cached sizes for each shard
        self._actual_memory_usage = 0
        self._cache_lock = asyncio.Lock()

    def _get_shard_size(self, shard_data: List[Optional[CID]]) -> int:
        """Compute actual size: list overhead + sum of item sizes."""
        if not shard_data:
            return sys.getsizeof(shard_data)
        total = sys.getsizeof(shard_data)
        for item in shard_data:
            total += sys.getsizeof(item)
        return total

    async def get(self, shard_idx: int) -> Optional[List[Optional[CID]]]:
        """Get a shard from cache, moving it to end (most recently used)."""
        async with self._cache_lock:
            if shard_idx not in self._cache:
                return None
            shard_data = self._cache.pop(shard_idx)
            self._cache[shard_idx] = shard_data
            return shard_data

    async def put(
        self, shard_idx: int, shard_data: List[Optional[CID]], is_dirty: bool = False
    ) -> None:
        """Add or update a shard in cache, evicting old items if needed."""
        async with self._cache_lock:
            shard_size = self._get_shard_size(shard_data)

            # If shard exists, remove its old size
            if shard_idx in self._cache:
                self._cache.pop(shard_idx)
                self._actual_memory_usage -= self._shard_sizes.pop(shard_idx, 0)

            # Track dirty status
            if is_dirty:
                self._dirty_shards.add(shard_idx)

            # Add new shard
            self._cache[shard_idx] = shard_data
            self._shard_sizes[shard_idx] = shard_size
            self._actual_memory_usage += shard_size

            # Evict old items if over memory limit, never evict dirty shards
            while (
                self._actual_memory_usage > self.max_memory_bytes
                and len(self._cache) > 1
            ):
                evicted = False
                checked_dirty_shards = set()
                while self._cache:
                    candidate_idx, candidate_data = self._cache.popitem(last=False)
                    if candidate_idx not in self._dirty_shards:
                        # Evict this clean LRU
                        self._actual_memory_usage -= self._shard_sizes.pop(
                            candidate_idx, 0
                        )
                        evicted = True
                        break
                    else:
                        # Dirty: move to MRU
                        self._cache[candidate_idx] = candidate_data
                        checked_dirty_shards.add(candidate_idx)
                        # If we've checked all dirty shards, no clean shards available
                        if len(checked_dirty_shards) == len(self._dirty_shards):
                            break
                if not evicted:
                    # No clean shards to evict
                    break

    async def mark_dirty(self, shard_idx: int) -> None:
        """Mark a shard as dirty (should not be evicted)."""
        async with self._cache_lock:
            if shard_idx in self._cache:
                self._dirty_shards.add(shard_idx)

    async def mark_clean(self, shard_idx: int) -> None:
        """Mark a shard as clean (can be evicted)."""
        async with self._cache_lock:
            self._dirty_shards.discard(shard_idx)

    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._cache_lock:
            self._cache.clear()
            self._dirty_shards.clear()
            self._shard_sizes.clear()
            self._actual_memory_usage = 0

    async def __contains__(self, shard_idx: int) -> bool:
        async with self._cache_lock:
            return shard_idx in self._cache

    @property
    def estimated_memory_usage(self) -> int:
        """Current memory usage in bytes, based on actual sizes."""
        return self._actual_memory_usage

    @property
    def cache_size(self) -> int:
        """Number of items currently cached."""
        return len(self._cache)

    @property
    def dirty_cache_size(self) -> int:
        """Number of dirty items currently cached."""
        return len(self._dirty_shards)


class ShardedZarrStore(zarr.abc.store.Store):
    """
    Implements the Zarr Store API using a sharded layout for chunk CIDs.

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
        root_cid: Optional[str] = None,
        *,
        max_cache_memory_bytes: int = 100 * 1024 * 1024,  # 100MB default
    ):
        """Use the async `open()` classmethod to instantiate this class."""
        super().__init__(read_only=read_only)
        self.cas = cas
        self._root_cid = root_cid
        self._root_obj: dict

        self._resize_lock = asyncio.Lock()
        # An event to signal when a resize is in-progress.
        # It starts in the "set" state, allowing all operations to proceed.
        self._resize_complete = asyncio.Event()
        self._resize_complete.set()
        self._shard_locks: DefaultDict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

        self._shard_data_cache = MemoryBoundedLRUCache(max_cache_memory_bytes)
        self._pending_shard_loads: Dict[int, asyncio.Event] = {}

        self._array_shape: Tuple[int, ...]
        self._chunk_shape: Tuple[int, ...]
        self._chunks_per_dim: Tuple[int, ...]
        self._chunks_per_shard: int
        self._num_shards: int = 0
        self._total_chunks: int = 0

        self._dirty_root = False

    def __update_geometry(self):
        """Calculates derived geometric properties from the base shapes."""

        if not all(cs > 0 for cs in self._chunk_shape):
            raise ValueError("All chunk_shape dimensions must be positive.")
        if not all(s >= 0 for s in self._array_shape):
            raise ValueError("All array_shape dimensions must be non-negative.")

        self._chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0
            for a, c in zip(self._array_shape, self._chunk_shape)
        )
        self._total_chunks = math.prod(self._chunks_per_dim)

        if not self._total_chunks == 0:
            self._num_shards = (
                self._total_chunks + self._chunks_per_shard - 1
            ) // self._chunks_per_shard

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
        max_cache_memory_bytes: int = 100 * 1024 * 1024,  # 100MB default
    ) -> "ShardedZarrStore":
        """
        Asynchronously opens an existing ShardedZarrStore or initializes a new one.
        """
        store = cls(
            cas, read_only, root_cid, max_cache_memory_bytes=max_cache_memory_bytes
        )
        if root_cid:
            await store._load_root_from_cid()
        elif not read_only:
            if array_shape is None or chunk_shape is None:
                raise ValueError(
                    "array_shape and chunk_shape must be provided for a new store."
                )

            if not isinstance(chunks_per_shard, int) or chunks_per_shard <= 0:
                raise ValueError("chunks_per_shard must be a positive integer.")

            store._initialize_new_root(array_shape, chunk_shape, chunks_per_shard)
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

        self.__update_geometry()

        self._root_obj = {
            "manifest_version": "sharded_zarr_v1",
            "metadata": {},
            "chunks": {
                "array_shape": list(self._array_shape),
                "chunk_shape": list(self._chunk_shape),
                "sharding_config": {
                    "chunks_per_shard": self._chunks_per_shard,
                },
                "shard_cids": [None] * self._num_shards,
            },
        }
        self._dirty_root = True

    async def _load_root_from_cid(self):
        root_bytes = await self.cas.load(self._root_cid)
        try:
            self._root_obj = dag_cbor.decode(root_bytes)
            if not isinstance(self._root_obj, dict) or "chunks" not in self._root_obj:
                raise ValueError(
                    "Root object is not a valid dictionary with 'chunks' key."
                )
            if not isinstance(self._root_obj["chunks"]["shard_cids"], list):
                raise ValueError("shard_cids is not a list.")
        except Exception as e:
            raise ValueError(f"Failed to decode root object: {e}")

        if self._root_obj.get("manifest_version") != "sharded_zarr_v1":
            raise ValueError(
                f"Incompatible manifest version: {self._root_obj.get('manifest_version')}. Expected 'sharded_zarr_v1'."
            )

        chunk_info = self._root_obj["chunks"]
        self._array_shape = tuple(chunk_info["array_shape"])
        self._chunk_shape = tuple(chunk_info["chunk_shape"])
        self._chunks_per_shard = chunk_info["sharding_config"]["chunks_per_shard"]

        self.__update_geometry()

        if len(chunk_info["shard_cids"]) != self._num_shards:
            raise ValueError(
                f"Inconsistent number of shards. Expected {self._num_shards}, found {len(chunk_info['shard_cids'])}."
            )

    async def _fetch_and_cache_full_shard(
        self,
        shard_idx: int,
        shard_cid: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Fetch a shard from CAS and cache it, with retry logic for transient errors.

        Args:
            shard_idx: The index of the shard to fetch.
            shard_cid: The CID of the shard.
            max_retries: Maximum number of retry attempts for transient errors.
            retry_delay: Delay between retry attempts in seconds.
        """
        for attempt in range(max_retries):
            try:
                shard_data_bytes = await self.cas.load(shard_cid)
                decoded_shard = dag_cbor.decode(shard_data_bytes)
                if not isinstance(decoded_shard, list):
                    raise TypeError(f"Shard {shard_idx} did not decode to a list.")
                await self._shard_data_cache.put(shard_idx, decoded_shard)
                # Always set the Event to unblock waiting coroutines
                if shard_idx in self._pending_shard_loads:
                    self._pending_shard_loads[shard_idx].set()
                    del self._pending_shard_loads[shard_idx]
                return  # Success
            except (ConnectionError, TimeoutError) as e:
                # Handle transient errors (e.g., network issues)
                if attempt < max_retries - 1:
                    await asyncio.sleep(
                        retry_delay * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    raise RuntimeError(
                        f"Failed to fetch shard {shard_idx} after {max_retries} attempts: {e}"
                    )

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

        # The part after "/c/" contains the chunk coordinates
        coord_part = key[marker_idx + len(chunk_marker) :]
        parts = coord_part.split("/")

        coords = tuple(map(int, parts))
        # Validate coordinates against the chunk grid of the store's configured array
        for i, c_coord in enumerate(coords):
            if not (0 <= c_coord < self._chunks_per_dim[i]):
                raise IndexError(
                    f"Chunk coordinate {c_coord} at dimension {i} is out of bounds for dimension size {self._chunks_per_dim[i]}."
                )
        return coords

    def _get_linear_chunk_index(self, chunk_coords: Tuple[int, ...]) -> int:
        linear_index = 0
        multiplier = 1
        # Convert N-D chunk coordinates to a flat 1-D index (row-major order)
        for i in reversed(range(len(self._chunks_per_dim))):
            linear_index += chunk_coords[i] * multiplier
            multiplier *= self._chunks_per_dim[i]
        return linear_index

    def _get_shard_info(self, linear_chunk_index: int) -> Tuple[int, int]:
        shard_idx = linear_chunk_index // self._chunks_per_shard
        index_in_shard = linear_chunk_index % self._chunks_per_shard
        return shard_idx, index_in_shard

    async def _load_or_initialize_shard_cache(
        self, shard_idx: int
    ) -> List[Optional[CID]]:
        """
        Load a shard into the cache or initialize an empty shard if it doesn't exist.

        Args:
            shard_idx: The index of the shard to load or initialize.

        Returns:
            List[Optional[CID]]: The shard data (list of CIDs or None).

        Raises:
            ValueError: If the shard index is out of bounds.
            RuntimeError: If the shard cannot be loaded or initialized.
        """
        cached_shard = await self._shard_data_cache.get(shard_idx)
        if cached_shard is not None:
            return cached_shard

        if shard_idx in self._pending_shard_loads:
            try:
                # Wait for the pending load with a timeout (e.g., 60 seconds)
                await asyncio.wait_for(
                    self._pending_shard_loads[shard_idx].wait(), timeout=60.0
                )
                cached_shard = await self._shard_data_cache.get(shard_idx)
                if cached_shard is not None:
                    return cached_shard
                else:
                    raise RuntimeError(
                        f"Shard {shard_idx} not found in cache after pending load completed."
                    )
            except asyncio.TimeoutError:
                # Clean up the pending load to allow retry
                if shard_idx in self._pending_shard_loads:
                    self._pending_shard_loads[shard_idx].set()
                    del self._pending_shard_loads[shard_idx]
                raise RuntimeError(f"Timeout waiting for shard {shard_idx} to load.")

        if not (0 <= shard_idx < self._num_shards):
            raise ValueError(f"Shard index {shard_idx} out of bounds.")

        shard_cid_obj = self._root_obj["chunks"]["shard_cids"][shard_idx]
        if shard_cid_obj:
            self._pending_shard_loads[shard_idx] = asyncio.Event()
            shard_cid_str = str(shard_cid_obj)
            await self._fetch_and_cache_full_shard(shard_idx, shard_cid_str)
        else:
            empty_shard = [None] * self._chunks_per_shard
            await self._shard_data_cache.put(shard_idx, empty_shard)

        result = await self._shard_data_cache.get(shard_idx)
        if result is None:
            raise RuntimeError(f"Failed to load or initialize shard {shard_idx}")
        return result  # type: ignore[return-value]

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
        return results

    def with_read_only(self, read_only: bool = False) -> "ShardedZarrStore":
        """
        Return this store (if the flag already matches) or a *shallow*
        clone that presents the requested read‑only status.

        The clone **shares** the same CAS instance and internal state;
        no flushing, network traffic or async work is done.
        """
        # Fast path
        if read_only == self.read_only:
            return self  # Same mode, return same instance

        # Create new instance with different read_only flag
        # Creates a *bare* instance without running its __init__
        clone = type(self).__new__(type(self))

        # Copy all attributes from the current instance
        clone.cas = self.cas
        clone._root_cid = self._root_cid
        clone._root_obj = self._root_obj

        clone._resize_lock = self._resize_lock
        clone._resize_complete = self._resize_complete
        clone._shard_locks = self._shard_locks

        clone._shard_data_cache = self._shard_data_cache
        clone._pending_shard_loads = self._pending_shard_loads

        clone._array_shape = self._array_shape
        clone._chunk_shape = self._chunk_shape
        clone._chunks_per_dim = self._chunks_per_dim
        clone._chunks_per_shard = self._chunks_per_shard
        clone._num_shards = self._num_shards
        clone._total_chunks = self._total_chunks

        clone._dirty_root = self._dirty_root

        # Re‑initialise the zarr base class so that Zarr sees the flag
        zarr.abc.store.Store.__init__(clone, read_only=read_only)
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardedZarrStore):
            return False
        # For equality, root CID is primary. Config like chunks_per_shard is part of that root's identity.
        return self._root_cid == other._root_cid

    # If nothing to flush, return the root CID.
    async def flush(self) -> str:
        async with self._shard_data_cache._cache_lock:
            dirty_shards = list(self._shard_data_cache._dirty_shards)
        if dirty_shards:
            for shard_idx in sorted(dirty_shards):
                # Get the list of CIDs/Nones from the cache
                shard_data_list = await self._shard_data_cache.get(shard_idx)
                if shard_data_list is None:
                    raise RuntimeError(f"Dirty shard {shard_idx} not found in cache")

                # Encode this list into a DAG-CBOR byte representation
                shard_data_bytes = dag_cbor.encode(shard_data_list)

                # Save the DAG-CBOR block and get its CID
                new_shard_cid_obj = await self.cas.save(
                    shard_data_bytes,
                    codec="dag-cbor",  # Use 'dag-cbor' codec
                )

                if (
                    self._root_obj["chunks"]["shard_cids"][shard_idx]
                    != new_shard_cid_obj
                ):
                    # Store the CID object directly
                    self._root_obj["chunks"]["shard_cids"][shard_idx] = (
                        new_shard_cid_obj
                    )
                    self._dirty_root = True
                    # Mark shard as clean after flushing
                    await self._shard_data_cache.mark_clean(shard_idx)

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

        # Ignore because root_cid will always exist after initialization or flush.
        return self._root_cid  # type: ignore[return-value]

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: Optional[zarr.abc.store.ByteRequest] = None,
    ) -> Optional[zarr.core.buffer.Buffer]:
        chunk_coords = self._parse_chunk_key(key)
        # Metadata request
        if chunk_coords is None:
            metadata_cid_obj = self._root_obj["metadata"].get(key)
            if metadata_cid_obj is None:
                return None
            if byte_range is not None:
                raise ValueError(
                    "Byte range requests are not supported for metadata keys."
                )
            data = await self.cas.load(str(metadata_cid_obj))
            return prototype.buffer.from_bytes(data)
        # Chunk data request
        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        # This will load the full shard into cache if it's not already there.
        shard_lock = self._shard_locks[shard_idx]
        async with shard_lock:
            target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)

        # Get the CID object (or None) from the cached list.
        chunk_cid_obj = target_shard_list[index_in_shard]

        if chunk_cid_obj is None:
            return None  # Chunk is empty/doesn't exist.

        chunk_cid_str = str(chunk_cid_obj)

        req_offset = None
        req_length = None
        req_suffix = None

        if byte_range:
            if isinstance(byte_range, RangeByteRequest):
                req_offset = byte_range.start
                if byte_range.end is not None:
                    if byte_range.start > byte_range.end:
                        raise ValueError(
                            f"Byte range start ({byte_range.start}) cannot be greater than end ({byte_range.end})"
                        )
                    req_length = byte_range.end - byte_range.start
            elif isinstance(byte_range, OffsetByteRequest):
                req_offset = byte_range.offset
            elif isinstance(byte_range, SuffixByteRequest):
                req_suffix = byte_range.suffix
        data = await self.cas.load(
            chunk_cid_str, offset=req_offset, length=req_length, suffix=req_suffix
        )
        return prototype.buffer.from_bytes(data)

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        if self.read_only:
            raise PermissionError("Cannot write to a read-only store.")
        await self._resize_complete.wait()

        if (
            key.endswith("zarr.json")
            and not key.startswith("time/")
            and not key.startswith(("lat/", "latitude/"))
            and not key.startswith(("lon/", "longitude/"))
            and not key == "zarr.json"
        ):
            metadata_json = json.loads(value.to_bytes().decode("utf-8"))
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
        try:
            data_cid_obj = await self.cas.save(raw_data_bytes, codec="raw")
            await self.set_pointer(key, str(data_cid_obj))
        except Exception as e:
            raise RuntimeError(f"Failed to save data for key {key}: {e}")
        return None  # type: ignore[return-value]

    async def set_pointer(self, key: str, pointer: str) -> None:
        chunk_coords = self._parse_chunk_key(key)

        pointer_cid_obj = CID.decode(pointer)  # Convert string to CID object

        if chunk_coords is None:  # Metadata key
            self._root_obj["metadata"][key] = pointer_cid_obj
            self._dirty_root = True
            return None

        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        shard_lock = self._shard_locks[shard_idx]
        async with shard_lock:
            target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)

            if target_shard_list[index_in_shard] != pointer_cid_obj:
                target_shard_list[index_in_shard] = pointer_cid_obj
                await self._shard_data_cache.mark_dirty(shard_idx)
        return None

    async def exists(self, key: str) -> bool:
        try:
            chunk_coords = self._parse_chunk_key(key)
            if chunk_coords is None:  # Metadata
                return key in self._root_obj.get("metadata", {})
            linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
            shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)
            # Load shard if not cached and check the index
            target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)
            return target_shard_list[index_in_shard] is not None
        except (ValueError, IndexError, KeyError):
            return False

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
        if self.read_only:
            raise PermissionError("Cannot delete from a read-only store.")

        chunk_coords = self._parse_chunk_key(key)
        if chunk_coords is None:  # Metadata
            if self._root_obj["metadata"].pop(key, None):
                self._dirty_root = True
            else:
                raise KeyError(f"Metadata key '{key}' not found.")
            return None

        linear_chunk_index = self._get_linear_chunk_index(chunk_coords)
        shard_idx, index_in_shard = self._get_shard_info(linear_chunk_index)

        shard_lock = self._shard_locks[shard_idx]
        async with shard_lock:
            target_shard_list = await self._load_or_initialize_shard_cache(shard_idx)
            if target_shard_list[index_in_shard] is not None:
                target_shard_list[index_in_shard] = None
                await self._shard_data_cache.mark_dirty(shard_idx)

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        for key in list(self._root_obj.get("metadata", {})):
            yield key

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    async def graft_store(self, store_to_graft_cid: str, chunk_offset: Tuple[int, ...]):
        if self.read_only:
            raise PermissionError("Cannot graft onto a read-only store.")

        store_to_graft = await ShardedZarrStore.open(
            cas=self.cas, read_only=True, root_cid=store_to_graft_cid
        )
        source_chunk_grid = store_to_graft._chunks_per_dim
        for local_coords in itertools.product(*[range(s) for s in source_chunk_grid]):
            linear_local_index = store_to_graft._get_linear_chunk_index(local_coords)
            local_shard_idx, index_in_local_shard = store_to_graft._get_shard_info(
                linear_local_index
            )
            # Load the source shard into its cache
            source_shard_list = await store_to_graft._load_or_initialize_shard_cache(
                local_shard_idx
            )

            pointer_cid_obj = source_shard_list[index_in_local_shard]
            if pointer_cid_obj is None:
                continue

            # Calculate global coordinates and write to the main store's index
            global_coords = tuple(
                c_local + c_offset
                for c_local, c_offset in zip(local_coords, chunk_offset)
            )
            linear_global_index = self._get_linear_chunk_index(global_coords)
            global_shard_idx, index_in_global_shard = self._get_shard_info(
                linear_global_index
            )

            shard_lock = self._shard_locks[global_shard_idx]
            async with shard_lock:
                target_shard_list = await self._load_or_initialize_shard_cache(
                    global_shard_idx
                )
                if target_shard_list[index_in_global_shard] != pointer_cid_obj:
                    target_shard_list[index_in_global_shard] = pointer_cid_obj
                    await self._shard_data_cache.mark_dirty(global_shard_idx)

    async def resize_store(self, new_shape: Tuple[int, ...]):
        """
        Resizes the store's main shard index to accommodate a new overall array shape.
        This is a metadata-only operation on the store's root object.
        Used when doing skeleton writes or appends via xarray where the array shape changes.
        """
        if self.read_only:
            raise PermissionError("Cannot resize a read-only store.")
        if (
            # self._root_obj is None
            self._chunk_shape is None
            or self._chunks_per_shard is None
            or self._array_shape is None
        ):
            raise RuntimeError("Store is not properly initialized for resizing.")
        if len(new_shape) != len(self._array_shape):
            raise ValueError(
                "New shape must have the same number of dimensions as the old shape."
            )

        self._array_shape = tuple(new_shape)
        self._chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0
            for a, c in zip(self._array_shape, self._chunk_shape)
        )
        self._total_chunks = math.prod(self._chunks_per_dim)
        old_num_shards = self._num_shards if self._num_shards is not None else 0
        self._num_shards = (
            (self._total_chunks + self._chunks_per_shard - 1) // self._chunks_per_shard
            if self._total_chunks > 0
            else 0
        )
        self._root_obj["chunks"]["array_shape"] = list(self._array_shape)
        if self._num_shards > old_num_shards:
            self._root_obj["chunks"]["shard_cids"].extend(
                [None] * (self._num_shards - old_num_shards)
            )
        elif self._num_shards < old_num_shards:
            self._root_obj["chunks"]["shard_cids"] = self._root_obj["chunks"][
                "shard_cids"
            ][: self._num_shards]

        self._dirty_root = True

    async def resize_variable(self, variable_name: str, new_shape: Tuple[int, ...]):
        """
        Resizes the Zarr metadata for a specific variable (e.g., '.json' file).
        This does NOT change the store's main shard index.
        """
        if self.read_only:
            raise PermissionError("Cannot resize a read-only store.")

        zarr_metadata_key = f"{variable_name}/zarr.json"

        old_zarr_metadata_cid = self._root_obj["metadata"].get(zarr_metadata_key)
        if not old_zarr_metadata_cid:
            raise KeyError(
                f"Cannot find metadata for key '{zarr_metadata_key}' to resize."
            )

        old_zarr_metadata_bytes = await self.cas.load(old_zarr_metadata_cid)
        zarr_metadata_json = json.loads(old_zarr_metadata_bytes)

        zarr_metadata_json["shape"] = list(new_shape)

        new_zarr_metadata_bytes = json.dumps(zarr_metadata_json, indent=2).encode(
            "utf-8"
        )
        # Metadata is a raw blob of bytes
        new_zarr_metadata_cid = await self.cas.save(
            new_zarr_metadata_bytes, codec="raw"
        )

        self._root_obj["metadata"][zarr_metadata_key] = new_zarr_metadata_cid
        self._dirty_root = True

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
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
            raise NotImplementedError("Listing with a prefix is not implemented yet.")
