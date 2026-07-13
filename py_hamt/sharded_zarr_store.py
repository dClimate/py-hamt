import asyncio
import itertools
import json
import math
import sys
import time
import warnings
from collections import OrderedDict, defaultdict
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import (
    ClassVar,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Set,
    Tuple,
    cast,
)

import dag_cbor
import zarr.abc.store
import zarr.core.buffer
from dag_cbor.ipld import IPLDKind
from multiformats.cid import CID
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest
from zarr.core.common import BytesLike

from . import instrumentation
from .store_httpx import ContentAddressedStore

SHARDED_ZARR_V1 = "sharded_zarr_v1"
SHARDED_ZARR_V2 = "sharded_zarr_v2"
ZARR_METADATA_SUFFIXES = ("zarr.json", ".zarray", ".zattrs", ".zgroup")

ShardCacheKey = int | tuple[str, int]
ShardReadMode = Literal["full", "sparse"]


def _read_cbor_argument(
    data: bytes, offset: int, additional_info: int
) -> tuple[int, int]:
    if additional_info < 24:
        return additional_info, offset
    if additional_info == 24:
        length = 1
    elif additional_info == 25:
        length = 2
    elif additional_info == 26:
        length = 4
    elif additional_info == 27:
        length = 8
    else:
        raise ValueError("Indefinite or reserved CBOR item length is not supported.")

    end = offset + length
    if end > len(data):
        raise ValueError("Truncated CBOR item.")
    return int.from_bytes(data[offset:end], "big"), end


def _skip_cbor_item(data: bytes, offset: int) -> int:
    if offset >= len(data):
        raise ValueError("Truncated CBOR item.")

    initial_byte = data[offset]
    offset += 1
    major_type = initial_byte >> 5
    additional_info = initial_byte & 0x1F
    value, offset = _read_cbor_argument(data, offset, additional_info)

    if major_type in {0, 1, 7}:
        return offset
    if major_type in {2, 3}:
        end = offset + value
        if end > len(data):
            raise ValueError("Truncated CBOR byte or text string.")
        return end
    if major_type == 4:
        for _ in range(value):
            offset = _skip_cbor_item(data, offset)
        return offset
    if major_type == 5:
        for _ in range(value * 2):
            offset = _skip_cbor_item(data, offset)
        return offset
    if major_type == 6:
        return _skip_cbor_item(data, offset)

    raise ValueError(  # pragma: no cover - all CBOR major types are handled above
        f"Unsupported CBOR major type: {major_type}."
    )


def _read_cbor_list_header(shard_bytes: bytes) -> tuple[int, int]:
    if not shard_bytes:
        raise ValueError("Shard bytes are empty.")

    initial_byte = shard_bytes[0]
    major_type = initial_byte >> 5
    if major_type != 4:
        raise ValueError("Shard bytes do not start with a DAG-CBOR list.")

    return _read_cbor_argument(shard_bytes, 1, initial_byte & 0x1F)


def decode_shard_entry(shard_bytes: bytes, index: int) -> CID | None:
    """
    Decode one entry from a DAG-CBOR shard list without materializing the full list.
    """
    if index < 0:
        raise IndexError("Shard entry index out of range.")

    entry_count, offset = _read_cbor_list_header(shard_bytes)
    if index >= entry_count:
        raise IndexError("Shard entry index out of range.")

    for _ in range(index):
        offset = _skip_cbor_item(shard_bytes, offset)

    entry_start = offset
    entry_end = _skip_cbor_item(shard_bytes, offset)
    entry = dag_cbor.decode(shard_bytes[entry_start:entry_end])
    if entry is not None and not isinstance(entry, CID):
        raise TypeError("Shard entry contains a non-CID value.")
    return entry


class ShardedZarrV1DeprecationWarning(FutureWarning):
    """Warning emitted when using deprecated sharded_zarr_v1 roots."""


@dataclass(frozen=True)
class ChunkKey:
    """A parsed Zarr v3 chunk key."""

    array_path: str
    coords: tuple[int, ...]


@dataclass
class ArrayIndex:
    """Path-local shard index and chunk geometry for one Zarr array."""

    array_path: str
    array_shape: tuple[int, ...]
    chunk_shape: tuple[int, ...]
    chunks_per_shard: int
    shard_cids: list[Optional[CID]]
    order: str = "C"

    def __post_init__(self) -> None:
        self.array_path = ShardedZarrStore._normalize_array_path(self.array_path)
        self.array_shape = tuple(self.array_shape)
        self.chunk_shape = tuple(self.chunk_shape)
        self._validate_geometry()
        self.chunks_per_dim = self._calculate_chunks_per_dim()
        self.total_chunks = math.prod(self.chunks_per_dim)
        self.num_shards = (
            (self.total_chunks + self.chunks_per_shard - 1) // self.chunks_per_shard
            if self.total_chunks > 0
            else 0
        )
        if len(self.shard_cids) != self.num_shards:
            raise ValueError(
                f"Inconsistent number of shards. Expected {self.num_shards}, found {len(self.shard_cids)}."
            )

    def _validate_geometry(self) -> None:
        if not isinstance(self.chunks_per_shard, int) or self.chunks_per_shard <= 0:
            raise ValueError("chunks_per_shard must be a positive integer.")
        if len(self.array_shape) != len(self.chunk_shape):
            raise ValueError("array_shape and chunk_shape must have the same rank.")
        if not all(cs > 0 for cs in self.chunk_shape):
            raise ValueError("All chunk_shape dimensions must be positive.")
        if not all(s >= 0 for s in self.array_shape):
            raise ValueError("All array_shape dimensions must be non-negative.")
        if self.order != "C":
            raise ValueError("Only row-major ('C') shard ordering is supported.")

    def _calculate_chunks_per_dim(self) -> tuple[int, ...]:
        return tuple(
            math.ceil(a / c) if c > 0 else 0
            for a, c in zip(self.array_shape, self.chunk_shape, strict=True)
        )

    @classmethod
    def new(
        cls,
        array_path: str,
        array_shape: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        chunks_per_shard: int,
        *,
        order: str = "C",
    ) -> "ArrayIndex":
        chunks_per_dim = tuple(
            math.ceil(a / c) if c > 0 else 0
            for a, c in zip(array_shape, chunk_shape, strict=True)
        )
        total_chunks = math.prod(chunks_per_dim)
        num_shards = (
            (total_chunks + chunks_per_shard - 1) // chunks_per_shard
            if total_chunks > 0
            else 0
        )
        return cls(
            array_path=array_path,
            array_shape=array_shape,
            chunk_shape=chunk_shape,
            chunks_per_shard=chunks_per_shard,
            shard_cids=[None] * num_shards,
            order=order,
        )

    @classmethod
    def from_manifest(cls, array_path: str, manifest: dict) -> "ArrayIndex":
        shard_cids = manifest.get("shard_cids")
        if not isinstance(shard_cids, list):
            raise ValueError("shard_cids is not a list.")

        sharding_config = manifest.get("sharding_config", {})
        if not isinstance(sharding_config, dict):
            raise ValueError("sharding_config is not a dictionary.")

        return cls(
            array_path=array_path,
            array_shape=tuple(manifest["array_shape"]),
            chunk_shape=tuple(manifest["chunk_shape"]),
            chunks_per_shard=sharding_config["chunks_per_shard"],
            order=sharding_config.get("order", "C"),
            shard_cids=list(shard_cids),
        )

    def resize(self, new_shape: tuple[int, ...]) -> None:
        if len(new_shape) != len(self.array_shape):
            raise ValueError(
                "New shape must have the same number of dimensions as the old shape."
            )

        old_shard_cids = self.shard_cids
        self.array_shape = tuple(new_shape)
        self._validate_geometry()
        self.chunks_per_dim = self._calculate_chunks_per_dim()
        self.total_chunks = math.prod(self.chunks_per_dim)
        old_num_shards = self.num_shards
        self.num_shards = (
            (self.total_chunks + self.chunks_per_shard - 1) // self.chunks_per_shard
            if self.total_chunks > 0
            else 0
        )

        if self.num_shards > old_num_shards:
            self.shard_cids = old_shard_cids + [None] * (
                self.num_shards - old_num_shards
            )
        elif self.num_shards < old_num_shards:
            self.shard_cids = old_shard_cids[: self.num_shards]
        else:
            self.shard_cids = old_shard_cids

    def to_manifest(self) -> dict:
        return {
            "array_shape": list(self.array_shape),
            "chunk_shape": list(self.chunk_shape),
            "sharding_config": {
                "chunks_per_shard": self.chunks_per_shard,
                "order": self.order,
            },
            "shard_cids": self.shard_cids,
        }


class MemoryBoundedLRUCache:
    """
    An LRU cache that evicts items when memory usage exceeds a threshold.

    Memory usage is calculated using sys.getsizeof for accurate sizing.
    Dirty shards (those marked for writing) and shards pinned by active users are
    never evicted. Pins are refcounted so concurrent users can safely share a shard.
    If protected shards exceed the configured budget, the cache temporarily
    overflows rather than evicting data that is in use or waiting to be flushed.
    All operations are thread-safe for async access using an asyncio.Lock.
    """

    def __init__(self, max_memory_bytes: int = 100 * 1024 * 1024):  # 100MB default
        self.max_memory_bytes = max_memory_bytes
        self._cache: OrderedDict[ShardCacheKey, List[Optional[CID]]] = OrderedDict()
        self._dirty_shards: Set[ShardCacheKey] = set()
        self._pin_counts: Dict[ShardCacheKey, int] = {}
        self._shard_sizes: Dict[ShardCacheKey, int] = {}
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

    async def get(self, shard_idx: ShardCacheKey) -> Optional[List[Optional[CID]]]:
        """Get a shard from cache, moving it to end (most recently used)."""
        async with self._cache_lock:
            if shard_idx not in self._cache:
                return None
            shard_data = self._cache.pop(shard_idx)
            self._cache[shard_idx] = shard_data
            return shard_data

    async def put(
        self,
        shard_idx: ShardCacheKey,
        shard_data: List[Optional[CID]],
        is_dirty: bool = False,
    ) -> None:
        """Add or update a shard in cache, evicting old items if needed."""
        async with self._cache_lock:
            shard_size = self._get_shard_size(shard_data)

            if shard_idx in self._cache:
                self._cache.pop(shard_idx)
                self._actual_memory_usage -= self._shard_sizes.pop(shard_idx, 0)

            if is_dirty:
                self._dirty_shards.add(shard_idx)

            self._cache[shard_idx] = shard_data
            self._shard_sizes[shard_idx] = shard_size
            self._actual_memory_usage += shard_size

            self._evict_if_needed_locked()

    def _evict_if_needed_locked(self) -> None:
        """Evict clean, unpinned LRU entries while the cache lock is held."""
        while (
            self._actual_memory_usage > self.max_memory_bytes and len(self._cache) > 1
        ):
            candidate_idx = next(
                (
                    cached_idx
                    for cached_idx in self._cache
                    if cached_idx not in self._dirty_shards
                    and self._pin_counts.get(cached_idx, 0) == 0
                ),
                None,
            )
            if candidate_idx is None:
                return
            self._cache.pop(candidate_idx)
            self._actual_memory_usage -= self._shard_sizes.pop(candidate_idx, 0)

    @asynccontextmanager
    async def pin(self, shard_idx: ShardCacheKey) -> AsyncIterator[None]:
        """Prevent a shard from being evicted for the duration of the context."""
        async with self._cache_lock:
            self._pin_counts[shard_idx] = self._pin_counts.get(shard_idx, 0) + 1
        try:
            yield
        finally:
            async with self._cache_lock:
                remaining_pins = self._pin_counts[shard_idx] - 1
                if remaining_pins:
                    self._pin_counts[shard_idx] = remaining_pins
                else:
                    del self._pin_counts[shard_idx]
                self._evict_if_needed_locked()

    async def update_entry(
        self, shard_idx: ShardCacheKey, entry_idx: int, value: Optional[CID]
    ) -> bool:
        """Update one cached entry and mark its shard dirty atomically."""
        async with self._cache_lock:
            shard_data = self._cache.get(shard_idx)
            if shard_data is None:
                raise RuntimeError(f"Shard {shard_idx} not found in cache")
            if shard_data[entry_idx] == value:
                return False

            old_size = self._shard_sizes[shard_idx]
            shard_data[entry_idx] = value
            new_size = self._get_shard_size(shard_data)
            self._shard_sizes[shard_idx] = new_size
            self._actual_memory_usage += new_size - old_size
            self._dirty_shards.add(shard_idx)
            self._evict_if_needed_locked()
            return True

    async def mark_dirty(self, shard_idx: ShardCacheKey) -> None:
        """Mark a shard as dirty (should not be evicted)."""
        async with self._cache_lock:
            if shard_idx in self._cache:
                self._dirty_shards.add(shard_idx)

    async def mark_clean(self, shard_idx: ShardCacheKey) -> None:
        """Mark a shard as clean (can be evicted)."""
        async with self._cache_lock:
            self._dirty_shards.discard(shard_idx)
            self._evict_if_needed_locked()

    async def discard(self, shard_idx: ShardCacheKey) -> None:
        """Remove one shard from cache and dirty tracking."""
        async with self._cache_lock:
            if shard_idx in self._cache:
                self._cache.pop(shard_idx)
                self._actual_memory_usage -= self._shard_sizes.pop(shard_idx, 0)
            self._dirty_shards.discard(shard_idx)

    async def clear(self) -> None:
        """Clear all cached data."""
        async with self._cache_lock:
            self._cache.clear()
            self._dirty_shards.clear()
            self._shard_sizes.clear()
            self._actual_memory_usage = 0

    async def __contains__(self, shard_idx: ShardCacheKey) -> bool:
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

    ``sharded_zarr_v1`` roots keep the original single global shard index for
    compatibility. ``sharded_zarr_v2`` roots keep one shard index per Zarr array
    path, allowing grouped arrays to reuse chunk coordinates without collisions.
    """

    _V1_COORDINATE_ARRAY_PREFIXES: ClassVar[frozenset[str]] = frozenset({
        "time",
        "lat",
        "lon",
        "latitude",
        "longitude",
        "forecast_reference_time",
        "step",
    })
    _V1_DEPRECATION_MESSAGE: ClassVar[str] = (
        "sharded_zarr_v1 is deprecated and will be removed in a future py-hamt "
        "release. Prefer sharded_zarr_v2 for new stores; pyramid Zarr readers "
        "should open the desired group explicitly, for example group='0'."
    )
    _V2_MULTI_GROUP_READ_MESSAGE: ClassVar[str] = (
        "sharded_zarr_v2 stores with multiple top-level groups require an "
        "explicit Zarr group when reading. Open the desired pyramid level with "
        "xr.open_zarr(..., group='0') or another available group."
    )
    _V2_WRITE_GROUP_MESSAGE: ClassVar[str] = (
        "sharded_zarr_v2 writes require an explicit Zarr group. Write the "
        "dataset with ds.to_zarr(..., group='0') or another group name."
    )

    def __init__(
        self,
        cas: ContentAddressedStore,
        read_only: bool,
        root_cid: Optional[str] = None,
        *,
        max_cache_memory_bytes: int = 100 * 1024 * 1024,  # 100MB default
        shard_read_mode: ShardReadMode = "sparse",
    ):
        """Use the async `open()` classmethod to instantiate this class."""
        super().__init__(read_only=read_only)
        if shard_read_mode not in {"full", "sparse"}:
            raise ValueError(
                f"Unsupported shard_read_mode: {shard_read_mode!r}. "
                "Expected 'full' or 'sparse'."
            )
        self.cas = cas
        self._root_cid = root_cid
        self.shard_read_mode = shard_read_mode
        self._root_obj: dict = {}
        self._manifest_version = SHARDED_ZARR_V1

        self._resize_lock = asyncio.Lock()
        self._resize_complete = asyncio.Event()
        self._resize_complete.set()
        self._write_lock = asyncio.Lock()
        self._shard_locks: DefaultDict[ShardCacheKey, asyncio.Lock] = defaultdict(
            asyncio.Lock
        )

        self._shard_data_cache = MemoryBoundedLRUCache(max_cache_memory_bytes)
        self._pending_shard_loads: Dict[ShardCacheKey, asyncio.Event] = {}
        self._metadata_read_cache: Dict[str, bytes] = {}

        self.array_indices: Dict[str, ArrayIndex] = {}
        self._primary_array_path: Optional[str] = None
        self._default_chunks_per_shard: Optional[int] = None

        self._array_shape: Tuple[int, ...] = ()
        self._chunk_shape: Tuple[int, ...] = ()
        self._chunks_per_dim: Tuple[int, ...] = ()
        self._chunks_per_shard: int = 0
        self._num_shards: int = 0
        self._total_chunks: int = 0

        self._dirty_root = False
        self._v2_pending_root_group_write = False

    @staticmethod
    def _normalize_array_path(array_path: str) -> str:
        return array_path.strip("/")

    @staticmethod
    def _array_path_from_metadata_key(key: str) -> Optional[str]:
        if key in {"zarr.json", ".zarray"}:
            return ""
        if key.endswith("/zarr.json"):
            return key[: -len("/zarr.json")]
        if key.endswith("/.zarray"):
            return key[: -len("/.zarray")]
        return None

    @staticmethod
    def _group_path_from_metadata_key(key: str) -> Optional[str]:
        if key in {"zarr.json", ".zgroup"}:
            return ""
        if key.endswith("/zarr.json"):
            return key[: -len("/zarr.json")]
        if key.endswith("/.zgroup"):
            return key[: -len("/.zgroup")]
        return None

    @staticmethod
    def _format_chunk_key(array_path: str, coords: tuple[int, ...]) -> str:
        coord_path = "/".join(str(coord) for coord in coords)
        if array_path:
            return f"{array_path}/c/{coord_path}"
        return f"c/{coord_path}"

    @staticmethod
    def _v2_group_metadata_key(group_path: str) -> str:
        return ".zgroup" if group_path == "" else f"{group_path}/.zgroup"

    @staticmethod
    def _v3_group_metadata_key(group_path: str) -> str:
        return "zarr.json" if group_path == "" else f"{group_path}/zarr.json"

    @staticmethod
    def _coords_from_linear_index(
        linear_index: int, chunks_per_dim: tuple[int, ...]
    ) -> tuple[int, ...]:
        coords: list[int] = []
        remaining = linear_index
        for stride in reversed(chunks_per_dim):
            coords.append(remaining % stride)
            remaining //= stride
        return tuple(reversed(coords))

    def __update_geometry(self) -> None:
        """Calculates legacy v1 geometric properties from the base shapes."""
        index = ArrayIndex.new(
            array_path="",
            array_shape=self._array_shape,
            chunk_shape=self._chunk_shape,
            chunks_per_shard=self._chunks_per_shard,
        )
        self._chunks_per_dim = index.chunks_per_dim
        self._total_chunks = index.total_chunks
        self._num_shards = index.num_shards

    @classmethod
    def _warn_v1_deprecated(cls, *, stacklevel: int) -> None:
        warnings.warn(
            cls._V1_DEPRECATION_MESSAGE,
            ShardedZarrV1DeprecationWarning,
            stacklevel=stacklevel,
        )

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
        manifest_version: Optional[str] = None,
        primary_array_path: str = "",
        shard_read_mode: ShardReadMode = "sparse",
    ) -> "ShardedZarrStore":
        """
        Asynchronously opens an existing ShardedZarrStore or initializes a new one.

        Shape-based creation remains the v1 compatibility path. To create a new
        path-aware v2 store, pass ``manifest_version="sharded_zarr_v2"`` or omit
        ``array_shape``/``chunk_shape`` and provide ``chunks_per_shard``.
        """
        store = cls(
            cas,
            read_only,
            root_cid,
            max_cache_memory_bytes=max_cache_memory_bytes,
            shard_read_mode=shard_read_mode,
        )
        if root_cid:
            await store._load_root_from_cid()
        elif not read_only:
            if manifest_version not in {None, SHARDED_ZARR_V1, SHARDED_ZARR_V2}:
                raise ValueError(f"Incompatible manifest version: {manifest_version}.")

            if (
                manifest_version in {None, SHARDED_ZARR_V1}
                and array_shape is None
                and chunk_shape is None
                and chunks_per_shard is None
            ):
                raise ValueError(
                    "array_shape and chunk_shape must be provided for a new store."
                )
            if manifest_version in {None, SHARDED_ZARR_V1} and (
                (array_shape is None) != (chunk_shape is None)
            ):
                raise ValueError(
                    "array_shape and chunk_shape must be provided for a new store."
                )
            if manifest_version == SHARDED_ZARR_V1 and (
                array_shape is None or chunk_shape is None
            ):
                raise ValueError(
                    "array_shape and chunk_shape must be provided for a new store."
                )

            if not isinstance(chunks_per_shard, int) or chunks_per_shard <= 0:
                raise ValueError("chunks_per_shard must be a positive integer.")

            use_v2 = manifest_version == SHARDED_ZARR_V2 or (
                array_shape is None and chunk_shape is None
            )
            if use_v2:
                if (array_shape is None) != (chunk_shape is None):
                    raise ValueError(
                        "array_shape and chunk_shape must both be provided when seeding a v2 array index."
                    )
                store._initialize_new_root_v2(
                    chunks_per_shard=chunks_per_shard,
                    array_shape=array_shape,
                    chunk_shape=chunk_shape,
                    primary_array_path=primary_array_path,
                )
            else:
                if array_shape is None or chunk_shape is None:  # pragma: no cover
                    raise ValueError(
                        "array_shape and chunk_shape must be provided for a new store."
                    )
                store._initialize_new_root(array_shape, chunk_shape, chunks_per_shard)
        else:
            raise ValueError("root_cid must be provided for a read-only store.")
        return store

    def _initialize_new_root(
        self,
        array_shape: Tuple[int, ...],
        chunk_shape: Tuple[int, ...],
        chunks_per_shard: int,
    ) -> None:
        self._warn_v1_deprecated(stacklevel=4)
        self._manifest_version = SHARDED_ZARR_V1
        self._array_shape = tuple(array_shape)
        self._chunk_shape = tuple(chunk_shape)
        self._chunks_per_shard = chunks_per_shard
        self._default_chunks_per_shard = chunks_per_shard

        self.__update_geometry()

        self._root_obj = {
            "manifest_version": SHARDED_ZARR_V1,
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
        self.array_indices = {
            "": ArrayIndex(
                array_path="",
                array_shape=self._array_shape,
                chunk_shape=self._chunk_shape,
                chunks_per_shard=self._chunks_per_shard,
                shard_cids=self._root_obj["chunks"]["shard_cids"],
            )
        }
        self._primary_array_path = ""
        self._dirty_root = True

    def _initialize_new_root_v2(
        self,
        *,
        chunks_per_shard: int,
        array_shape: Optional[Tuple[int, ...]] = None,
        chunk_shape: Optional[Tuple[int, ...]] = None,
        primary_array_path: str = "",
    ) -> None:
        self._manifest_version = SHARDED_ZARR_V2
        self._default_chunks_per_shard = chunks_per_shard
        self._root_obj = {
            "manifest_version": SHARDED_ZARR_V2,
            "store_type": "py_hamt.sharded_zarr",
            "zarr_format": 3,
            "sharding_config": {
                "chunks_per_shard": chunks_per_shard,
                "order": "C",
            },
            "metadata": {},
            "arrays": {},
        }
        self.array_indices = {}
        self._primary_array_path = None
        self._array_shape = ()
        self._chunk_shape = ()
        self._chunks_per_dim = ()
        self._chunks_per_shard = chunks_per_shard
        self._num_shards = 0
        self._total_chunks = 0

        if array_shape is not None and chunk_shape is not None:
            self._register_or_update_array_index(
                array_path=primary_array_path,
                array_shape=tuple(array_shape),
                chunk_shape=tuple(chunk_shape),
                chunks_per_shard=chunks_per_shard,
            )
        self._dirty_root = True

    async def _load_root_from_cid(self) -> None:
        root_bytes = await self.cas.load(self._root_cid)
        try:
            decoded_root = dag_cbor.decode(root_bytes)
            if not isinstance(decoded_root, dict):
                raise ValueError("Root object is not a valid dictionary.")
            self._root_obj = decoded_root
        except Exception as e:
            raise ValueError(f"Failed to decode root object: {e}") from e

        manifest_version = self._root_obj.get("manifest_version")
        if manifest_version == SHARDED_ZARR_V1:
            self._load_v1_root()
        elif manifest_version == SHARDED_ZARR_V2:
            self._load_v2_root()
        else:
            raise ValueError(
                f"Incompatible manifest version: {manifest_version!r}. Expected '{SHARDED_ZARR_V1}' or '{SHARDED_ZARR_V2}'."
            )

    def _load_v1_root(self) -> None:
        self._warn_v1_deprecated(stacklevel=5)
        if "chunks" not in self._root_obj:
            raise ValueError("Root object is not a valid dictionary with 'chunks' key.")
        chunk_info = self._root_obj["chunks"]
        if not isinstance(chunk_info.get("shard_cids"), list):
            raise ValueError("shard_cids is not a list.")

        self._manifest_version = SHARDED_ZARR_V1
        self._array_shape = tuple(chunk_info["array_shape"])
        self._chunk_shape = tuple(chunk_info["chunk_shape"])
        self._chunks_per_shard = chunk_info["sharding_config"]["chunks_per_shard"]
        self._default_chunks_per_shard = self._chunks_per_shard

        self.__update_geometry()

        if len(chunk_info["shard_cids"]) != self._num_shards:
            raise ValueError(
                f"Inconsistent number of shards. Expected {self._num_shards}, found {len(chunk_info['shard_cids'])}."
            )
        self.array_indices = {
            "": ArrayIndex(
                array_path="",
                array_shape=self._array_shape,
                chunk_shape=self._chunk_shape,
                chunks_per_shard=self._chunks_per_shard,
                shard_cids=chunk_info["shard_cids"],
            )
        }
        primary_array_path = chunk_info.get("primary_array_path", "")
        self._primary_array_path = (
            self._normalize_array_path(primary_array_path)
            if isinstance(primary_array_path, str)
            else ""
        )

    def _load_v2_root(self) -> None:
        metadata = self._root_obj.get("metadata")
        arrays = self._root_obj.get("arrays")
        if not isinstance(metadata, dict) or not isinstance(arrays, dict):
            raise ValueError(
                "Root object is not a valid v2 dictionary with 'metadata' and 'arrays' keys."
            )

        self._manifest_version = SHARDED_ZARR_V2
        self.array_indices = {}
        self._primary_array_path = None
        root_sharding_config = self._root_obj.get("sharding_config", {})
        if isinstance(root_sharding_config, dict):
            self._default_chunks_per_shard = root_sharding_config.get(
                "chunks_per_shard"
            )
        else:
            self._default_chunks_per_shard = None

        for array_path, array_manifest in arrays.items():
            if not isinstance(array_path, str) or not isinstance(array_manifest, dict):
                raise ValueError("arrays must map string paths to dictionaries.")
            try:
                array_index = ArrayIndex.from_manifest(array_path, array_manifest)
            except ValueError as exc:
                if str(exc).startswith("Inconsistent number of shards"):
                    raise ValueError(
                        f"Inconsistent number of shards for array '{array_path}'. {exc}"
                    ) from exc
                raise
            self.array_indices[array_index.array_path] = array_index
            if self._primary_array_path is None:
                self._primary_array_path = array_index.array_path

        if self.array_indices:
            primary_index = self.array_indices[self._primary_array_path or ""]
            self._default_chunks_per_shard = primary_index.chunks_per_shard
            self._set_legacy_geometry_from_index(primary_index)
        else:
            self._array_shape = ()
            self._chunk_shape = ()
            self._chunks_per_dim = ()
            self._chunks_per_shard = 0
            self._num_shards = 0
            self._total_chunks = 0

    def _set_legacy_geometry_from_index(self, array_index: ArrayIndex) -> None:
        self._array_shape = array_index.array_shape
        self._chunk_shape = array_index.chunk_shape
        self._chunks_per_dim = array_index.chunks_per_dim
        self._chunks_per_shard = array_index.chunks_per_shard
        self._num_shards = array_index.num_shards
        self._total_chunks = array_index.total_chunks

    def _sync_arrays_to_root(self) -> None:
        if self._manifest_version == SHARDED_ZARR_V2:
            self._root_obj["arrays"] = {
                array_path: array_index.to_manifest()
                for array_path, array_index in self.array_indices.items()
            }

    def _register_or_update_array_index(
        self,
        *,
        array_path: str,
        array_shape: tuple[int, ...],
        chunk_shape: tuple[int, ...],
        chunks_per_shard: Optional[int] = None,
    ) -> ArrayIndex:
        normalized_path = self._normalize_array_path(array_path)
        if chunks_per_shard is None:
            chunks_per_shard = self._default_chunks_per_shard
        if chunks_per_shard is None:
            raise RuntimeError("Store is missing a default chunks_per_shard value.")

        existing = self.array_indices.get(normalized_path)
        if existing is None:
            array_index = ArrayIndex.new(
                array_path=normalized_path,
                array_shape=array_shape,
                chunk_shape=chunk_shape,
                chunks_per_shard=chunks_per_shard,
            )
            self.array_indices[normalized_path] = array_index
            if self._primary_array_path is None:
                self._primary_array_path = normalized_path
                self._set_legacy_geometry_from_index(array_index)
        else:
            new_chunk_shape = tuple(chunk_shape)
            if existing.chunk_shape != new_chunk_shape:
                raise ValueError(
                    f"Cannot change chunk_shape for existing array index '{normalized_path}'."
                )
            existing.resize(tuple(array_shape))
            array_index = existing

        if self._primary_array_path == normalized_path:
            self._set_legacy_geometry_from_index(array_index)
        self._sync_arrays_to_root()
        self._dirty_root = True
        return array_index

    def _v2_top_level_groups(self) -> set[str]:
        if self._manifest_version != SHARDED_ZARR_V2:
            return set()
        return {
            array_path.split("/", 1)[0]
            for array_path in self.array_indices
            if "/" in array_path
        }

    def _v2_is_grouped_only(self) -> bool:
        if self._manifest_version != SHARDED_ZARR_V2 or not self.array_indices:
            return False
        return all("/" in array_path for array_path in self.array_indices)

    def _v2_default_group_for_root_read(self) -> Optional[str]:
        if not self._v2_is_grouped_only():
            return None
        groups = self._v2_top_level_groups()
        if len(groups) != 1:
            return None
        return next(iter(groups))

    def _v2_requires_explicit_group_for_root_read(self) -> bool:
        return self._v2_is_grouped_only() and len(self._v2_top_level_groups()) > 1

    def _v2_effective_read_key(self, key: str) -> str:
        default_group = self._v2_default_group_for_root_read()
        if default_group is None:
            return key

        normalized_key = key.strip("/")
        if normalized_key in {"zarr.json", ".zgroup", ".zattrs", ".zmetadata", ""}:
            return key
        if normalized_key == default_group or normalized_key.startswith(
            f"{default_group}/"
        ):
            return key
        return f"{default_group}/{normalized_key}"

    def _v2_effective_list_dir_prefix(self, normalized_prefix: str) -> str:
        default_group = self._v2_default_group_for_root_read()
        if default_group is None:
            return normalized_prefix
        if normalized_prefix == "":
            return default_group
        if normalized_prefix == default_group or normalized_prefix.startswith(
            f"{default_group}/"
        ):
            return normalized_prefix
        return f"{default_group}/{normalized_prefix}"

    def _strip_v2_root_consolidated_metadata(self, key: str, raw_data: bytes) -> bytes:
        if key != "zarr.json" or not self._v2_is_grouped_only():
            return raw_data

        metadata_json = self._decode_metadata_json(raw_data)
        if (
            metadata_json is None
            or metadata_json.get("node_type") != "group"
            or "consolidated_metadata" not in metadata_json
        ):
            return raw_data

        metadata_json.pop("consolidated_metadata")
        return json.dumps(metadata_json).encode("utf-8")

    @staticmethod
    def _v2_path_has_group(array_path: str) -> bool:
        normalized_path = ShardedZarrStore._normalize_array_path(array_path)
        return "/" in normalized_path

    def _raise_if_v2_write_without_group(self, key: str, raw_data: bytes) -> None:
        if self._manifest_version != SHARDED_ZARR_V2:
            return

        metadata_json = self._decode_metadata_json(raw_data)
        if metadata_json is None:
            return

        group_path = self._group_path_from_metadata_key(key)
        metadata_path = self._array_path_from_metadata_key(key)
        array_metadata = self._extract_array_metadata(metadata_json)
        is_group_metadata = group_path is not None and array_metadata is None

        if is_group_metadata and group_path == "":
            self._v2_pending_root_group_write = True
            return
        if is_group_metadata:
            self._v2_pending_root_group_write = False
            return

        if metadata_path is None or array_metadata is None:
            return

        if self._v2_path_has_group(metadata_path):
            self._v2_pending_root_group_write = False
            return

        if self._v2_pending_root_group_write:
            self._v2_pending_root_group_write = False
            raise ValueError(self._V2_WRITE_GROUP_MESSAGE)

    async def _snapshot_shards_for_resize(
        self,
        array_index: ArrayIndex,
    ) -> dict[int, list[Optional[CID]]]:
        shards_by_index: dict[int, list[Optional[CID]]] = {}
        for shard_idx, shard_cid_obj in enumerate(array_index.shard_cids):
            cache_key = self._cache_key(array_index.array_path, shard_idx)
            shard_lock = self._shard_locks[cache_key]
            async with shard_lock:
                shard_data = await self._shard_data_cache.get(cache_key)
                if shard_data is None and shard_cid_obj is not None:
                    await self._fetch_and_cache_full_shard(
                        cache_key,
                        shard_idx,
                        str(shard_cid_obj),
                        array_index.chunks_per_shard,
                    )
                    shard_data = await self._shard_data_cache.get(cache_key)
                    if shard_data is None:  # pragma: no cover
                        raise RuntimeError(f"Failed to load shard {shard_idx}")
                if shard_data is not None:
                    shards_by_index[shard_idx] = list(shard_data)
        return shards_by_index

    @staticmethod
    def _remap_shards_for_resize(
        old_shards_by_index: dict[int, list[Optional[CID]]],
        old_chunks_per_dim: tuple[int, ...],
        old_total_chunks: int,
        new_array_index: ArrayIndex,
    ) -> dict[int, list[Optional[CID]]]:
        new_shards_by_index = {
            shard_idx: [None] * new_array_index.chunks_per_shard
            for shard_idx in range(new_array_index.num_shards)
        }
        for old_shard_idx, old_shard in old_shards_by_index.items():
            for old_index_in_shard, pointer_cid_obj in enumerate(old_shard):
                if pointer_cid_obj is None:
                    continue
                old_linear_index = (
                    old_shard_idx * new_array_index.chunks_per_shard
                    + old_index_in_shard
                )
                if old_linear_index >= old_total_chunks:
                    continue
                coords = ShardedZarrStore._coords_from_linear_index(
                    old_linear_index, old_chunks_per_dim
                )
                if any(
                    coord >= chunks
                    for coord, chunks in zip(
                        coords, new_array_index.chunks_per_dim, strict=True
                    )
                ):
                    continue

                new_linear_index = ShardedZarrStore._get_linear_chunk_index_for_index(
                    coords, new_array_index
                )
                new_shard_idx, new_index_in_shard = (
                    ShardedZarrStore._get_shard_info_for_index(
                        new_linear_index, new_array_index
                    )
                )
                new_shards_by_index[new_shard_idx][new_index_in_shard] = pointer_cid_obj
        return new_shards_by_index

    async def _replace_shards_after_resize(
        self,
        array_index: ArrayIndex,
        old_num_shards: int,
        old_shard_cids: list[Optional[CID]],
        old_shards_by_index: dict[int, list[Optional[CID]]],
        new_shards_by_index: dict[int, list[Optional[CID]]],
    ) -> None:
        async with self._shard_data_cache._cache_lock:
            dirty_cache_keys = set(self._shard_data_cache._dirty_shards)

        for shard_idx in range(array_index.num_shards):
            cache_key = self._cache_key(array_index.array_path, shard_idx)
            shard_lock = self._shard_locks[cache_key]
            new_shard = new_shards_by_index[shard_idx]
            old_shard = old_shards_by_index.get(shard_idx)
            old_shard_cid = (
                old_shard_cids[shard_idx] if shard_idx < len(old_shard_cids) else None
            )
            async with shard_lock:
                if all(pointer_cid_obj is None for pointer_cid_obj in new_shard):
                    array_index.shard_cids[shard_idx] = None
                    await self._shard_data_cache.discard(cache_key)
                elif old_shard == new_shard and (
                    old_shard_cid is not None or cache_key in dirty_cache_keys
                ):
                    array_index.shard_cids[shard_idx] = old_shard_cid
                else:
                    array_index.shard_cids[shard_idx] = None
                    await self._shard_data_cache.put(
                        cache_key, new_shard, is_dirty=True
                    )

        for shard_idx in range(array_index.num_shards, old_num_shards):
            cache_key = self._cache_key(array_index.array_path, shard_idx)
            shard_lock = self._shard_locks[cache_key]
            async with shard_lock:
                await self._shard_data_cache.discard(cache_key)

    async def _resize_array_index(
        self, array_index: ArrayIndex, new_shape: tuple[int, ...]
    ) -> None:
        old_num_shards = array_index.num_shards
        old_total_chunks = array_index.total_chunks
        old_chunks_per_dim = array_index.chunks_per_dim
        old_shard_cids = list(array_index.shard_cids)
        old_shards_by_index = await self._snapshot_shards_for_resize(array_index)
        array_index.resize(tuple(new_shape))
        new_shards_by_index = self._remap_shards_for_resize(
            old_shards_by_index,
            old_chunks_per_dim,
            old_total_chunks,
            array_index,
        )
        await self._replace_shards_after_resize(
            array_index,
            old_num_shards,
            old_shard_cids,
            old_shards_by_index,
            new_shards_by_index,
        )
        if self._primary_array_path == array_index.array_path:
            self._set_legacy_geometry_from_index(array_index)
        self._sync_arrays_to_root()
        self._dirty_root = True

    async def _resize_array_index_guarded(
        self, array_index: ArrayIndex, new_shape: tuple[int, ...]
    ) -> None:
        async with self._resize_lock:
            self._resize_complete.clear()
            try:
                await self._resize_array_index(array_index, new_shape)
            finally:
                self._resize_complete.set()

    @staticmethod
    def _decode_metadata_json(raw_data: bytes) -> Optional[dict]:
        try:
            decoded = json.loads(raw_data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
        return decoded if isinstance(decoded, dict) else None

    @staticmethod
    def _extract_array_metadata(
        metadata_json: dict,
    ) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
        shape = metadata_json.get("shape")
        if shape is None:
            return None

        chunk_shape = None
        chunk_grid = metadata_json.get("chunk_grid")
        if isinstance(chunk_grid, dict):
            configuration = chunk_grid.get("configuration")
            if isinstance(configuration, dict):
                chunk_shape = configuration.get("chunk_shape")

        if chunk_shape is None:
            chunk_shape = metadata_json.get("chunks")

        if chunk_shape is None:
            return None

        return tuple(int(dim) for dim in shape), tuple(int(dim) for dim in chunk_shape)

    def _infer_v1_migration_source_array_path(self, primary_array_path: str) -> str:
        metadata = self._root_obj.get("metadata", {})
        candidates = [primary_array_path]
        primary_leaf = primary_array_path.rsplit("/", 1)[-1]
        if primary_leaf not in candidates:
            candidates.append(primary_leaf)
        candidates.append("")

        for candidate in candidates:
            metadata_keys = (
                ("zarr.json", ".zarray")
                if candidate == ""
                else (f"{candidate}/zarr.json", f"{candidate}/.zarray")
            )
            if any(key in metadata for key in metadata_keys):
                return candidate
        return primary_leaf

    @staticmethod
    def _rewrite_v1_metadata_key_for_migration(
        key: str, source_array_path: str, primary_array_path: str
    ) -> str:
        parent_path = (
            primary_array_path.rsplit("/", 1)[0] if "/" in primary_array_path else ""
        )

        if source_array_path:
            source_prefix = f"{source_array_path}/"
            if key.startswith(source_prefix):
                return f"{primary_array_path}/{key[len(source_prefix) :]}"
        elif key.startswith("c/"):
            return f"{primary_array_path}/{key}"
        elif key in {"zarr.json", ".zarray"}:
            return f"{primary_array_path}/{key}"

        if parent_path and "/" in key and not key.startswith(f"{parent_path}/"):
            return f"{parent_path}/{key}"
        return key

    async def _add_missing_group_metadata(
        self, metadata: dict[str, IPLDKind], array_path: str
    ) -> None:
        parts = array_path.split("/")
        group_paths = ["/".join(parts[:idx]) for idx in range(len(parts))]
        uses_zarr_v2_metadata = f"{array_path}/.zarray" in metadata
        if uses_zarr_v2_metadata:
            group_metadata_key = self._v2_group_metadata_key
            group_metadata = json.dumps({"zarr_format": 2}).encode("utf-8")
        else:
            group_metadata_key = self._v3_group_metadata_key
            group_metadata = json.dumps({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {},
            }).encode("utf-8")

        for group_path in group_paths:
            metadata_key = group_metadata_key(group_path)
            if metadata_key in metadata:
                if not uses_zarr_v2_metadata:
                    await self._strip_consolidated_metadata(metadata, metadata_key)
                continue
            metadata[metadata_key] = await self.cas.save(group_metadata, codec="raw")

    async def _strip_consolidated_metadata(
        self, metadata: dict[str, IPLDKind], metadata_key: str
    ) -> None:
        raw_metadata = await self.cas.load(str(metadata[metadata_key]))
        metadata_json = self._decode_metadata_json(raw_metadata)
        if (
            metadata_json is None
            or metadata_json.get("node_type") != "group"
            or "consolidated_metadata" not in metadata_json
        ):
            return

        metadata_json.pop("consolidated_metadata")
        metadata[metadata_key] = await self.cas.save(
            json.dumps(metadata_json).encode("utf-8"), codec="raw"
        )

    async def _register_array_metadata_from_bytes(
        self, key: str, raw_data: bytes
    ) -> None:
        array_path = self._array_path_from_metadata_key(key)
        if array_path is None:
            return

        metadata_json = self._decode_metadata_json(raw_data)
        if metadata_json is None:
            return

        array_metadata = self._extract_array_metadata(metadata_json)
        if array_metadata is None and self._manifest_version == SHARDED_ZARR_V2:
            return
        if array_metadata is None:
            shape = metadata_json.get("shape")
            if shape is None:
                return
            new_array_shape = tuple(int(dim) for dim in shape)
            new_chunk_shape = self._chunk_shape
        else:
            new_array_shape, new_chunk_shape = array_metadata

        if self._manifest_version == SHARDED_ZARR_V2:
            normalized_path = self._normalize_array_path(array_path)
            existing = self.array_indices.get(normalized_path)
            if existing is None:
                self._register_or_update_array_index(
                    array_path=array_path,
                    array_shape=new_array_shape,
                    chunk_shape=new_chunk_shape,
                )
            else:
                if existing.chunk_shape != new_chunk_shape:
                    raise ValueError(
                        f"Cannot change chunk_shape for existing array index '{normalized_path}'."
                    )
                if existing.array_shape != new_array_shape:
                    await self._resize_array_index_guarded(existing, new_array_shape)
            return

        if (
            len(new_array_shape) == len(self._array_shape)
            and new_array_shape != self._array_shape
        ):
            await self._resize_store_unlocked(new_shape=new_array_shape)

    async def _ensure_v2_parent_group_metadata(self, key: str) -> None:
        if self._manifest_version != SHARDED_ZARR_V2:
            return

        metadata_path = self._array_path_from_metadata_key(key)
        if metadata_path is None:
            return

        if key == ".zarray" or key.endswith("/.zarray"):
            group_metadata_key = self._v2_group_metadata_key
            group_metadata = json.dumps({"zarr_format": 2}).encode("utf-8")
        else:
            group_metadata_key = self._v3_group_metadata_key
            group_metadata = json.dumps({
                "zarr_format": 3,
                "node_type": "group",
                "attributes": {},
            }).encode("utf-8")

        normalized_path = self._normalize_array_path(metadata_path)
        parent_paths = [""]
        if normalized_path:
            parts = normalized_path.split("/")
            parent_paths.extend("/".join(parts[:idx]) for idx in range(1, len(parts)))

        for parent_path in parent_paths:
            metadata_key = group_metadata_key(parent_path)
            if metadata_key in self._root_obj["metadata"]:
                continue
            metadata_cid = await self.cas.save(group_metadata, codec="raw")
            self._root_obj["metadata"][metadata_key] = metadata_cid
            self._metadata_read_cache[metadata_key] = group_metadata
            self._dirty_root = True

    async def _fetch_and_cache_full_shard(
        self,
        cache_key: ShardCacheKey,
        shard_idx: int,
        shard_cid: IPLDKind,
        expected_entries: int,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """
        Fetch a shard from CAS and cache it, with retry logic for transient errors.
        """
        for attempt in range(max_retries):
            try:
                shard_data_bytes = await self.cas.load(shard_cid)
                decoded_shard = dag_cbor.decode(shard_data_bytes)
                if not isinstance(decoded_shard, list):
                    raise TypeError(f"Shard {shard_idx} did not decode to a list.")
                if len(decoded_shard) != expected_entries:
                    raise ValueError(
                        f"Shard {shard_idx} contains {len(decoded_shard)} entries; expected {expected_entries}."
                    )
                shard_data: List[Optional[CID]] = []
                for item in decoded_shard:
                    if item is not None and not isinstance(item, CID):
                        raise TypeError(f"Shard {shard_idx} contains a non-CID entry.")
                    shard_data.append(item)
                await self._shard_data_cache.put(cache_key, shard_data)
                if cache_key in self._pending_shard_loads:
                    self._pending_shard_loads[cache_key].set()
                    del self._pending_shard_loads[cache_key]
                return
            except (ConnectionError, TimeoutError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (2**attempt))
                    continue
                raise RuntimeError(
                    f"Failed to fetch shard {shard_idx} after {max_retries} attempts: {e}"
                ) from e

    async def _load_sparse_shard_entry(
        self,
        cache_key: ShardCacheKey,
        shard_idx: int,
        shard_cid: IPLDKind,
        index_in_shard: int,
        expected_entries: int,
    ) -> Optional[CID]:
        if await self._shard_data_cache.get(cache_key) is not None:
            return None

        shard_data_bytes = await self.cas.load(shard_cid)
        entry_count, _ = _read_cbor_list_header(shard_data_bytes)
        if entry_count != expected_entries:
            raise ValueError(
                f"Shard {shard_idx} contains {entry_count} entries; expected {expected_entries}."
            )
        return decode_shard_entry(shard_data_bytes, index_in_shard)

    def _parse_chunk_key(self, key: str) -> Optional[ChunkKey]:
        if key.endswith(ZARR_METADATA_SUFFIXES):
            return None

        chunk_marker = "/c/"
        marker_idx = key.rfind(chunk_marker)
        if marker_idx != -1:
            array_path = key[:marker_idx]
            coord_part = key[marker_idx + len(chunk_marker) :]
        elif key.startswith("c/"):
            if self._manifest_version == SHARDED_ZARR_V1:
                return None
            array_path = ""
            coord_part = key[len("c/") :]
        else:
            return self._parse_classic_v2_chunk_key(key)

        normalized_path = self._normalize_array_path(array_path)
        if self._manifest_version == SHARDED_ZARR_V1:
            actual_array_name = (
                normalized_path.split("/")[-1] if normalized_path else ""
            )
            if actual_array_name in self._V1_COORDINATE_ARRAY_PREFIXES:
                return None
            recorded_path = self._root_obj.get("chunks", {}).get("primary_array_path")
            if isinstance(
                recorded_path, str
            ) and normalized_path != self._normalize_array_path(recorded_path):
                return None

        parts = coord_part.split("/")
        try:
            coords = tuple(map(int, parts))
        except ValueError:
            classic_chunk = self._parse_classic_v2_chunk_key(key)
            if classic_chunk is not None:
                return classic_chunk
            raise

        if self._manifest_version == SHARDED_ZARR_V1:
            self._validate_chunk_coords(coords, self.array_indices[""])
        elif normalized_path in self.array_indices:
            self._validate_chunk_coords(coords, self.array_indices[normalized_path])

        return ChunkKey(array_path=normalized_path, coords=coords)

    def _parse_classic_v2_chunk_key(self, key: str) -> Optional[ChunkKey]:
        if self._manifest_version != SHARDED_ZARR_V2 or not self.array_indices:
            return None

        dotted_chunk = self._parse_classic_dotted_v2_chunk_key(key)
        if dotted_chunk is not None:
            return dotted_chunk

        for array_path, array_index in sorted(
            self.array_indices.items(), key=lambda item: len(item[0]), reverse=True
        ):
            prefix = f"{array_path}/" if array_path else ""
            if prefix:
                if not key.startswith(prefix):
                    continue
                coord_part = key[len(prefix) :]
            else:
                coord_part = key

            parts = coord_part.split("/")
            if len(parts) != len(array_index.chunks_per_dim):
                continue
            if not all(part.isdecimal() for part in parts):
                continue
            coords = tuple(int(part) for part in parts)
            self._validate_chunk_coords(coords, array_index)
            return ChunkKey(array_path=array_path, coords=coords)
        return None

    def _parse_classic_dotted_v2_chunk_key(self, key: str) -> Optional[ChunkKey]:
        array_path, _, coord_part = key.rpartition("/")
        if "." not in coord_part:
            return None

        parts = coord_part.split(".")
        if not parts or not all(part.isdecimal() for part in parts):
            return None

        normalized_path = self._normalize_array_path(array_path)
        array_index = self.array_indices.get(normalized_path)
        if array_index is None or len(parts) != len(array_index.chunks_per_dim):
            return None

        coords = tuple(int(part) for part in parts)
        self._validate_chunk_coords(coords, array_index)
        return ChunkKey(array_path=normalized_path, coords=coords)

    @staticmethod
    def _validate_chunk_coords(
        chunk_coords: tuple[int, ...], array_index: ArrayIndex
    ) -> None:
        if len(chunk_coords) != len(array_index.chunks_per_dim):
            raise IndexError("tuple index out of range")
        for i, c_coord in enumerate(chunk_coords):
            if not (0 <= c_coord < array_index.chunks_per_dim[i]):
                raise IndexError(
                    f"Chunk coordinate {c_coord} at dimension {i} is out of bounds for dimension size {array_index.chunks_per_dim[i]}."
                )

    def _get_linear_chunk_index(self, chunk_coords: Tuple[int, ...]) -> int:
        return self._get_linear_chunk_index_for_index(
            tuple(chunk_coords), self.array_indices[""]
        )

    @staticmethod
    def _get_linear_chunk_index_for_index(
        chunk_coords: tuple[int, ...], array_index: ArrayIndex
    ) -> int:
        linear_index = 0
        multiplier = 1
        for i in reversed(range(len(array_index.chunks_per_dim))):
            linear_index += chunk_coords[i] * multiplier
            multiplier *= array_index.chunks_per_dim[i]
        return linear_index

    def _get_shard_info(self, linear_chunk_index: int) -> Tuple[int, int]:
        shard_idx = linear_chunk_index // self._chunks_per_shard
        index_in_shard = linear_chunk_index % self._chunks_per_shard
        return shard_idx, index_in_shard

    @staticmethod
    def _get_shard_info_for_index(
        linear_chunk_index: int, array_index: ArrayIndex
    ) -> Tuple[int, int]:
        shard_idx = linear_chunk_index // array_index.chunks_per_shard
        index_in_shard = linear_chunk_index % array_index.chunks_per_shard
        return shard_idx, index_in_shard

    def _array_index_for_path(self, array_path: Optional[str]) -> ArrayIndex:
        if self._manifest_version == SHARDED_ZARR_V1:
            return self.array_indices[""]

        normalized_path = self._normalize_array_path(array_path or "")
        try:
            return self.array_indices[normalized_path]
        except KeyError as exc:
            raise KeyError(
                f"No array index registered for chunk path '{normalized_path}'."
            ) from exc

    def _cache_key(self, array_path: Optional[str], shard_idx: int) -> ShardCacheKey:
        if self._manifest_version == SHARDED_ZARR_V1:
            return shard_idx
        return (self._normalize_array_path(array_path or ""), shard_idx)

    def _map_byte_request(
        self, byte_range: Optional[zarr.abc.store.ByteRequest]
    ) -> tuple[Optional[int], Optional[int], Optional[int]]:
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
        return req_offset, req_length, req_suffix

    async def _get_legacy_metadata_chunk(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: Optional[zarr.abc.store.ByteRequest],
    ) -> Optional[zarr.core.buffer.Buffer]:
        metadata_cid_obj = self._root_obj["metadata"].get(key)
        if metadata_cid_obj is None:
            return None
        req_offset, req_length, req_suffix = self._map_byte_request(byte_range)
        data = await self.cas.load(
            metadata_cid_obj,
            offset=req_offset,
            length=req_length,
            suffix=req_suffix,
        )
        return prototype.buffer.from_bytes(data)

    async def _load_or_initialize_shard_cache(
        self, shard_idx: int, array_path: Optional[str] = None
    ) -> List[Optional[CID]]:
        """Return a shard after keeping it pinned throughout cache population."""
        array_index = self._array_index_for_path(array_path)
        cache_key = self._cache_key(array_index.array_path, shard_idx)
        async with self._shard_data_cache.pin(cache_key):
            return await self._load_or_initialize_shard_cache_pinned(
                shard_idx, array_path
            )

    async def _load_or_initialize_shard_cache_pinned(
        self, shard_idx: int, array_path: Optional[str] = None
    ) -> List[Optional[CID]]:
        """
        Load a shard into the cache or initialize an empty shard if it doesn't exist.
        """
        started_at = time.perf_counter()
        array_index = self._array_index_for_path(array_path)
        cache_key = self._cache_key(array_index.array_path, shard_idx)

        cached_shard = await self._shard_data_cache.get(cache_key)
        if cached_shard is not None:
            instrumentation.record_shard_load(
                shard_idx=shard_idx,
                cache_hit=True,
                seconds=time.perf_counter() - started_at,
                entries=len(cached_shard),
            )
            return cached_shard

        if cache_key in self._pending_shard_loads:
            try:
                await asyncio.wait_for(
                    self._pending_shard_loads[cache_key].wait(), timeout=60.0
                )
                cached_shard = await self._shard_data_cache.get(cache_key)
                if cached_shard is not None:
                    return cached_shard
                raise RuntimeError(
                    f"Shard {shard_idx} not found in cache after pending load completed."
                )
            except asyncio.TimeoutError as exc:
                if cache_key in self._pending_shard_loads:
                    self._pending_shard_loads[cache_key].set()
                    del self._pending_shard_loads[cache_key]
                raise RuntimeError(
                    f"Timeout waiting for shard {shard_idx} to load."
                ) from exc

        if not (0 <= shard_idx < array_index.num_shards):
            raise ValueError(f"Shard index {shard_idx} out of bounds.")

        shard_cid_obj = array_index.shard_cids[shard_idx]
        if shard_cid_obj:
            self._pending_shard_loads[cache_key] = asyncio.Event()
            try:
                await self._fetch_and_cache_full_shard(
                    cache_key, shard_idx, shard_cid_obj, array_index.chunks_per_shard
                )
            finally:
                pending_load = self._pending_shard_loads.pop(cache_key, None)
                if pending_load is not None:
                    pending_load.set()
        else:
            empty_shard: List[Optional[CID]] = [None] * array_index.chunks_per_shard
            await self._shard_data_cache.put(cache_key, empty_shard)

        result = await self._shard_data_cache.get(cache_key)
        if result is None:
            raise RuntimeError(f"Failed to load or initialize shard {shard_idx}")
        instrumentation.record_shard_load(
            shard_idx=shard_idx,
            cache_hit=False,
            seconds=time.perf_counter() - started_at,
            entries=len(result),
        )
        return result

    @asynccontextmanager
    async def _use_shard(
        self, shard_idx: int, array_path: Optional[str] = None
    ) -> AsyncIterator[List[Optional[CID]]]:
        """Yield a pinned shard while serializing access to its contents."""
        array_index = self._array_index_for_path(array_path)
        cache_key = self._cache_key(array_index.array_path, shard_idx)
        async with self._shard_data_cache.pin(cache_key):
            async with self._shard_locks[cache_key]:
                yield await self._load_or_initialize_shard_cache(
                    shard_idx, array_index.array_path
                )

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
        Return this store (if the flag already matches) or a shallow clone with
        the requested read-only status.
        """
        if read_only == self.read_only:
            return self

        clone = type(self).__new__(type(self))

        clone.cas = self.cas
        clone._root_cid = self._root_cid
        clone._root_obj = self._root_obj
        clone._manifest_version = self._manifest_version
        clone.shard_read_mode = self.shard_read_mode

        clone._resize_lock = self._resize_lock
        clone._resize_complete = self._resize_complete
        clone._write_lock = self._write_lock
        clone._shard_locks = self._shard_locks

        clone._shard_data_cache = self._shard_data_cache
        clone._pending_shard_loads = self._pending_shard_loads
        clone._metadata_read_cache = self._metadata_read_cache

        clone.array_indices = self.array_indices
        clone._primary_array_path = self._primary_array_path
        clone._default_chunks_per_shard = self._default_chunks_per_shard

        clone._array_shape = self._array_shape
        clone._chunk_shape = self._chunk_shape
        clone._chunks_per_dim = self._chunks_per_dim
        clone._chunks_per_shard = self._chunks_per_shard
        clone._num_shards = self._num_shards
        clone._total_chunks = self._total_chunks

        clone._dirty_root = self._dirty_root
        clone._v2_pending_root_group_write = self._v2_pending_root_group_write

        zarr.abc.store.Store.__init__(clone, read_only=read_only)
        return clone

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ShardedZarrStore):
            return False
        return self._root_cid == other._root_cid

    async def flush(self) -> str:
        async with self._write_lock:
            return await self._flush_unlocked()

    async def _flush_unlocked(self) -> str:
        async with self._shard_data_cache._cache_lock:
            dirty_shards = list(self._shard_data_cache._dirty_shards)
        if dirty_shards:
            for cache_key in sorted(dirty_shards, key=str):
                async with self._shard_data_cache.pin(cache_key):
                    shard_lock = self._shard_locks[cache_key]
                    async with shard_lock:
                        shard_data_list = await self._shard_data_cache.get(cache_key)
                        if shard_data_list is None:
                            raise RuntimeError(
                                f"Dirty shard {cache_key} not found in cache"
                            )

                        shard_data_bytes = dag_cbor.encode(
                            cast(IPLDKind, shard_data_list)
                        )
                        new_shard_cid_obj = await self.cas.save(
                            shard_data_bytes,
                            codec="dag-cbor",
                        )
                        if not isinstance(new_shard_cid_obj, CID):  # pragma: no cover
                            raise TypeError(
                                "ShardedZarrStore requires CAS.save to return CIDs."
                            )

                        if self._manifest_version == SHARDED_ZARR_V1:
                            if not isinstance(cache_key, int):  # pragma: no cover
                                raise TypeError("v1 shard cache keys must be integers.")
                            shard_idx = int(cache_key)
                            if (
                                self._root_obj["chunks"]["shard_cids"][shard_idx]
                                != new_shard_cid_obj
                            ):
                                self._root_obj["chunks"]["shard_cids"][shard_idx] = (
                                    new_shard_cid_obj
                                )
                                self.array_indices[""].shard_cids[shard_idx] = (
                                    new_shard_cid_obj
                                )
                                self._dirty_root = True
                        else:
                            if isinstance(cache_key, int):  # pragma: no cover
                                raise TypeError(
                                    "v2 shard cache keys must include array paths."
                                )
                            array_path, shard_idx = cache_key
                            array_index = self.array_indices[array_path]
                            if array_index.shard_cids[shard_idx] != new_shard_cid_obj:
                                array_index.shard_cids[shard_idx] = new_shard_cid_obj
                                self._dirty_root = True
                                self._sync_arrays_to_root()

                        await self._shard_data_cache.mark_clean(cache_key)

        if self._dirty_root:
            self._root_obj["metadata"] = {
                k: (CID.decode(v) if isinstance(v, str) else v)
                for k, v in self._root_obj["metadata"].items()
            }
            self._sync_arrays_to_root()
            root_obj_bytes = dag_cbor.encode(self._root_obj)
            new_root_cid = await self.cas.save(root_obj_bytes, codec="dag-cbor")
            self._root_cid = str(new_root_cid)
            self._dirty_root = False

        return self._root_cid  # type: ignore[return-value]

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: Optional[zarr.abc.store.ByteRequest] = None,
    ) -> Optional[zarr.core.buffer.Buffer]:
        with instrumentation.span(
            "py_hamt.sharded_store.get",
            {
                "py_hamt.zarr.key": key,
                "py_hamt.zarr.byte_range": byte_range is not None,
            },
        ):
            started_at = time.perf_counter()
            hit = False
            kind = "metadata"
            shard_idx_for_trace: int | None = None
            lookup_key = self._v2_effective_read_key(key)
            try:
                parsed_chunk = self._parse_chunk_key(lookup_key)
            except (ValueError, IndexError):
                if self._manifest_version != SHARDED_ZARR_V2:
                    raise
                return None
            try:
                if parsed_chunk is None:
                    metadata_cid_obj = self._root_obj["metadata"].get(lookup_key)
                    if metadata_cid_obj is None:
                        return None
                    data = (
                        self._metadata_read_cache.get(lookup_key)
                        if byte_range is None
                        else None
                    )
                    if data is None:
                        req_offset, req_length, req_suffix = self._map_byte_request(
                            byte_range
                        )
                        data = await self.cas.load(
                            metadata_cid_obj,
                            offset=req_offset,
                            length=req_length,
                            suffix=req_suffix,
                        )
                    if byte_range is None:
                        self._metadata_read_cache[lookup_key] = data
                    hit = True
                    return prototype.buffer.from_bytes(data)

                kind = "chunk"
                try:
                    array_index = self._array_index_for_path(parsed_chunk.array_path)
                except KeyError:
                    return await self._get_legacy_metadata_chunk(
                        lookup_key, prototype, byte_range
                    )
                linear_chunk_index = self._get_linear_chunk_index_for_index(
                    parsed_chunk.coords, array_index
                )
                shard_idx, index_in_shard = self._get_shard_info_for_index(
                    linear_chunk_index, array_index
                )
                shard_idx_for_trace = shard_idx

                cache_key = self._cache_key(array_index.array_path, shard_idx)
                shard_lock = self._shard_locks[cache_key]
                async with shard_lock:
                    cached_shard = await self._shard_data_cache.get(cache_key)
                    if (
                        self.read_only
                        and self.shard_read_mode == "sparse"
                        and cached_shard is None
                        and byte_range is None
                        and 0 <= shard_idx < array_index.num_shards
                        and array_index.shard_cids[shard_idx] is not None
                    ):
                        chunk_cid_obj = await self._load_sparse_shard_entry(
                            cache_key,
                            shard_idx,
                            cast(CID, array_index.shard_cids[shard_idx]),
                            index_in_shard,
                            array_index.chunks_per_shard,
                        )
                    else:
                        if cached_shard is None:
                            cached_shard = await self._load_or_initialize_shard_cache(
                                shard_idx, array_index.array_path
                            )
                        chunk_cid_obj = cached_shard[index_in_shard]
                if chunk_cid_obj is None:
                    legacy_buffer = await self._get_legacy_metadata_chunk(
                        lookup_key, prototype, byte_range
                    )
                    hit = legacy_buffer is not None
                    return legacy_buffer

                req_offset, req_length, req_suffix = self._map_byte_request(byte_range)
                data = await self.cas.load(
                    chunk_cid_obj,
                    offset=req_offset,
                    length=req_length,
                    suffix=req_suffix,
                )
                hit = True
                return prototype.buffer.from_bytes(data)
            finally:
                instrumentation.record_zarr_get(
                    store="sharded_store",
                    key=key,
                    kind=kind,
                    hit=hit,
                    seconds=time.perf_counter() - started_at,
                    byte_range=byte_range is not None,
                    shard_idx=shard_idx_for_trace,
                )

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        if self.read_only:
            raise PermissionError("Cannot write to a read-only store.")
        async with self._write_lock:
            await self._set_unlocked(key, value)
        return None  # type: ignore[return-value]

    async def _set_unlocked(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        await self._resize_complete.wait()

        raw_data_bytes = self._strip_v2_root_consolidated_metadata(
            key, value.to_bytes()
        )
        self._raise_if_v2_write_without_group(key, raw_data_bytes)
        await self._register_array_metadata_from_bytes(key, raw_data_bytes)
        await self._ensure_v2_parent_group_metadata(key)

        try:
            parsed_chunk = self._parse_chunk_key(key)
        except (ValueError, IndexError):
            if self._manifest_version != SHARDED_ZARR_V2:
                raise
            return None

        try:
            data_cid_obj = await self.cas.save(raw_data_bytes, codec="raw")
            await self._set_pointer_cid(
                key, cast(CID, data_cid_obj), register_metadata=False
            )
            if parsed_chunk is None:
                self._metadata_read_cache[key] = raw_data_bytes
        except Exception as e:
            raise RuntimeError(f"Failed to save data for key {key}: {e}") from e
        return None  # type: ignore[return-value]

    async def set_pointer(self, key: str, pointer: str) -> None:
        if self.read_only:
            raise PermissionError("Cannot write to a read-only store.")
        async with self._write_lock:
            await self._resize_complete.wait()
            await self._set_pointer(key, pointer, register_metadata=True)

    async def _set_pointer(
        self, key: str, pointer: str, *, register_metadata: bool
    ) -> None:
        await self._set_pointer_cid(
            key, CID.decode(pointer), register_metadata=register_metadata
        )

    async def _set_pointer_cid(
        self, key: str, pointer_cid_obj: CID, *, register_metadata: bool
    ) -> None:
        try:
            parsed_chunk = self._parse_chunk_key(key)
        except (ValueError, IndexError):
            if self._manifest_version != SHARDED_ZARR_V2:
                raise
            return None
        if parsed_chunk is None:
            if register_metadata and self._manifest_version == SHARDED_ZARR_V2:
                raw_metadata = await self.cas.load(pointer_cid_obj)
                stripped_metadata = self._strip_v2_root_consolidated_metadata(
                    key, raw_metadata
                )
                if stripped_metadata != raw_metadata:
                    stripped_pointer = await self.cas.save(
                        stripped_metadata, codec="raw"
                    )
                    if not isinstance(stripped_pointer, CID):  # pragma: no cover
                        raise TypeError(
                            "ShardedZarrStore requires CAS.save to return CIDs."
                        )
                    pointer_cid_obj = stripped_pointer
            if (
                register_metadata
                and self._array_path_from_metadata_key(key) is not None
            ):
                raw_metadata = await self.cas.load(pointer_cid_obj)
                await self._register_array_metadata_from_bytes(key, raw_metadata)
            if register_metadata:
                await self._ensure_v2_parent_group_metadata(key)
            self._root_obj["metadata"][key] = pointer_cid_obj
            self._metadata_read_cache.pop(key, None)
            self._dirty_root = True
            return None

        if self._manifest_version == SHARDED_ZARR_V1:
            chunk_info = self._root_obj["chunks"]
            if "primary_array_path" not in chunk_info:
                self._primary_array_path = parsed_chunk.array_path
                chunk_info["primary_array_path"] = parsed_chunk.array_path
                self._dirty_root = True

        array_index = self._array_index_for_path(parsed_chunk.array_path)
        linear_chunk_index = self._get_linear_chunk_index_for_index(
            parsed_chunk.coords, array_index
        )
        shard_idx, index_in_shard = self._get_shard_info_for_index(
            linear_chunk_index, array_index
        )

        cache_key = self._cache_key(array_index.array_path, shard_idx)
        async with self._use_shard(shard_idx, array_index.array_path):
            await self._shard_data_cache.update_entry(
                cache_key, index_in_shard, pointer_cid_obj
            )
        return None

    async def exists(self, key: str) -> bool:
        lookup_key = self._v2_effective_read_key(key)
        try:
            parsed_chunk = self._parse_chunk_key(lookup_key)
            if parsed_chunk is None:
                return lookup_key in self._root_obj.get("metadata", {})
            try:
                array_index = self._array_index_for_path(parsed_chunk.array_path)
            except KeyError:
                return lookup_key in self._root_obj.get("metadata", {})
            linear_chunk_index = self._get_linear_chunk_index_for_index(
                parsed_chunk.coords, array_index
            )
            shard_idx, index_in_shard = self._get_shard_info_for_index(
                linear_chunk_index, array_index
            )
            async with self._use_shard(
                shard_idx, array_index.array_path
            ) as target_shard_list:
                return target_shard_list[
                    index_in_shard
                ] is not None or lookup_key in self._root_obj.get("metadata", {})
        except (ValueError, IndexError, KeyError):
            return False

    @property
    def supports_writes(self) -> bool:
        return not self.read_only

    @property
    def supports_partial_writes(self) -> bool:
        return False

    @property
    def supports_deletes(self) -> bool:
        return not self.read_only

    async def delete(self, key: str) -> None:
        if self.read_only:
            raise PermissionError("Cannot delete from a read-only store.")
        async with self._write_lock:
            await self._delete_unlocked(key)

    async def _delete_unlocked(self, key: str) -> None:
        await self._resize_complete.wait()

        try:
            parsed_chunk = self._parse_chunk_key(key)
        except (ValueError, IndexError):
            if self._manifest_version != SHARDED_ZARR_V2:
                raise
            return None
        if parsed_chunk is None:
            if self._root_obj["metadata"].pop(key, None) is not None:
                self._metadata_read_cache.pop(key, None)
                self._dirty_root = True
            return None

        try:
            array_index = self._array_index_for_path(parsed_chunk.array_path)
        except KeyError:
            if self._root_obj["metadata"].pop(key, None) is not None:
                self._metadata_read_cache.pop(key, None)
                self._dirty_root = True
            return None
        linear_chunk_index = self._get_linear_chunk_index_for_index(
            parsed_chunk.coords, array_index
        )
        shard_idx, index_in_shard = self._get_shard_info_for_index(
            linear_chunk_index, array_index
        )

        cache_key = self._cache_key(array_index.array_path, shard_idx)
        async with self._use_shard(shard_idx, array_index.array_path):
            changed = await self._shard_data_cache.update_entry(
                cache_key, index_in_shard, None
            )
            if not changed and self._root_obj["metadata"].pop(key, None) is not None:
                self._metadata_read_cache.pop(key, None)
                self._dirty_root = True

    async def delete_dir(self, prefix: str) -> None:
        if self._manifest_version != SHARDED_ZARR_V2:
            await zarr.abc.store.Store.delete_dir(self, prefix)
            return
        if self.read_only:
            raise PermissionError("Cannot delete from a read-only store.")

        async with self._write_lock:
            await self._resize_complete.wait()
            normalized_prefix = prefix.strip("/")
            if normalized_prefix == "":
                await self._clear_v2_unlocked()
                return

            match_prefix = f"{normalized_prefix}/"
            metadata_keys_to_delete = [
                key
                for key in self._root_obj.get("metadata", {})
                if key.startswith(match_prefix)
            ]
            for key in metadata_keys_to_delete:
                await self._delete_unlocked(key)
            await self._prune_v2_array_indices_for_prefix(normalized_prefix)

    async def clear(self) -> None:
        if self._manifest_version != SHARDED_ZARR_V2:
            await zarr.abc.store.Store.clear(self)
            return
        if self.read_only:
            raise PermissionError("Cannot clear a read-only store.")

        async with self._write_lock:
            await self._resize_complete.wait()
            await self._clear_v2_unlocked()

    async def _clear_v2_unlocked(self) -> None:
        if self._manifest_version != SHARDED_ZARR_V2:
            return
        for pending_load in self._pending_shard_loads.values():
            pending_load.set()
        self._pending_shard_loads.clear()
        await self._shard_data_cache.clear()
        self._root_obj["metadata"] = {}
        self._root_obj["arrays"] = {}
        self.array_indices.clear()
        self._primary_array_path = None
        self._metadata_read_cache.clear()
        self._array_shape = ()
        self._chunk_shape = ()
        self._chunks_per_dim = ()
        self._chunks_per_shard = 0
        self._num_shards = 0
        self._total_chunks = 0
        self._dirty_root = True

    async def _prune_v2_array_indices_for_prefix(self, prefix: str) -> None:
        if self._manifest_version != SHARDED_ZARR_V2:
            return

        normalized_prefix = self._normalize_array_path(prefix)
        array_paths = [
            array_path
            for array_path in self.array_indices
            if array_path == normalized_prefix
            or array_path.startswith(f"{normalized_prefix}/")
        ]
        if not array_paths:
            return

        for array_path in array_paths:
            array_index = self.array_indices.pop(array_path)
            self._root_obj["arrays"].pop(array_path, None)
            for shard_idx in range(array_index.num_shards):
                cache_key = self._cache_key(array_path, shard_idx)
                pending_load = self._pending_shard_loads.pop(cache_key, None)
                if pending_load is not None:
                    pending_load.set()
                shard_lock = self._shard_locks[cache_key]
                async with shard_lock:
                    await self._shard_data_cache.discard(cache_key)

        if self.array_indices:
            self._primary_array_path = next(iter(self.array_indices))
            self._set_legacy_geometry_from_index(
                self.array_indices[self._primary_array_path]
            )
        else:
            self._primary_array_path = None
            self._array_shape = ()
            self._chunk_shape = ()
            self._chunks_per_dim = ()
            self._chunks_per_shard = 0
            self._num_shards = 0
            self._total_chunks = 0
        self._sync_arrays_to_root()
        self._dirty_root = True

    @property
    def supports_listing(self) -> bool:
        return True

    async def list(self) -> AsyncIterator[str]:
        yielded: set[str] = set()
        for key in list(self._root_obj.get("metadata", {})):
            yielded.add(key)
            yield key

        async for chunk_key in self._iter_chunk_keys():
            if chunk_key not in yielded:
                yield chunk_key

    async def _iter_chunk_keys(self) -> AsyncIterator[str]:
        for array_path, array_index in self.array_indices.items():
            listed_array_path = (
                self._primary_array_path
                if self._manifest_version == SHARDED_ZARR_V1
                else array_path
            )
            for shard_idx in range(array_index.num_shards):
                cache_key = self._cache_key(array_path, shard_idx)
                shard_data = await self._shard_data_cache.get(cache_key)
                if shard_data is None:
                    if array_index.shard_cids[shard_idx] is None:
                        continue
                    shard_data = await self._load_or_initialize_shard_cache(
                        shard_idx, array_path
                    )

                for index_in_shard, cid_obj in enumerate(shard_data):
                    if cid_obj is None:
                        continue
                    linear_index = (
                        shard_idx * array_index.chunks_per_shard + index_in_shard
                    )
                    if linear_index >= array_index.total_chunks:
                        continue
                    coords = self._coords_from_linear_index(
                        linear_index, array_index.chunks_per_dim
                    )
                    yield self._format_chunk_key(listed_array_path or "", coords)

    async def list_prefix(self, prefix: str) -> AsyncIterator[str]:
        async for key in self.list():
            if key.startswith(prefix):
                yield key

    def _list_dir_candidate_keys(self) -> Set[str]:
        keys = set(self._root_obj.get("metadata", {}))
        if self._manifest_version != SHARDED_ZARR_V2:
            chunk_prefix = (
                "c" if not self._primary_array_path else f"{self._primary_array_path}/c"
            )
            keys.add(chunk_prefix)
            return keys

        for array_path in self.array_indices:
            if array_path:
                parts = array_path.split("/")
                keys.update("/".join(parts[:idx]) for idx in range(1, len(parts) + 1))
            keys.add("c" if array_path == "" else f"{array_path}/c")
        return keys

    def _is_v2_chunk_listing_prefix(self, normalized_prefix: str) -> bool:
        if self._manifest_version != SHARDED_ZARR_V2:
            return False
        for array_path in self.array_indices:
            chunk_prefix = "c" if array_path == "" else f"{array_path}/c"
            if normalized_prefix == chunk_prefix or normalized_prefix.startswith(
                f"{chunk_prefix}/"
            ):
                return True
        return False

    async def graft_store(
        self,
        store_to_graft_cid: str,
        chunk_offset: Tuple[int, ...],
        *,
        source_array_path: Optional[str] = None,
        target_array_path: Optional[str] = None,
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot graft onto a read-only store.")
        async with self._write_lock:
            await self._graft_store_unlocked(
                store_to_graft_cid,
                chunk_offset,
                source_array_path=source_array_path,
                target_array_path=target_array_path,
            )

    async def _graft_store_unlocked(
        self,
        store_to_graft_cid: str,
        chunk_offset: Tuple[int, ...],
        *,
        source_array_path: Optional[str] = None,
        target_array_path: Optional[str] = None,
    ) -> None:
        await self._resize_complete.wait()

        store_to_graft = await ShardedZarrStore.open(
            cas=self.cas, read_only=True, root_cid=store_to_graft_cid
        )
        source_path = (
            source_array_path
            if source_array_path is not None
            else store_to_graft._primary_array_path
        )
        if source_path is None:
            return None

        source_index = store_to_graft._array_index_for_path(source_path)
        target_path = (
            target_array_path if target_array_path is not None else source_path
        )
        target_index = self._array_index_for_path(target_path)
        if len(chunk_offset) != len(source_index.chunks_per_dim) or len(
            chunk_offset
        ) != len(target_index.chunks_per_dim):
            raise ValueError(
                "chunk_offset must have the same number of dimensions as both source and target arrays."
            )

        for local_coords in itertools.product(*[
            range(s) for s in source_index.chunks_per_dim
        ]):
            linear_local_index = self._get_linear_chunk_index_for_index(
                tuple(local_coords), source_index
            )
            local_shard_idx, index_in_local_shard = self._get_shard_info_for_index(
                linear_local_index, source_index
            )
            source_shard_list = await store_to_graft._load_or_initialize_shard_cache(
                local_shard_idx, source_index.array_path
            )

            pointer_cid_obj = source_shard_list[index_in_local_shard]
            if pointer_cid_obj is None:
                continue

            global_coords = tuple(
                c_local + c_offset
                for c_local, c_offset in zip(local_coords, chunk_offset, strict=True)
            )
            try:
                self._validate_chunk_coords(global_coords, target_index)
            except IndexError as exc:
                raise ValueError(
                    f"Graft target chunk coordinates {global_coords} are out of bounds."
                ) from exc
            linear_global_index = self._get_linear_chunk_index_for_index(
                global_coords, target_index
            )
            global_shard_idx, index_in_global_shard = self._get_shard_info_for_index(
                linear_global_index, target_index
            )

            cache_key = self._cache_key(target_index.array_path, global_shard_idx)
            async with self._use_shard(global_shard_idx, target_index.array_path):
                await self._shard_data_cache.update_entry(
                    cache_key, index_in_global_shard, pointer_cid_obj
                )

    async def resize_store(
        self, new_shape: Tuple[int, ...], *, array_path: Optional[str] = None
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot resize a read-only store.")
        async with self._write_lock:
            await self._resize_store_unlocked(new_shape, array_path=array_path)

    async def _resize_store_unlocked(
        self, new_shape: Tuple[int, ...], *, array_path: Optional[str] = None
    ) -> None:
        """
        Resizes one shard index to accommodate a new array shape.
        """

        if self._manifest_version == SHARDED_ZARR_V2:
            target_path = (
                array_path if array_path is not None else self._primary_array_path
            )
            if target_path is None:
                raise RuntimeError("Store is not properly initialized for resizing.")
            array_index = self._array_index_for_path(target_path)
            await self._resize_array_index_guarded(array_index, tuple(new_shape))
            return None

        if (
            self._chunk_shape is None
            or self._chunks_per_shard is None
            or self._array_shape is None
        ):
            raise RuntimeError("Store is not properly initialized for resizing.")
        if len(new_shape) != len(self._array_shape):
            raise ValueError(
                "New shape must have the same number of dimensions as the old shape."
            )

        array_index = self.array_indices[""]
        await self._resize_array_index_guarded(array_index, tuple(new_shape))
        self._root_obj["chunks"]["array_shape"] = list(array_index.array_shape)
        self._root_obj["chunks"]["shard_cids"] = array_index.shard_cids
        return None

    async def resize_variable(
        self, variable_name: str, new_shape: Tuple[int, ...]
    ) -> None:
        if self.read_only:
            raise PermissionError("Cannot resize a read-only store.")
        async with self._write_lock:
            await self._resize_variable_unlocked(variable_name, new_shape)

    async def _resize_variable_unlocked(
        self, variable_name: str, new_shape: Tuple[int, ...]
    ) -> None:
        """
        Resizes the Zarr metadata and shard index for a specific variable.
        """
        await self._resize_complete.wait()

        normalized_name = self._normalize_array_path(variable_name)
        zarr_metadata_key = (
            "zarr.json" if normalized_name == "" else f"{normalized_name}/zarr.json"
        )

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
        new_zarr_metadata_cid = await self.cas.save(
            new_zarr_metadata_bytes, codec="raw"
        )

        self._root_obj["metadata"][zarr_metadata_key] = new_zarr_metadata_cid
        self._metadata_read_cache[zarr_metadata_key] = new_zarr_metadata_bytes
        if self._manifest_version == SHARDED_ZARR_V2:
            await self._register_array_metadata_from_bytes(
                zarr_metadata_key, new_zarr_metadata_bytes
            )
        self._dirty_root = True

    async def migrate_v1_to_v2(self, primary_array_path: str) -> str:
        if self.read_only:
            raise PermissionError("Cannot migrate a read-only store.")
        async with self._write_lock:
            return await self._migrate_v1_to_v2_unlocked(primary_array_path)

    async def _migrate_v1_to_v2_unlocked(self, primary_array_path: str) -> str:
        """
        Rewrite this store root as a v2 manifest, reusing the existing v1 shards
        under ``primary_array_path``.
        """
        normalized_path = self._normalize_array_path(primary_array_path)
        if not normalized_path:
            raise ValueError("primary_array_path must be a non-empty array path.")
        if self._manifest_version != SHARDED_ZARR_V1:
            raise ValueError("Only sharded_zarr_v1 stores can be migrated to v2.")

        await self._flush_unlocked()
        await self._shard_data_cache.clear()

        source_array_path = self._infer_v1_migration_source_array_path(normalized_path)
        old_metadata = dict(self._root_obj.get("metadata", {}))
        migrated_metadata = {
            self._rewrite_v1_metadata_key_for_migration(
                key, source_array_path, normalized_path
            ): cid
            for key, cid in old_metadata.items()
        }
        await self._add_missing_group_metadata(migrated_metadata, normalized_path)
        old_shard_cids = list(self._root_obj["chunks"]["shard_cids"])
        migrated_index = ArrayIndex(
            array_path=normalized_path,
            array_shape=self._array_shape,
            chunk_shape=self._chunk_shape,
            chunks_per_shard=self._chunks_per_shard,
            shard_cids=old_shard_cids,
        )

        self._manifest_version = SHARDED_ZARR_V2
        self.array_indices = {normalized_path: migrated_index}
        self._primary_array_path = normalized_path
        self._default_chunks_per_shard = migrated_index.chunks_per_shard
        self._set_legacy_geometry_from_index(migrated_index)
        self._root_obj = {
            "manifest_version": SHARDED_ZARR_V2,
            "store_type": "py_hamt.sharded_zarr",
            "zarr_format": 3,
            "sharding_config": {
                "chunks_per_shard": migrated_index.chunks_per_shard,
                "order": migrated_index.order,
            },
            "metadata": migrated_metadata,
            "arrays": {normalized_path: migrated_index.to_manifest()},
        }
        self._metadata_read_cache.clear()
        self._dirty_root = True
        return await self._flush_unlocked()

    async def list_dir(self, prefix: str) -> AsyncIterator[str]:
        seen: Set[str] = set()
        normalized_prefix = prefix.strip("/")
        if (
            self.read_only
            and normalized_prefix == ""
            and self._v2_requires_explicit_group_for_root_read()
        ):
            raise ValueError(self._V2_MULTI_GROUP_READ_MESSAGE)
        effective_prefix = self._v2_effective_list_dir_prefix(normalized_prefix)
        match_prefix = f"{effective_prefix}/" if effective_prefix else ""

        if self._is_v2_chunk_listing_prefix(effective_prefix):
            async for key in self._iter_chunk_keys():
                if not key.startswith(match_prefix):
                    continue
                suffix = key[len(match_prefix) :]
                first_component = suffix.split("/", 1)[0]
                if first_component not in seen:
                    seen.add(first_component)
                    yield first_component
            return

        for key in self._list_dir_candidate_keys():
            if not key.startswith(match_prefix):
                continue
            suffix = key[len(match_prefix) :]
            if suffix == "":
                continue
            first_component = suffix.split("/", 1)[0]
            if first_component not in seen:
                seen.add(first_component)
                yield first_component
