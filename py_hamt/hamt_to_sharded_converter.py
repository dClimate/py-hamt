import argparse
import asyncio
import time
from collections.abc import Mapping

from multiformats import CID

from .hamt import HAMT
from .sharded_zarr_store import SHARDED_ZARR_V2, ArrayIndex, ShardedZarrStore
from .store_httpx import ContentAddressedStore, KuboCAS

ZARR_METADATA_SUFFIXES = ("zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata")


def _is_zarr_metadata_key(key: str) -> bool:
    return key.endswith(ZARR_METADATA_SUFFIXES)


def _classic_dotted_chunk_key_to_v3(key: str) -> str | None:
    array_path, _, coord_part = key.rpartition("/")
    if "." not in coord_part:
        return None

    parts = coord_part.split(".")
    if not parts or not all(part.isdecimal() for part in parts):
        return None
    return ShardedZarrStore._format_chunk_key(
        array_path, tuple(int(part) for part in parts)
    )


def _classic_slash_chunk_key_to_v3(
    key: str, array_indices: Mapping[str, ArrayIndex]
) -> str | None:
    for array_path, array_index in sorted(
        array_indices.items(), key=lambda item: len(item[0]), reverse=True
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
        if all(part.isdecimal() for part in parts):
            return ShardedZarrStore._format_chunk_key(
                array_path, tuple(int(part) for part in parts)
            )
    return None


def _normalize_zarr_chunk_key(
    key: str, array_indices: Mapping[str, ArrayIndex] | None = None
) -> str | None:
    if _is_zarr_metadata_key(key):
        return None

    classic_key = _classic_dotted_chunk_key_to_v3(key)
    if classic_key is not None:
        return classic_key

    if array_indices is not None:
        classic_key = _classic_slash_chunk_key_to_v3(key, array_indices)
        if classic_key is not None:
            return classic_key

    if key.startswith("c/") or "/c/" in key:
        return key
    return None


def _is_zarr_chunk_key(key: str) -> bool:
    return _normalize_zarr_chunk_key(key) is not None


async def convert_hamt_to_sharded(
    cas: ContentAddressedStore, hamt_root_cid: str, chunks_per_shard: int
) -> str:
    """
    Converts a Zarr dataset from a HAMT-based store to a ShardedZarrStore.

    Args:
        cas: An initialized ContentAddressedStore instance (KuboCAS).
        hamt_root_cid: The root CID of the source ZarrHAMTStore.
        chunks_per_shard: The number of chunks to group into a single shard in the new store.

    Returns:
        The root CID of the newly created ShardedZarrStore.
    """
    print(f"--- Starting Conversion from HAMT Root {hamt_root_cid} ---")
    start_time = time.perf_counter()
    # 1. Open the source HAMT store for reading
    print("Opening source HAMT store...")
    hamt_ro = await HAMT.build(
        cas=cas, root_node_id=hamt_root_cid, values_are_bytes=True, read_only=True
    )

    # 2. Create the destination ShardedZarrStore for writing.
    print(
        f"Initializing new ShardedZarrStore v2 with {chunks_per_shard} chunks per shard..."
    )
    dest_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=chunks_per_shard,
        manifest_version=SHARDED_ZARR_V2,
    )

    print("Destination store initialized.")

    # 3. Copy metadata first so each chunked array path registers its own shard
    # index before chunk pointers are inserted.
    print("Starting data migration...")
    count = 0
    async for key in hamt_ro.keys():
        if not _is_zarr_metadata_key(key):
            continue
        count += 1
        cid = await hamt_ro.get_pointer(key)
        if not isinstance(cid, CID):  # pragma: no cover
            raise TypeError(f"Expected CID pointer for key {key!r}.")
        cid_base32_str = str(cid.encode("base32"))
        await dest_store.set_pointer(key, cid_base32_str)
        if count % 200 == 0:  # pragma: no cover
            print(f"Migrated {count} keys...")  # pragma: no cover

    async for key in hamt_ro.keys():
        chunk_key = _normalize_zarr_chunk_key(key, dest_store.array_indices)
        if chunk_key is None:
            if _is_zarr_metadata_key(key):
                continue
            raise ValueError(
                f"Cannot classify Zarr key {key!r} as metadata or chunk during conversion."
            )
        count += 1
        cid = await hamt_ro.get_pointer(key)
        if not isinstance(cid, CID):  # pragma: no cover
            raise TypeError(f"Expected CID pointer for key {key!r}.")
        cid_base32_str = str(cid.encode("base32"))
        await dest_store.set_pointer(chunk_key, cid_base32_str)
        if count % 200 == 0:  # pragma: no cover
            print(f"Migrated {count} keys...")  # pragma: no cover

    print(f"Migration of {count} total keys complete.")

    # 5. Finalize the new store by flushing it to the CAS
    print("Flushing new store to get final root CID...")
    new_root_cid = await dest_store.flush()
    end_time = time.perf_counter()

    print("\n--- Conversion Complete! ---")
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"New ShardedZarrStore Root CID: {new_root_cid}")
    return new_root_cid


async def sharded_converter_cli():
    parser = argparse.ArgumentParser(
        description="Convert a Zarr HAMT store to a Sharded Zarr store."
    )
    parser.add_argument(
        "hamt_cid", type=str, help="The root CID of the source Zarr HAMT store."
    )
    parser.add_argument(
        "--chunks-per-shard",
        type=int,
        default=6250,
        help="Number of chunk CIDs to store per shard in the new store.",
    )
    parser.add_argument(
        "--rpc-url",
        type=str,
        default="http://127.0.0.1:5001",
        help="The URL of the IPFS Kubo RPC API.",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default="http://127.0.0.1:8080",
        help="The URL of the IPFS Gateway.",
    )
    args = parser.parse_args()
    # Initialize the KuboCAS client with the provided RPC and Gateway URLs
    async with KuboCAS(
        rpc_base_url=args.rpc_url, gateway_base_url=args.gateway_url
    ) as cas_client:
        try:
            await convert_hamt_to_sharded(
                cas=cas_client,
                hamt_root_cid=args.hamt_cid,
                chunks_per_shard=args.chunks_per_shard,
            )
        except Exception as e:
            print(f"\nAn error occurred: {e}")


if __name__ == "__main__":
    asyncio.run(sharded_converter_cli())  # pragma: no cover
