import argparse
import asyncio
import time

from multiformats import CID

from .hamt import HAMT
from .sharded_zarr_store import SHARDED_ZARR_V2, ShardedZarrStore
from .store_httpx import KuboCAS


def _is_zarr_chunk_key(key: str) -> bool:
    if key.endswith(("zarr.json", ".zarray", ".zattrs", ".zgroup")):
        return False
    return key.startswith("c/") or "/c/" in key


async def convert_hamt_to_sharded(
    cas: KuboCAS, hamt_root_cid: str, chunks_per_shard: int
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
        if _is_zarr_chunk_key(key):
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
        if not _is_zarr_chunk_key(key):
            continue
        count += 1
        cid = await hamt_ro.get_pointer(key)
        if not isinstance(cid, CID):  # pragma: no cover
            raise TypeError(f"Expected CID pointer for key {key!r}.")
        cid_base32_str = str(cid.encode("base32"))
        await dest_store.set_pointer(key, cid_base32_str)
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
