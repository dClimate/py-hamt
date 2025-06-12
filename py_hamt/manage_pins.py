"""
A command-line tool to recursively pin or unpin all CIDs associated with a
sharded Zarr dataset on IPFS using its root CID.
"""

import asyncio
import argparse
import sys
from py_hamt import KuboCAS, ShardedZarrStore

# --- CLI Logic Functions ---


async def handle_pin(args):
    """
    Connects to IPFS, loads the dataset from the root CID, and pins all
    associated CIDs (root, metadata, shards, and data chunks).
    """
    async with KuboCAS(
        rpc_base_url=args.rpc_url, gateway_base_url=args.gateway_url
    ) as kubo_cas:
        try:
            print(f"-> Opening store with root CID: {args.root_cid}")
            store = await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=args.root_cid
            )
        except Exception as e:
            print(
                f"Error: Failed to open Zarr store for CID {args.root_cid}. Ensure the CID is correct and the daemon is running.",
                file=sys.stderr,
            )
            print(f"Details: {e}", file=sys.stderr)
            return

        print(f"-> Sending commands to pin the entire dataset to {args.rpc_url}...")
        await store.pin_entire_dataset()
        print("\n--- Pinning Commands Sent Successfully ---")
        print("The IPFS node will now pin all objects in the background.")


async def handle_unpin(args):
    """
    Connects to IPFS, loads the dataset from the root CID, and unpins all
    associated CIDs.
    """
    async with KuboCAS(
        rpc_base_url=args.rpc_url, gateway_base_url=args.gateway_url
    ) as kubo_cas:
        try:
            print(f"-> Opening store with root CID: {args.root_cid}")
            store = await ShardedZarrStore.open(
                cas=kubo_cas, read_only=True, root_cid=args.root_cid
            )
        except Exception as e:
            print(
                f"Error: Failed to open Zarr store for CID {args.root_cid}. Ensure the CID is correct and the daemon is running.",
                file=sys.stderr,
            )
            print(f"Details: {e}", file=sys.stderr)
            return

        print(f"-> Sending commands to unpin the entire dataset from {args.rpc_url}...")
        await store.unpin_entire_dataset()
        print("\n--- Unpinning Commands Sent Successfully ---")
        print("The IPFS node will now unpin all objects in the background.")


def main():
    """Sets up the argument parser and runs the selected command."""
    parser = argparse.ArgumentParser(
        description="A CLI tool to pin or unpin sharded Zarr datasets on IPFS.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--rpc-url",
        default="http://127.0.0.1:5001",
        help="IPFS Kubo RPC API endpoint URL.",
    )
    parser.add_argument(
        "--gateway-url",
        default="http://127.0.0.1:8080",
        help="IPFS Gateway URL (needed for loading shards).",
    )

    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Pin Command ---
    parser_pin = subparsers.add_parser(
        "pin", help="Recursively pin a dataset using its root CID."
    )
    parser_pin.add_argument("root_cid", help="The root CID of the dataset to pin.")
    parser_pin.set_defaults(func=handle_pin)

    # --- Unpin Command ---
    parser_unpin = subparsers.add_parser(
        "unpin", help="Recursively unpin a dataset using its root CID."
    )
    parser_unpin.add_argument("root_cid", help="The root CID of the dataset to unpin.")
    parser_unpin.set_defaults(func=handle_unpin)

    args = parser.parse_args()

    try:
        asyncio.run(args.func(args))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
