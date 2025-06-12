#!/usr/bin/env python3

import asyncio

import xarray as xr
from xarray import Dataset

from py_hamt import HAMT, KuboCAS, ZarrHAMTStore


async def fetch_zarr_from_gateway(cid: str, gateway: str = "https://ipfs.io"):
    """
    Fetch and open a Zarr dataset from a public gateway using py-hamt.

    Parameters:
    -----------
    cid : str
        The IPFS CID of the Zarr dataset root
    gateway : str
        The IPFS gateway URL to use (default: ipfs.io)

    Returns:
    --------
    Dataset
        The xarray Dataset from the Zarr store
    """
    print(f"Fetching CID {cid} from gateway {gateway}")

    # Use KuboCAS with the public gateway
    # Setting RPC to None since we're only reading, not writing
    async with KuboCAS(rpc_base_url=None, gateway_base_url=gateway) as kubo_cas:
        # Initialize the HAMT with the root CID
        hamt = await HAMT.build(
            cas=kubo_cas, root_node_id=cid, values_are_bytes=True, read_only=True
        )

        # Initialize the Zarr store
        zhs = ZarrHAMTStore(hamt, read_only=True)

        # Open the dataset with xarray
        print("Opening Zarr dataset...")
        zarr_ds: Dataset = xr.open_zarr(store=zhs)

        # Print info about the dataset
        print("\nDataset summary:")
        print(f"Dimensions: {dict(zarr_ds.dims)}")
        print(f"Data variables: {list(zarr_ds.data_vars)}")
        print(f"Coordinates: {list(zarr_ds.coords)}")

        # Return the dataset
        return zarr_ds


async def main():
    # Example CID - this points to a weather dataset stored on IPFS
    cid = "bafyr4iecw3faqyvj75psutabk2jxpddpjdokdy5b26jdnjjzpkzbgb5xoq"

    # Try different public gateways
    gateways = [
        "https://ipfs.io",  # IPFS.io gateway
        "https://dweb.link",  # Protocol Labs gateway
        "https://cloudflare-ipfs.com",  # Cloudflare gateway
    ]

    # Try each gateway
    for gateway in gateways:
        print(f"\n===== Testing gateway: {gateway} =====")
        try:
            ds = await fetch_zarr_from_gateway(cid, gateway)
            print("Success! Dataset loaded correctly.")

            # Demonstrate accessing data
            if "precip" in ds and "time" in ds.coords:
                first_timestamp = ds["time"].values[0]
                print(f"First timestamp: {first_timestamp}")

                # Get a slice of the data
                first_slice = ds["precip"].isel(time=0)
                print(f"First precipitation slice shape: {first_slice.shape}")

            break  # Stop after first successful gateway
        except Exception as e:
            print(f"Error with gateway {gateway}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
