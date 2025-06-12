import asyncio
import os
import sys

import httpx
import xarray as xr
from xarray import Dataset

from py_hamt import HAMT, KuboCAS, ZarrHAMTStore

print(f"Python version: {sys.version}")
print(f"Python path: {os.path.dirname(sys.executable)}")


print(f"httpx version: {httpx.__version__}")


async def main():
    cid = "bafyr4iecw3faqyvj75psutabk2jxpddpjdokdy5b26jdnjjzpkzbgb5xoq"

    # Use KuboCAS as an async context manager
    async with KuboCAS() as kubo_cas:  # connects to a local kubo node
        hamt = await HAMT.build(
            cas=kubo_cas, root_node_id=cid, values_are_bytes=True, read_only=True
        )
        # Initialize the store
        zhs = ZarrHAMTStore(hamt, read_only=True)

        # Open the dataset with xarray
        zarr_ds: Dataset = xr.open_zarr(store=zhs)
        # List all variables
        print("All variables:", list(zarr_ds.variables))

        # Alternatively, just the data variables (not coordinates or attributes)
        print("Data variables:", list(zarr_ds.data_vars))

        # Or, if you want coordinate variables too
        print("Coordinates:", list(zarr_ds.coords))

        first_timestamp = zarr_ds["time"].values[0]

        first_year = str(first_timestamp.astype("datetime64[Y]"))

        first_year_precip = zarr_ds["precip"].sel(
            time=slice(f"{first_year}-01-01", f"{first_year}-12-31")
        )

        print(f"Precip data for {first_year}:")
        print(first_year_precip)


if __name__ == "__main__":
    asyncio.run(main())
