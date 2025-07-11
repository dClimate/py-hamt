# import time

# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr
# from dag_cbor.ipld import IPLDKind
# from multiformats import CID

# # Import both store implementations
# from py_hamt import HAMT, KuboCAS, ShardedZarrStore
# from py_hamt.zarr_hamt_store import ZarrHAMTStore


# @pytest.mark.asyncio(loop_scope="session")
# async def test_benchmark_sharded_store():
#     """Benchmarks write and read performance for the new ShardedZarrStore.""" # Updated docstring
#     print("\n\n" + "=" * 80)
#     print("🚀 STARTING BENCHMARK for ShardedZarrStore") # Updated print
#     print("=" * 80)


#     rpc_base_url = f"https://ipfs-gateway.dclimate.net"
#     gateway_base_url = f"https://ipfs-gateway.dclimate.net"
#     headers = {
#         "X-API-Key": "",
#     }

#     async with KuboCAS(
#         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url, headers=headers
#     ) as kubo_cas:
#         # --- Write ---
#         # root_cid = "bafyr4ifayhevbtfg2qzffuicic3rwli4fhnnkhrfduuxkwvetppfk4ogbe"
#         root_cid = "bafyr4ifs4oejlvtwvb57udbbhba5yllss4cixjkxrevq54g3mo5kwknpwy"
#         print(f"\n--- [ShardedZarr] STARTING READ ---") # Updated print
#         # --- Read ---
#         start = time.perf_counter()
#         # When opening for read, chunks_per_shard is read from the store's metadata
#         store_read = await ShardedZarrStore.open( # Use ShardedZarrStore
#             cas=kubo_cas, read_only=True, root_cid=root_cid
#         )
#         stop = time.perf_counter()
#         print(f"Total time to open ShardedZarrStore: {stop - start:.2f} seconds")
#         print(f"Opened ShardedZarrStore for reading with root CID: {root_cid}")

#         start_read = time.perf_counter()
#         ipfs_ds = xr.open_zarr(store=store_read)
#         start_read = time.perf_counter()
#         print(ipfs_ds)
#         stop_read = time.perf_counter()
#         print(f"Total time to read dataset: {stop_read - start_read:.2f} seconds")
#         # start_read = time.perf_counter()
#         # print(ipfs_ds.variables, ipfs_ds.coords)  # Print available variables and coordinates for debugging
#         # stop_read = time.perf_counter()
#         # print(f"Total time to read dataset variables and coordinates: {stop_read - start_read:.2f} seconds")
#         start_read = time.perf_counter()
#         # Force a read of some data to ensure it's loaded (e.g., first time slice of 'temp' variable)
#         if "FPAR" in ipfs_ds.variables and "time" in ipfs_ds.coords:
#             print("Fetching 'FPAR' data...")

#             # Define date range
#             date_from = "2000-05-15"
#             date_to = "2004-05-30"

#             # Define bounding box from polygon coordinates
#             min_lon, max_lon = 4.916695, 5.258908
#             min_lat, max_lat = 51.921763, 52.160344

#             print(ipfs_ds["FPAR"].sel(
#                 time=slice(date_from, date_to),
#                 latitude=slice(min_lat, max_lat),
#                 longitude=slice(min_lon, max_lon)
#             ))

#             # Fetch data for the specified time and region
#             data_fetched = ipfs_ds["FPAR"].sel(
#                 time=slice(date_from, date_to),
#                 latitude=slice(min_lat, max_lat),
#                 longitude=slice(min_lon, max_lon)
#             ).values

#             # Calculate the size of the fetched data
#             data_size = data_fetched.nbytes if data_fetched is not None else 0
#             print(f"Fetched data size: {data_size / (1024 * 1024):.4f} MB")
#         elif len(ipfs_ds.data_vars) > 0 : # Fallback: try to read from the first data variable
#             first_var_name = list(ipfs_ds.data_vars.keys())[0]
#             # Construct a minimal selection based on available dimensions
#             selection = {dim: 0 for dim in ipfs_ds[first_var_name].dims}
#             if selection:
#                  _ = ipfs_ds[first_var_name].isel(**selection).values
#             else: # If no dimensions, try loading the whole variable (e.g. scalar)
#                  _ = ipfs_ds[first_var_name].values
#         end_read = time.perf_counter()

#         print(f"\n--- [ShardedZarr] Read Stats ---") # Updated print
#         print(f"Total time to open and read some data: {end_read - start_read:.2f} seconds")
#         print("=" * 80)
#         # Speed in MB/s
#         if data_size > 0:
#             speed = data_size / (end_read - start_read) / (1024 * 1024)
#             print(f"Read speed: {speed:.2f} MB/s")
#         else:
#             print("No data fetched, cannot calculate speed.")

# # ###
# # BENCHMARK FOR THE ORIGINAL ZarrHAMTStore
# # ###
# @pytest.mark.asyncio(loop_scope="session")
# async def test_benchmark_hamt_store():
#     """Benchmarks write and read performance for the ZarrHAMTStore."""
#     print("\n\n" + "=" * 80)
#     print("🚀 STARTING BENCHMARK for ZarrHAMTStore")
#     print("=" * 80)

#     rpc_base_url = f"https://ipfs-gateway.dclimate.net/"
#     gateway_base_url = f"https://ipfs-gateway.dclimate.net/"
#     # headers = {
#     #     "X-API-Key": "",
#     # }
#     headers = {}

#     async with KuboCAS(
#         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url, headers=headers
#     ) as kubo_cas:

#         root_cid = "bafyr4igl3pmswu5pfzb6dcgcxj3ipxlpxxxad7j7tf45obxe5pkp4xgpwe"
#         # root_node_id = CID.decode(root_cid)

#         hamt = await HAMT.build(
#             cas=kubo_cas, root_node_id=root_cid, values_are_bytes=True, read_only=True
#         )
#         start = time.perf_counter()
#         ipfs_ds: xr.Dataset
#         zhs = ZarrHAMTStore(hamt, read_only=True)
#         ipfs_ds = xr.open_zarr(store=zhs)
#         print(ipfs_ds)

#         # --- Read ---
#         hamt = HAMT(cas=kubo_cas, values_are_bytes=True, root_node_id=root_cid, read_only=True)

#         # Initialize the store
#         zhs = ZarrHAMTStore(hamt, read_only=True)

#         start_read = time.perf_counter()
#         ipfs_ds = xr.open_zarr(store=zhs)
#         # Force a read of some data to ensure it's loaded
#         data_fetched = ipfs_ds.precip.values

#         # Calculate the size of the fetched data
#         data_size = data_fetched.nbytes if data_fetched is not None else 0
#         print(f"Fetched data size: {data_size / (1024 * 1024):.4f} MB")
#         end_read = time.perf_counter()

#         print(f"\n--- [HAMT] Read Stats ---")
#         print(f"Total time to open and read: {end_read - start_read:.2f} seconds")

#         if data_size > 0:
#             speed = data_size / (end_read - start_read) / (1024 * 1024)
#             print(f"Read speed: {speed:.2f} MB/s")
#         else:
#             print("No data fetched, cannot calculate speed.")
