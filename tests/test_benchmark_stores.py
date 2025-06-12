# import time

# import numpy as np
# import pandas as pd
# import pytest
# import xarray as xr
# from dag_cbor.ipld import IPLDKind

# # Import both store implementations
# from py_hamt import HAMT, KuboCAS, FlatZarrStore, ShardedZarrStore
# from py_hamt.zarr_hamt_store import ZarrHAMTStore


# @pytest.fixture(scope="module")
# def random_zarr_dataset():
#     """Creates a random xarray Dataset for benchmarking."""
#     # Using a slightly larger dataset for a more meaningful benchmark
#     times = pd.date_range("2024-01-01", periods=100)
#     lats = np.linspace(-90, 90, 18)
#     lons = np.linspace(-180, 180, 36)

#     temp = np.random.randn(len(times), len(lats), len(lons))
#     precip = np.random.gamma(2, 0.5, size=(len(times), len(lats), len(lons)))

#     ds = xr.Dataset(
#         {
#             "temp": (["time", "lat", "lon"], temp),
#         },
#         coords={"time": times, "lat": lats, "lon": lons},
#     )

#     # Define chunking for the store
#     ds = ds.chunk({"time": 20, "lat": 18, "lon": 36})
#     yield ds

# @pytest.fixture(scope="module")
# def random_shard_dataset():
#     """Creates a random xarray Dataset for benchmarking."""
#     # Using a slightly larger dataset for a more meaningful benchmark
#     times = pd.date_range("2024-01-01", periods=100)
#     lats = np.linspace(-90, 90, 18)
#     lons = np.linspace(-180, 180, 36)

#     temp = np.random.randn(len(times), len(lats), len(lons))
#     precip = np.random.gamma(4, 2.5, size=(len(times), len(lats), len(lons)))

#     ds = xr.Dataset(
#         {
#             "precip": (["time", "lat", "lon"], precip),
#         },
#         coords={"time": times, "lat": lats, "lon": lons},
#     )

#     # Define chunking for the store
#     ds = ds.chunk({"time": 20, "lat": 18, "lon": 36})
#     yield ds


# # # ###
# # # BENCHMARK FOR THE NEW FlatZarrStore
# # # ###
# # @pytest.mark.asyncio(loop_scope="session")
# # async def test_benchmark_flat_store(
# #     create_ipfs: tuple[str, str],
# #     random_zarr_dataset: xr.Dataset,
# # ):
# #     """Benchmarks write and read performance for the new FlatZarrStore."""
# #     print("\n\n" + "=" * 80)
# #     print("🚀 STARTING BENCHMARK for FlatZarrStore")
# #     print("=" * 80)

# #     rpc_base_url, gateway_base_url = create_ipfs
# #     # rpc_base_url = f"https://ipfs-gateway.dclimate.net"
# #     # gateway_base_url = f"https://ipfs-gateway.dclimate.net"
# #     # headers = {
# #     #     "X-API-Key": "",
# #     # }
# #     headers = {}
# #     test_ds = random_zarr_dataset

# #     async with KuboCAS(
# #         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url, headers=headers
# #     ) as kubo_cas:
# #         # --- Write ---
# #         # The full shape after appending
# #         appended_shape = list(test_ds.dims.values())
# #         time_axis_index = list(test_ds.dims).index("time")
# #         appended_shape[time_axis_index] *= 2
# #         final_array_shape = tuple(appended_shape)

# #         final_chunk_shape = []
# #         for dim_name in test_ds.dims: # Preserves dimension order
# #             if dim_name in test_ds.chunks:
# #                 # test_ds.chunks[dim_name] is a tuple e.g. (20,)
# #                 final_chunk_shape.append(test_ds.chunks[dim_name][0])
# #             else:
# #                 # Fallback if a dimension isn't explicitly chunked (should use its full size)
# #                 final_chunk_shape.append(test_ds.dims[dim_name])
# #         final_chunk_shape = tuple(final_chunk_shape)

# #         store_write = await FlatZarrStore.open(
# #             cas=kubo_cas,
# #             read_only=False,
# #             array_shape=final_array_shape,
# #             chunk_shape=final_chunk_shape,
# #         )

# #         start_write = time.perf_counter()
# #         # Perform an initial write and an append
# #         test_ds.to_zarr(store=store_write, mode="w")
# #         test_ds.to_zarr(store=store_write, mode="a", append_dim="time")
# #         root_cid = await store_write.flush()  # Flush to get the final CID
# #         end_write = time.perf_counter()

# #         print(f"\n--- [FlatZarr] Write Stats ---")
# #         print(f"Total time to write and append: {end_write - start_write:.2f} seconds")
# #         print(f"Final Root CID: {root_cid}")

# #         # --- Read ---
# #         store_read = await FlatZarrStore.open(
# #             cas=kubo_cas, read_only=True, root_cid=root_cid
# #         )

# #         start_read = time.perf_counter()
# #         ipfs_ds = xr.open_zarr(store=store_read)
# #         # Force a read of some data to ensure it's loaded
# #         _ = ipfs_ds.temp.isel(time=0).values
# #         end_read = time.perf_counter()

# #         print(f"\n--- [FlatZarr] Read Stats ---")
# #         print(f"Total time to open and read: {end_read - start_read:.2f} seconds")

# #         # --- Verification ---
# #         full_test_ds = xr.concat([test_ds, test_ds], dim="time")
# #         xr.testing.assert_identical(full_test_ds, ipfs_ds)
# #         print("\n✅ [FlatZarr] Data verification successful.")
# #         print("=" * 80)

# @pytest.mark.asyncio(loop_scope="session")
# async def test_benchmark_sharded_store( # Renamed function
#     create_ipfs: tuple[str, str],
#     random_shard_dataset: xr.Dataset,
# ):
#     """Benchmarks write and read performance for the new ShardedZarrStore.""" # Updated docstring
#     print("\n\n" + "=" * 80)
#     print("🚀 STARTING BENCHMARK for ShardedZarrStore") # Updated print
#     print("=" * 80)

#     rpc_base_url, gateway_base_url = create_ipfs

#     rpc_base_url = f"https://ipfs-gateway.dclimate.net"
#     gateway_base_url = f"https://ipfs-gateway.dclimate.net"
#     headers = {
#         "X-API-Key": "",
#     }
#     # headers = {}
#     test_ds = random_shard_dataset

#     # Define chunks_per_shard for the ShardedZarrStore
#     chunks_per_shard_config = 50 # Configuration for sharding

#     async with KuboCAS(
#         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url, headers=headers
#     ) as kubo_cas:
#         # --- Write ---
#         # The full shape after appending
#         appended_shape = list(test_ds.dims.values())
#         time_axis_index = list(test_ds.dims).index("time")
#         appended_shape[time_axis_index] *= 2 # Simulating appending along time dimension
#         final_array_shape = tuple(appended_shape)

#         # Determine chunk shape from the dataset's encoding or dimensions
#         final_chunk_shape_list = []
#         for dim_name in test_ds.dims: # Preserves dimension order from the dataset
#             if dim_name in test_ds.chunks:
#                 # test_ds.chunks is a dict like {'time': (20,), 'y': (20,), 'x': (20,)}
#                 final_chunk_shape_list.append(test_ds.chunks[dim_name][0])
#             else:
#                 # Fallback if a dimension isn't explicitly chunked (should use its full size)
#                 final_chunk_shape_list.append(test_ds.dims[dim_name])
#         final_chunk_shape = tuple(final_chunk_shape_list)

#         # Use ShardedZarrStore and provide chunks_per_shard
#         store_write = await ShardedZarrStore.open(
#             cas=kubo_cas,
#             read_only=False,
#             array_shape=final_array_shape,
#             chunk_shape=final_chunk_shape,
#             chunks_per_shard=chunks_per_shard_config # Added new parameter
#         )

#         start_write = time.perf_counter()
#         # Perform an initial write and an append
#         test_ds.to_zarr(store=store_write, mode="w")
#         test_ds.to_zarr(store=store_write, mode="a", append_dim="time")
#         root_cid = await store_write.flush()  # Flush to get the final CID
#         end_write = time.perf_counter()

#         print(f"\n--- [ShardedZarr] Write Stats (chunks_per_shard={chunks_per_shard_config}) ---") # Updated print
#         print(f"Total time to write and append: {end_write - start_write:.2f} seconds")
#         print(f"Final Root CID: {root_cid}")

#         print(f"\n--- [ShardedZarr] STARTING READ ---") # Updated print
#         # --- Read ---
#         # When opening for read, chunks_per_shard is read from the store's metadata
#         store_read = await ShardedZarrStore.open( # Use ShardedZarrStore
#             cas=kubo_cas, read_only=True, root_cid=root_cid
#         )

#         start_read = time.perf_counter()
#         ipfs_ds = xr.open_zarr(store=store_read)
#         # Force a read of some data to ensure it's loaded (e.g., first time slice of 'temp' variable)
#         if "precip" in ipfs_ds.variables and "time" in ipfs_ds.coords:
#             # _ = ipfs_ds.temp.isel(time=0).values
#             data_fetched = ipfs_ds.precip.isel(time=slice(0, 1)).values

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

#         # --- Verification ---
#         # Create the expected full dataset after append operation
#         full_test_ds = xr.concat([test_ds, test_ds], dim="time")
#         xr.testing.assert_identical(full_test_ds, ipfs_ds)
#         print("\n✅ [ShardedZarr] Data verification successful.") # Updated print
#         print("=" * 80)

# # ###
# # BENCHMARK FOR THE ORIGINAL ZarrHAMTStore
# # ###
# @pytest.mark.asyncio(loop_scope="session")
# async def test_benchmark_hamt_store(
#     create_ipfs: tuple[str, str],
#     random_zarr_dataset: xr.Dataset,
# ):
#     """Benchmarks write and read performance for the ZarrHAMTStore."""
#     print("\n\n" + "=" * 80)
#     print("🚀 STARTING BENCHMARK for ZarrHAMTStore")
#     print("=" * 80)

#     rpc_base_url, gateway_base_url = create_ipfs

#     # rpc_base_url = f"https://ipfs-gateway.dclimate.net"
#     # gateway_base_url = f"https://ipfs-gateway.dclimate.net"
#     # headers = {
#     #     "X-API-Key": "",
#     # }
#     headers = {}
#     test_ds = random_zarr_dataset

#     async with KuboCAS(
#         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url, headers=headers
#     ) as kubo_cas:
#         # --- Write ---
#         print("Building HAMT store...")
#         hamt = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
#         print("HAMT store built successfully.")
#         zhs = ZarrHAMTStore(hamt)
#         print("ZarrHAMTStore created successfully.")

#         start_write = time.perf_counter()
#         # Perform an initial write and an append to simulate a common workflow
#         test_ds.to_zarr(store=zhs, mode="w")
#         print("Initial write completed, now appending...")
#         test_ds.to_zarr(store=zhs, mode="a", append_dim="time")
#         await hamt.make_read_only()  # Flush and freeze to get the final CID
#         end_write = time.perf_counter()

#         cid: IPLDKind = hamt.root_node_id
#         print(f"\n--- [HAMT] Write Stats ---")
#         print(f"Total time to write and append: {end_write - start_write:.2f} seconds")
#         print(f"Final Root CID: {cid}")

#         # --- Read ---
#         hamt_ro = await HAMT.build(
#             cas=kubo_cas, root_node_id=cid, values_are_bytes=True, read_only=True
#         )
#         zhs_ro = ZarrHAMTStore(hamt_ro, read_only=True)

#         start_read = time.perf_counter()
#         ipfs_ds = xr.open_zarr(store=zhs_ro)
#         # Force a read of some data to ensure it's loaded
#         data_fetched = ipfs_ds.temp.isel(time=slice(0, 1)).values

#         # Calculate the size of the fetched data
#         data_size = data_fetched.nbytes if data_fetched is not None else 0
#         print(f"Fetched data size: {data_size / (1024 * 1024):.4f} MB")
#         end_read = time.perf_counter()

#         print(f"\n--- [HAMT] Read Stats ---")
#         print(f"Total time to open and read: {end_read - start_read:.2f} seconds")


#         # --- Verification ---
#         full_test_ds = xr.concat([test_ds, test_ds], dim="time")
#         xr.testing.assert_identical(full_test_ds, ipfs_ds)
#         print("\n✅ [HAMT] Data verification successful.")
#         print("=" * 80)
