"""
This file can be run by pytest, but is not automatically included since it includes some tests that may run for a long time, and are not useful for verifying the HAMT's correctness properties.

This test suite contains various performance tests, which are meant to be run individually.
"""

# import time

# import pytest
# import xarray as xr

# from py_hamt import HAMT, KuboCAS
# from py_hamt.zarr_hamt_store import ZarrHAMTStore

# @pytest.mark.asyncio
# async def test_large_kv_set() -> None:
#     """This test is meant for finding whether the HAMT performance scales linearly with increasing set size, an issue with HAMT v2.
#     Feel free to tune and run the LARGE_KV_SET_SIZE variable as needed for gathering the different timepoints.
#     """
#     LARGE_KV_SET_SIZE: int = 1_000_000

#     cas = InMemoryCAS()
#     hamt = await HAMT.build(cas=cas)
#     start: float = time.perf_counter()
#     await asyncio.gather(
#         *[hamt.set(str(k_int), k_int) for k_int in range(LARGE_KV_SET_SIZE)]
#     )
#     await hamt.make_read_only()
#     end: float = time.perf_counter()
#     elapsed: float = end - start
#     print(f"Took {elapsed:.2f} seconds")
#     assert (
#         len([key async for key in hamt.keys()])
#         == (await hamt.len())
#         == LARGE_KV_SET_SIZE
#     )
#     for k_int in range(LARGE_KV_SET_SIZE):
#         assert (await hamt.get(str(k_int))) == k_int


# # ###
# # BENCHMARK FOR THE ORIGINAL ZarrHAMTStore
# # ###
# @pytest.mark.asyncio(loop_scope="session")
# async def test_benchmark_hamt_store():
#     """Benchmarks write and read performance for the ZarrHAMTStore."""
#     print("\n\n" + "=" * 80)
#     print("🚀 STARTING BENCHMARK for ZarrHAMTStore")
#     print("=" * 80)

#     rpc_base_url = "https://ipfs-gateway.dclimate.net"
#     gateway_base_url = "https://ipfs-gateway.dclimate.net"
#     # headers = {
#     #     "X-API-Key": "",
#     # }
#     headers = {}

#     async with KuboCAS(
#         rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url, headers=headers
#     ) as kubo_cas:
#         root_cid = "bafyr4ialorauxcpw77mgmnyoeptn4g4zkqdqhtsobff4v76rllvd3m6cqi"
#         # root_node_id = CID.decode(root_cid)

#         hamt = await HAMT.build(
#             cas=kubo_cas, root_node_id=root_cid, values_are_bytes=True, read_only=True
#         )
#         # start = time.perf_counter()
#         ipfs_ds: xr.Dataset
#         zhs = ZarrHAMTStore(hamt, read_only=True)
#         ipfs_ds = xr.open_zarr(store=zhs)

#         # --- Read ---
#         hamt = HAMT(
#             cas=kubo_cas, values_are_bytes=True, root_node_id=root_cid, read_only=True
#         )

#         # Initialize the store
#         zhs = ZarrHAMTStore(hamt, read_only=True)

#         start_read = time.perf_counter()
#         ipfs_ds = xr.open_zarr(store=zhs)
#         # Uncomment the next line to read only a subset of the data for performance testing
#         # data_fetched = ipfs_ds.isel(time=slice(0, len(ipfs_ds.time) // 8)).precip.values
#         # Force a read of some data to ensure it's loaded
#         data_fetched = ipfs_ds.precip.values

#         # Calculate the size of the fetched data
#         data_size = data_fetched.nbytes if data_fetched is not None else 0
#         print(f"Fetched data size: {data_size / (1024 * 1024):.4f} MB")
#         end_read = time.perf_counter()

#         print("\n--- [HAMT] Read Stats ---")
#         print(f"Total time to open and read: {end_read - start_read:.2f} seconds")

#         if data_size > 0:
#             speed = data_size / (end_read - start_read) / (1024 * 1024)
#             print(f"Read speed: {speed:.2f} MB/s")
#         else:
#             print("No data fetched, cannot calculate speed.")
