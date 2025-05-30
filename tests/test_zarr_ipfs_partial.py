import time

import numpy as np
import pandas as pd
import xarray as xr
import pytest
import zarr
import zarr.core.buffer
# Make sure to import the ByteRequest types
from zarr.abc.store import RangeByteRequest, OffsetByteRequest, SuffixByteRequest
import aiohttp
from typing import Optional



from py_hamt import HAMT, KuboCAS

from py_hamt.zarr_hamt_store import ZarrHAMTStore


@pytest.fixture(scope="module")
def random_zarr_dataset():
    """Creates a random xarray Dataset.

    Returns:
        tuple: (dataset_path, expected_data)
            - dataset_path: Path to the zarr store
            - expected_data: The original xarray Dataset for comparison
    """
    # Coordinates of the random data
    times = pd.date_range("2024-01-01", periods=100)
    lats = np.linspace(-90, 90, 18)
    lons = np.linspace(-180, 180, 36)

    # Create random variables with different shapes
    temp = np.random.randn(len(times), len(lats), len(lons))
    precip = np.random.gamma(2, 0.5, size=(len(times), len(lats), len(lons)))

    # Create the dataset
    ds = xr.Dataset(
        {
            "temp": (
                ["time", "lat", "lon"],
                temp,
                {"units": "celsius", "long_name": "Surface Temperature"},
            ),
            "precip": (
                ["time", "lat", "lon"],
                precip,
                {"units": "mm/day", "long_name": "Daily Precipitation"},
            ),
        },
        coords={
            "time": times,
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={"description": "Test dataset with random weather data"},
    )

    yield ds

# This test also collects miscellaneous statistics about performance, run with pytest -s to see these statistics being printed out
@pytest.mark.asyncio(loop_scope="session")  # ← match the loop of the fixture
async def test_write_read(
    create_ipfs,
    random_zarr_dataset: xr.Dataset,
):  # noqa for fixture which is imported above but then "redefined"
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset
    print("=== Writing this xarray Dataset to a Zarr v3 on IPFS ===")
    print(test_ds)


    async with KuboCAS(  # <-- own and auto-close session
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        hamt = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        zhs = ZarrHAMTStore(hamt)
        assert zhs.supports_writes
        start = time.perf_counter()
        # Do an initial write along with an append which is a common xarray/zarr operation
        # Ensure chunks are not too small for partial value tests
        test_ds.to_zarr(store=zhs, chunk_store={'time': 50, 'lat': 18, 'lon': 36})
        test_ds.to_zarr(store=zhs, mode="a", append_dim="time", zarr_format=3)
        end = time.perf_counter()
        elapsed = end - start
        print("=== Write Stats")
        print(f"Total time in seconds: {elapsed:.2f}")
        print("=== Root CID")
        await hamt.make_read_only()
        cid = hamt.root_node_id

        print(f"=== Verifying Gateway Suffix Support (CID: {cid}) ===")
        # Get the gateway URL without the /ipfs/ part
        gateway_only_url = gateway_base_url 
        
        # You can add an assertion here if you expect it to work
        # If you know the gateway *might* be buggy, just printing is okay too.
        assert is_correct, "IPFS Gateway did not return the correct suffix data."

        print("=== Reading data back in and checking if identical")
        hamt_read = await HAMT.build( # Renamed to avoid confusion
            cas=kubo_cas, root_node_id=cid, values_are_bytes=True, read_only=True
        )
        start = time.perf_counter()
        zhs_read = ZarrHAMTStore(hamt_read, read_only=True) # Use the read-only hamt
        ipfs_ds = xr.open_zarr(store=zhs_read)
        print(ipfs_ds)

        # Check both halves, since each are an identical copy
        ds1 = ipfs_ds.isel(time=slice(0, len(ipfs_ds.time) // 2))
        ds2 = ipfs_ds.isel(time=slice(len(ipfs_ds.time) // 2, len(ipfs_ds.time)))
        xr.testing.assert_identical(ds1, ds2)
        xr.testing.assert_identical(test_ds, ds1)
        xr.testing.assert_identical(test_ds, ds2)

        end = time.perf_counter()
        elapsed = end - start
        print("=== Read Stats")
        print(f"Total time in seconds: {elapsed:.2f}")

        # --- Start: New Partial Values Tests ---

        print("=== Testing get_partial_values ===")
        proto = zarr.core.buffer.default_buffer_prototype()

        # Find a chunk key to test with (e.g., the first chunk of 'temp')
        chunk_key = None
        async for k in zhs_read.list():
            if k.startswith("temp/") and k != "temp/.zarray":
                chunk_key = k
                break
        
        assert chunk_key is not None, "Could not find a chunk key to test."
        print(f"Testing with chunk key: {chunk_key}")

        # Get the full chunk data for comparison
        full_chunk_buffer = await zhs_read.get(chunk_key, proto)
        assert full_chunk_buffer is not None
        full_chunk_data = full_chunk_buffer.to_bytes()
        chunk_len = len(full_chunk_data)
        print(f"Full chunk size: {chunk_len} bytes")
        
        # Ensure the chunk is large enough for meaningful tests
        assert chunk_len > 100, "Chunk size too small for partial value tests"

        # Define some byte requests
        range_req = RangeByteRequest(start=10, end=50) # Request 40 bytes
        offset_req = OffsetByteRequest(offset=chunk_len - 30) # Request last 30 bytes
        suffix_req = SuffixByteRequest(suffix=20) # Request last 20 bytes

        key_ranges_to_test = [
            (chunk_key, range_req),
            (chunk_key, offset_req),
            (chunk_key, suffix_req),
            (chunk_key, None), # Full read
        ]

        # Call get_partial_values
        results = await zhs_read.get_partial_values(proto, key_ranges_to_test)

        # Assertions
        assert len(results) == 4
        assert results[0] is not None
        assert results[1] is not None
        assert results[2] is not None
        assert results[3] is not None

        # Check RangeByteRequest result
        expected_range = full_chunk_data[10:50]
        assert results[0].to_bytes() == expected_range, "RangeByteRequest failed"
        print(f"RangeByteRequest: OK (Got {len(results[0].to_bytes())} bytes)")

        # Check OffsetByteRequest result
        expected_offset = full_chunk_data[chunk_len - 30:]
        assert results[1].to_bytes() == expected_offset, "OffsetByteRequest failed"
        print(f"OffsetByteRequest: OK (Got {len(results[1].to_bytes())} bytes)")

        # Check SuffixByteRequest result
        expected_suffix = full_chunk_data[-20:]
        # Broken until Kubo 0.36.0
        assert results[2].to_bytes() == expected_suffix, "SuffixByteRequest failed"
        print(f"SuffixByteRequest: OK (Got {len(results[2].to_bytes())} bytes)")

        # Check full read result
        assert results[3].to_bytes() == full_chunk_data, "Full read via get_partial_values failed"
        print(f"Full Read: OK (Got {len(results[3].to_bytes())} bytes)")


        # --- End: New Partial Values Tests ---


        # Tests for code coverage's sake
        assert await zhs_read.exists("zarr.json")
        # __eq__
        assert zhs_read == zhs_read
        assert zhs_read != hamt_read
        assert not zhs_read.supports_writes
        assert not zhs_read.supports_partial_writes
        assert zhs_read.supports_deletes # Should be true in read-only mode for HAMT? Usually False

        hamt_keys = set()
        async for k in zhs_read.hamt.keys():
            hamt_keys.add(k)

        zhs_keys: set[str] = set()
        async for k in zhs_read.list():
            zhs_keys.add(k)
        assert hamt_keys == zhs_keys

        zhs_keys: set[str] = set()
        async for k in zhs_read.list_prefix(""):
            zhs_keys.add(k)
        assert hamt_keys == zhs_keys

        with pytest.raises(NotImplementedError):
            await zhs_read.set_partial_values([])

        # REMOVED: The old NotImplementedError check for get_partial_values
        # with pytest.raises(NotImplementedError):
        #     await zhs_read.get_partial_values(
        #         zarr.core.buffer.default_buffer_prototype(), []
        #     )

        previous_zarr_json = await zhs_read.get(
            "zarr.json", zarr.core.buffer.default_buffer_prototype()
        )
        assert previous_zarr_json is not None

        # --- Test set_if_not_exists (needs a writable store) ---
        await hamt_read.enable_write()
        zhs_write = ZarrHAMTStore(hamt_read, read_only=False)

        # Setting a metadata file that should always exist should not change anything
        await zhs_write.set_if_not_exists(
            "zarr.json",
            zarr.core.buffer.Buffer.from_bytes(b"should_not_change"),
        )
        zarr_json_now = await zhs_write.get(
            "zarr.json", zarr.core.buffer.default_buffer_prototype()
        )
        assert zarr_json_now is not None
        assert previous_zarr_json.to_bytes() == zarr_json_now.to_bytes()

        # now remove that metadata file and then add it back
        await zhs_write.delete("zarr.json")
        # doing a duplicate delete should not result in an error
        await zhs_write.delete("zarr.json")
        
        zhs_keys_after_delete: set[str] = set()
        async for k in zhs_write.list():
            zhs_keys_after_delete.add(k)
        assert hamt_keys != zhs_keys_after_delete
        assert "zarr.json" not in zhs_keys_after_delete

        await zhs_write.set_if_not_exists("zarr.json", previous_zarr_json)
        zarr_json_now = await zhs_write.get(
            "zarr.json", zarr.core.buffer.default_buffer_prototype()
        )
        assert zarr_json_now is not None
        assert previous_zarr_json.to_bytes() == zarr_json_now.to_bytes()