import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.core.buffer

# Make sure to import the ByteRequest types
from zarr.abc.store import OffsetByteRequest, RangeByteRequest, SuffixByteRequest

from py_hamt import HAMT, InMemoryCAS, KuboCAS
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
        test_ds.to_zarr(store=zhs, chunk_store={"time": 50, "lat": 18, "lon": 36})  # type: ignore
        test_ds.to_zarr(store=zhs, mode="a", append_dim="time", zarr_format=3)  # type: ignore
        end = time.perf_counter()
        elapsed = end - start
        print("=== Write Stats")
        print(f"Total time in seconds: {elapsed:.2f}")
        print("=== Root CID")
        await hamt.make_read_only()
        cid = hamt.root_node_id

        print(f"=== Verifying Gateway Suffix Support (CID: {cid}) ===")
        # Get the gateway URL without the /ipfs/ part

        print("=== Reading data back in and checking if identical")
        hamt_read = await HAMT.build(  # Renamed to avoid confusion
            cas=kubo_cas, root_node_id=cid, values_are_bytes=True, read_only=True
        )
        start = time.perf_counter()
        zhs_read = ZarrHAMTStore(hamt_read, read_only=True)  # Use the read-only hamt
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
        range_req = RangeByteRequest(start=10, end=50)  # Request 40 bytes
        offset_req = OffsetByteRequest(offset=chunk_len - 30)  # Request last 30 bytes
        suffix_req = SuffixByteRequest(suffix=20)  # Request last 20 bytes

        key_ranges_to_test = [
            (chunk_key, range_req),
            (chunk_key, offset_req),
            (chunk_key, suffix_req),
            (chunk_key, None),  # Full read
        ]

        # Call get_partial_values
        results = await zhs_read.get_partial_values(proto, key_ranges_to_test)  # type: ignore

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
        expected_offset = full_chunk_data[chunk_len - 30 :]
        assert results[1].to_bytes() == expected_offset, "OffsetByteRequest failed"
        print(f"OffsetByteRequest: OK (Got {len(results[1].to_bytes())} bytes)")

        # Check SuffixByteRequest result
        expected_suffix = full_chunk_data[-20:]
        # Broken until Kubo 0.36.0
        assert results[2].to_bytes() == expected_suffix, "SuffixByteRequest failed"
        print(f"SuffixByteRequest: OK (Got {len(results[2].to_bytes())} bytes)")

        # Check full read result
        assert results[3].to_bytes() == full_chunk_data, (
            "Full read via get_partial_values failed"
        )
        print(f"Full Read: OK (Got {len(results[3].to_bytes())} bytes)")

        # --- End: New Partial Values Tests ---

        # Tests for code coverage's sake
        assert await zhs_read.exists("zarr.json")
        # __eq__
        assert zhs_read == zhs_read
        assert zhs_read != hamt_read
        assert not zhs_read.supports_writes
        assert not zhs_read.supports_partial_writes
        assert not (
            zhs_read.supports_deletes
        )  # Should be true in read-only mode for HAMT? Usually False

        hamt_keys = set()
        async for k in zhs_read.hamt.keys():
            hamt_keys.add(k)

        zhs_keys: set[str] = set()
        async for k in zhs_read.list():
            zhs_keys.add(k)
        assert hamt_keys == zhs_keys

        zhs_keys_prefix: set[str] = set()
        async for k in zhs_read.list_prefix(""):
            zhs_keys_prefix.add(k)
        assert hamt_keys == zhs_keys_prefix

        with pytest.raises(NotImplementedError):
            await zhs_read.set_partial_values([])


@pytest.mark.asyncio
async def test_zarr_hamt_store_byte_request_errors():
    """Tests error handling for unsupported or invalid ByteRequest types."""
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    zhs = ZarrHAMTStore(hamt)
    proto = zarr.core.buffer.default_buffer_prototype()
    await zhs.set("some_key", proto.buffer.from_bytes(b"0123456789"))

    # Test for ValueError with an invalid range (end < start)
    invalid_range_req = RangeByteRequest(start=10, end=5)
    with pytest.raises(ValueError, match="End must be >= start for RangeByteRequest"):
        await zhs.get("some_key", proto, byte_range=invalid_range_req)

    # Test for TypeError with a custom, unsupported request type
    class DummyUnsupportedRequest:
        pass

    unsupported_req = DummyUnsupportedRequest()
    with pytest.raises(TypeError, match="Unsupported ByteRequest type"):
        await zhs.get("some_key", proto, byte_range=unsupported_req)


@pytest.mark.asyncio
async def test_ipfs_gateway_compression_behavior(create_ipfs):
    """
    Test to verify whether IPFS gateways decompress data before applying
    byte range requests, which would negate compression benefits for partial reads.

    This test creates highly compressible data, stores it via IPFS, and then
    compares the bytes returned by partial vs full reads to determine if
    the gateway is operating on compressed or decompressed data.
    """
    rpc_base_url, gateway_base_url = create_ipfs

    print("\n=== IPFS Gateway Compression Behavior Test ===")

    # Create highly compressible test data
    print("Creating highly compressible test data...")
    data = np.zeros((50, 50, 50), dtype=np.float32)  # 500KB of zeros
    # Add small amount of variation
    data[0:5, 0:5, 0:5] = np.random.randn(5, 5, 5)

    ds = xr.Dataset({"compressible_var": (["x", "y", "z"], data)})

    print(f"Original data shape: {data.shape}")
    print(f"Original data size: {data.nbytes:,} bytes")

    # Custom CAS to track actual network transfers
    class NetworkTrackingKuboCAS(KuboCAS):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.load_sizes = {}
            self.save_sizes = {}

        async def save(self, data, codec=None):
            cid = await super().save(data, codec)
            self.save_sizes[str(cid)] = len(data)
            print(f"Saved to IPFS: {str(cid)[:12]}... ({len(data):,} bytes)")
            return cid

        async def load(self, cid, offset=None, length=None, suffix=None):
            result = await super().load(cid, offset, length, suffix)

            range_desc = "full"
            if offset is not None or length is not None or suffix is not None:
                range_desc = f"offset={offset}, length={length}, suffix={suffix}"

            key = f"{str(cid)[:12]}... ({range_desc})"
            self.load_sizes[key] = len(result)
            print(f"Loaded from IPFS: {key} -> {len(result):,} bytes")
            return result

    async with NetworkTrackingKuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as tracking_cas:
        # Write dataset with compression
        print("\n=== Writing to IPFS with Blosc compression ===")
        hamt = await HAMT.build(cas=tracking_cas, values_are_bytes=True)
        store = ZarrHAMTStore(hamt)

        # Use smaller chunks to ensure meaningful compression
        ds.chunk({"x": 25, "y": 25, "z": 25}).to_zarr(
            store=store, mode="w", zarr_format=3
        )

        await hamt.make_read_only()
        root_cid = hamt.root_node_id
        print(f"Root CID: {root_cid}")

        # Read back and test compression behavior
        print("\n=== Testing Compression vs Partial Reads ===")
        hamt_read = await HAMT.build(
            cas=tracking_cas,
            root_node_id=root_cid,
            values_are_bytes=True,
            read_only=True,
        )
        store_read = ZarrHAMTStore(hamt_read, read_only=True)

        # Find the largest data chunk (likely the actual array data)
        chunk_key = None
        chunk_size = 0
        async for key in store_read.list():
            if (
                "compressible_var" in key
                and not key.endswith(".zarray")
                and not key.endswith("zarr.json")
            ):
                # Get size to find the largest chunk
                proto = zarr.core.buffer.default_buffer_prototype()
                chunk_buffer = await store_read.get(key, proto)
                if chunk_buffer and len(chunk_buffer.to_bytes()) > chunk_size:
                    chunk_key = key
                    chunk_size = len(chunk_buffer.to_bytes())

        assert chunk_key is not None, "No data chunk found"
        print(f"Testing with largest chunk: {chunk_key}")

        # Get full chunk for baseline
        proto = zarr.core.buffer.default_buffer_prototype()
        full_chunk = await store_read.get(chunk_key, proto)
        full_compressed_size = len(full_chunk.to_bytes())
        print(f"Full chunk compressed size: {full_compressed_size:,} bytes")

        # Calculate expected uncompressed size
        # 25x25x25 float32 = 62,500 bytes uncompressed
        expected_uncompressed_size = 25 * 25 * 25 * 4
        compression_ratio = expected_uncompressed_size / full_compressed_size
        print(f"Estimated compression ratio: {compression_ratio:.1f}:1")

        # Test different partial read sizes
        test_ranges = [
            (0, full_compressed_size // 4, "25% of compressed"),
            (0, full_compressed_size // 2, "50% of compressed"),
            (full_compressed_size // 4, full_compressed_size // 2, "25%-50% range"),
        ]

        print("\n=== Partial Read Analysis ===")
        print("If gateway operates on compressed data:")
        print("  - Partial reads should return exactly the requested byte ranges")
        print("  - Network transfer should be proportional to compressed size")
        print("If gateway decompresses before range requests:")
        print("  - Partial reads may return more data than expected")
        print("  - Network transfer loses compression benefits")
        print()

        compression_preserved = True

        for start, end, description in test_ranges:
            length = end - start
            byte_req = RangeByteRequest(start=start, end=end)

            # Clear the load tracking for this specific test
            original_load_count = len(tracking_cas.load_sizes)

            partial_chunk = await store_read.get(chunk_key, proto, byte_range=byte_req)
            partial_size = len(partial_chunk.to_bytes())

            # Find the new load entry
            new_loads = list(tracking_cas.load_sizes.items())[original_load_count:]
            network_bytes = new_loads[0][1] if new_loads else partial_size

            expected_percentage = length / full_compressed_size
            actual_percentage = partial_size / full_compressed_size
            network_efficiency = network_bytes / full_compressed_size

            print(f"Range request: {description}")
            print(
                f"  Requested: {length:,} bytes ({expected_percentage:.1%} of compressed)"
            )
            print(
                f"  Received: {partial_size:,} bytes ({actual_percentage:.1%} of compressed)"
            )
            print(
                f"  Network transfer: {network_bytes:,} bytes ({network_efficiency:.1%} of compressed)"
            )

            # Key test: if we get significantly more data than requested,
            # the gateway is likely decompressing before applying ranges
            if partial_size > length * 1.1:  # 10% tolerance for overhead
                compression_preserved = False
                print(
                    f"  ⚠️  Received {partial_size / length:.1f}x more data than requested!"
                )
                print("  ⚠️  Gateway appears to decompress before applying ranges")
            else:
                print("  ✅ Range applied efficiently to compressed data")

            # Verify the partial data makes sense
            full_data = full_chunk.to_bytes()
            expected_partial = full_data[start:end]
            assert partial_chunk.to_bytes() == expected_partial, (
                "Partial data doesn't match expected range"
            )
            print("  ✅ Partial data content verified")
            print()

        # Summary analysis
        print("=== Final Analysis ===")
        if compression_preserved:
            print("✅ IPFS gateway preserves compression benefits for partial reads")
            print("   - Byte ranges are applied to compressed data")
            print("   - Network transfers are efficient")
        else:
            print("⚠️  IPFS gateway appears to decompress before applying ranges")
            print("   - Partial reads may not provide expected bandwidth savings")
            print("   - Consider alternative storage strategies (sharding, etc.)")

        print("\nCompression statistics:")
        print(f"  - Uncompressed chunk size: {expected_uncompressed_size:,} bytes")
        print(f"  - Compressed chunk size: {full_compressed_size:,} bytes")
        print(f"  - Compression ratio: {compression_ratio:.1f}:1")
        print(f"  - Compression preserved in partial reads: {compression_preserved}")


@pytest.mark.asyncio
async def test_in_memory_cas_partial_reads():
    """
    Tests the partial read logic (offset, length, suffix) in the InMemoryCAS.
    """
    cas = InMemoryCAS()
    test_data = b"0123456789abcdefghijklmnopqrstuvwxyz"  # 36 bytes long
    data_id = await cas.save(test_data, "raw")

    # Test case 1: offset and length
    result = await cas.load(data_id, offset=10, length=5)
    assert result == b"abcde"

    # Test case 2: offset only
    result = await cas.load(data_id, offset=30)
    assert result == b"uvwxyz"

    # Test case 3: suffix only
    result = await cas.load(data_id, suffix=6)
    assert result == b"uvwxyz"

    # Test case 4: Full read (for completeness)
    result = await cas.load(data_id)
    assert result == test_data

    # Test case 5: Key not found (covers `try...except KeyError`)
    with pytest.raises(KeyError, match="Object not found in in-memory store"):
        await cas.load(b"\x00\x01\x02\x03\x04")  # Some random, non-existent key

    # Test case 6: Invalid key type (covers `isinstance` check)
    with pytest.raises(TypeError, match="InMemoryCAS only supports byte‐hash keys"):
        await cas.load(12345)  # Pass an integer instead of bytes
