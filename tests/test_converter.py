import sys
import time
import uuid
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

# Import store implementations
from py_hamt import (
    HAMT,
    KuboCAS,
    ShardedZarrStore,
)
from py_hamt.hamt_to_sharded_converter import (
    convert_hamt_to_sharded,
    sharded_converter_cli,
)
from py_hamt.zarr_hamt_store import ZarrHAMTStore


@pytest.fixture(scope="module")
def converter_test_dataset():
    """
    Creates a random, uniquely-named xarray Dataset specifically for the converter test.
    Using a unique variable name helps avoid potential caching issues between test runs.
    """
    # A smaller dataset is fine for a verification test
    times = pd.date_range("2025-01-01", periods=20)
    lats = np.linspace(40, 50, 10)
    lons = np.linspace(-85, -75, 20)

    # Generate a unique variable name for this test run
    unique_var_name = f"data_{str(uuid.uuid4())[:8]}"

    data = np.random.randn(len(times), len(lats), len(lons))

    ds = xr.Dataset(
        {unique_var_name: (["time", "latitude", "longitude"], data)},
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={"description": "Test dataset for converter verification."},
    )

    # Define chunking for the store
    ds = ds.chunk({"time": 10, "latitude": 10, "longitude": 10})
    yield ds


@pytest.mark.asyncio(loop_scope="session")
async def test_converter_produces_identical_dataset(
    create_ipfs: tuple[str, str],
    converter_test_dataset: xr.Dataset,
):
    """
    Tests the hamt_to_sharded_converter by performing a full conversion
    and verifying that the resulting dataset is identical to the source.
    """
    print("\n\n" + "=" * 80)
    print("🚀 STARTING TEST for HAMT to Sharded Converter")
    print("=" * 80)

    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = converter_test_dataset
    chunks_per_shard_config = 64  # A reasonable value for this test size

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # --------------------------------------------------------------------
        # STEP 1: Create the source HAMT store from our test dataset
        # --------------------------------------------------------------------
        print("\n--- STEP 1: Creating source HAMT store ---")
        hamt_write = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        source_hamt_store = ZarrHAMTStore(hamt_write)

        start_write = time.perf_counter()
        test_ds.to_zarr(store=source_hamt_store, mode="w")
        await hamt_write.make_read_only()  # Flush to get the final CID
        end_write = time.perf_counter()

        hamt_root_cid = str(hamt_write.root_node_id)
        print(f"Source HAMT store created in {end_write - start_write:.2f}s")
        print(f"Source HAMT Root CID: {hamt_root_cid}")

        # --------------------------------------------------------------------
        # STEP 2: Run the conversion script
        # --------------------------------------------------------------------
        print("\n--- STEP 2: Running conversion script ---")
        sharded_root_cid = await convert_hamt_to_sharded(
            cas=kubo_cas,
            hamt_root_cid=hamt_root_cid,
            chunks_per_shard=chunks_per_shard_config,
        )
        print("Conversion script finished.")
        print(f"New Sharded Store Root CID: {sharded_root_cid}")
        assert sharded_root_cid is not None

        # --------------------------------------------------------------------
        # STEP 3: Verification
        # --------------------------------------------------------------------
        print("\n--- STEP 3: Verifying data integrity ---")

        # Open the original dataset from the HAMT store
        print("Reading data back from original HAMT store...")

        hamt_ro = await HAMT.build(
            cas=kubo_cas,
            root_node_id=hamt_root_cid,
            values_are_bytes=True,
            read_only=True,
        )
        zhs_ro = ZarrHAMTStore(hamt_ro, read_only=True)

        start_read = time.perf_counter()
        ds_from_hamt = xr.open_zarr(store=zhs_ro)

        end_read = time.perf_counter()
        print(f"Original HAMT store read in {end_read - start_read:.2f}s")

        # Open the converted dataset from the new Sharded store
        print("Reading data back from new Sharded store...")
        dest_store_ro = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=sharded_root_cid
        )
        ds_from_sharded = xr.open_zarr(dest_store_ro)

        # The ultimate test: are the two xarray.Dataset objects identical?
        # This checks coordinates, variables, data values, and attributes.
        print("Comparing the two datasets...")
        xr.testing.assert_identical(ds_from_hamt, ds_from_sharded)
        # Ask for random samples from both datasets to ensure they match
        for var in ds_from_hamt.data_vars:
            # Assert all identical
            np.testing.assert_array_equal(
                ds_from_hamt[var].values, ds_from_sharded[var].values
            )

        print("\n✅ Verification successful! The datasets are identical.")
        print("=" * 80)


@pytest.mark.asyncio(loop_scope="session")
async def test_hamt_to_sharded_cli_success(
    create_ipfs: tuple[str, str], converter_test_dataset: xr.Dataset, capsys
):
    """
    Tests the CLI for successful conversion of a HAMT store to a ShardedZarrStore.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = converter_test_dataset

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Step 1: Create a HAMT store with the test dataset
        hamt_write = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        source_hamt_store = ZarrHAMTStore(hamt_write)
        test_ds.to_zarr(store=source_hamt_store, mode="w", consolidated=True)
        await hamt_write.make_read_only()
        hamt_root_cid = str(hamt_write.root_node_id)

        # Step 2: Simulate CLI execution with valid arguments
        test_args = [
            "script.py",  # Dummy script name
            hamt_root_cid,
            "--chunks-per-shard",
            "64",
            "--rpc-url",
            rpc_base_url,
            "--gateway-url",
            gateway_base_url,
        ]
        with patch.object(sys, "argv", test_args):
            await sharded_converter_cli()

        # Step 3: Capture and verify CLI output
        captured = capsys.readouterr()
        assert "Starting Conversion from HAMT Root" in captured.out
        assert "Conversion Complete!" in captured.out
        assert "New ShardedZarrStore Root CID" in captured.out

        # Step 4: Verify the converted dataset
        # Extract the new root CID from output (assuming it's the last line)
        output_lines = captured.out.strip().split("\n")
        new_root_cid = output_lines[-1].split(": ")[-1]
        dest_store_ro = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=new_root_cid
        )
        ds_from_sharded = xr.open_zarr(dest_store_ro)
        xr.testing.assert_identical(test_ds, ds_from_sharded)


@pytest.mark.asyncio(loop_scope="session")
async def test_hamt_to_sharded_cli_default_args(
    create_ipfs: tuple[str, str], converter_test_dataset: xr.Dataset, capsys
):
    """
    Tests the CLI with default argument values.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = converter_test_dataset

    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        # Create a HAMT store
        hamt_write = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        source_hamt_store = ZarrHAMTStore(hamt_write)
        test_ds.to_zarr(store=source_hamt_store, mode="w", consolidated=True)
        await hamt_write.make_read_only()
        hamt_root_cid = str(hamt_write.root_node_id)

        # Simulate CLI with only hamt_cid and gateway URLs.
        test_args = [
            "script.py",  # Dummy script name
            hamt_root_cid,
            "--rpc-url",
            rpc_base_url,
            "--gateway-url",
            gateway_base_url,
        ]
        with patch.object(sys, "argv", test_args):
            await sharded_converter_cli()

        # Verify output and conversion
        captured = capsys.readouterr()
        output_lines = captured.out.strip().split("\n")
        print("Captured CLI Output:")
        for line in output_lines:
            print(line)
        new_root_cid = output_lines[-1].split(": ")[-1]
        dest_store_ro = await ShardedZarrStore.open(
            cas=kubo_cas, read_only=True, root_cid=new_root_cid
        )
        ds_from_sharded = xr.open_zarr(dest_store_ro)
        xr.testing.assert_identical(test_ds, ds_from_sharded)


@pytest.mark.asyncio(loop_scope="session")
async def test_hamt_to_sharded_cli_invalid_cid(create_ipfs: tuple[str, str], capsys):
    """
    Tests the CLI with an invalid hamt_cid.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    invalid_cid = "invalid_cid"

    test_args = [
        "script.py",
        invalid_cid,
        "--chunks-per-shard",
        "64",
        "--rpc-url",
        rpc_base_url,
        "--gateway-url",
        gateway_base_url,
    ]
    with patch.object(sys, "argv", test_args):
        await sharded_converter_cli()

    # Verify error handling
    captured = capsys.readouterr()
    assert "An error occurred" in captured.out
    assert f"{invalid_cid}" in captured.out
