import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.core.buffer
from Crypto.Random import get_random_bytes

from py_hamt import HAMT, KuboCAS, SimpleEncryptedZarrHAMTStore
from py_hamt.zarr_hamt_store import ZarrHAMTStore


@pytest.fixture(scope="module")
def random_zarr_dataset():
    """Creates a random xarray Dataset."""
    times = pd.date_range("2024-01-01", periods=10)  # Reduced size for faster tests
    lats = np.linspace(-90, 90, 5)
    lons = np.linspace(-180, 180, 10)
    temp = np.random.randn(len(times), len(lats), len(lons))
    ds = xr.Dataset(
        {"temp": (["time", "lat", "lon"], temp)},
        coords={"time": times, "lat": lats, "lon": lons},
        attrs={"description": "Test dataset"},
    )
    yield ds


async def test_invalid_encryption_key_length():
    """
    Tests that SimpleEncryptedZarrHAMTStore raises a ValueError
    if the encryption key is not exactly 32 bytes long.
    """
    async with KuboCAS() as kubo_cas:
        hamt_write = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        header = b"test-header"
        # --- Test with a key that is too short (31 bytes) ---
        short_key = get_random_bytes(31)
        with pytest.raises(
            ValueError, match="Encryption key must be exactly 32 bytes long."
        ):
            print("\nTesting with a 31-byte key...")
            SimpleEncryptedZarrHAMTStore(hamt_write, False, short_key, header)
            print("Failed as expected.")

        # --- Test with a key that is too long (33 bytes) ---
        long_key = get_random_bytes(33)
        with pytest.raises(
            ValueError, match="Encryption key must be exactly 32 bytes long."
        ):
            print("Testing with a 33-byte key...")
            SimpleEncryptedZarrHAMTStore(hamt_write, False, long_key, header)
            print("Failed as expected.")

        # --- Test with an empty key ---
        empty_key = b""
        with pytest.raises(
            ValueError, match="Encryption key must be exactly 32 bytes long."
        ):
            print("Testing with an empty key...")
            SimpleEncryptedZarrHAMTStore(hamt_write, False, empty_key, header)
            print("Failed as expected.")

        # --- Test with a correct key (should NOT raise) ---
        correct_key = get_random_bytes(32)
        try:
            print("Testing with a 32-byte key...")
            SimpleEncryptedZarrHAMTStore(hamt_write, False, correct_key, header)
            print("Initialized successfully (as expected).")
        except ValueError as e:
            pytest.fail(f"Should NOT raise ValueError for a 32-byte key, but got: {e}")


# Assume create_ipfs fixture exists and provides (rpc_base_url, gateway_base_url)
# You might need to add: @pytest.mark.asyncio(loop_scope="session")
@pytest.mark.asyncio
async def test_encrypted_write_read(
    create_ipfs,  # This fixture needs to be available in your conftest.py
    random_zarr_dataset: xr.Dataset,
):
    """
    Tests writing and reading a Zarr dataset using SimpleEncryptedZarrHAMTStore.
    It covers successful reads, reads with incorrect keys, and reads
    attempted without decryption.
    """
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset

    # --- Setup Encryption ---
    correct_key = get_random_bytes(32)
    wrong_key = get_random_bytes(32)
    header = b"test-encryption-header"

    root_cid = None

    # --- Write Phase ---
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        print("\n--- Writing Encrypted Zarr ---")
        hamt_write = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        ezhs_write = SimpleEncryptedZarrHAMTStore(
            hamt_write, False, correct_key, header
        )

        # __eq__
        assert ezhs_write == ezhs_write
        assert ezhs_write != hamt_write

        assert ezhs_write.supports_writes
        test_ds.to_zarr(store=ezhs_write, mode="w", zarr_format=3)  # Use mode='w'

        await hamt_write.make_read_only()
        root_cid = hamt_write.root_node_id
        print(f"Encrypted Root CID: {root_cid}")
        assert root_cid is not None

    # --- Read Phase ---
    async with KuboCAS(
        rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url
    ) as kubo_cas:
        print("\n--- Reading Encrypted Zarr (Correct Key) ---")
        hamt_read_ok = await HAMT.build(
            cas=kubo_cas, root_node_id=root_cid, values_are_bytes=True, read_only=True
        )
        ezhs_read_ok = SimpleEncryptedZarrHAMTStore(
            hamt_read_ok, True, correct_key, header
        )

        ds_read_ok = xr.open_zarr(store=ezhs_read_ok)
        print("Read successful. Verifying data...")
        xr.testing.assert_identical(test_ds, ds_read_ok)
        print("Data verified.")

        # Open another zarr with the same store to ensure caching works
        ds_check_again = xr.open_zarr(store=ezhs_read_ok)
        xr.testing.assert_identical(test_ds, ds_check_again)

        # --- Test: Read with Incorrect Key ---
        print("\n--- Reading Encrypted Zarr (Incorrect Key) ---")
        hamt_read_bad_key = await HAMT.build(
            cas=kubo_cas, root_node_id=root_cid, values_are_bytes=True, read_only=True
        )
        ezhs_read_bad_key = SimpleEncryptedZarrHAMTStore(
            hamt_read_bad_key, True, wrong_key, header
        )

        with pytest.raises(ValueError, match="Decryption failed"):
            xr.open_zarr(store=ezhs_read_bad_key)
        print("Read failed as expected (ValueError).")

        # --- Test: Read without Decryption ("No Key") ---
        print("\n--- Reading Encrypted Zarr (No Decryption) ---")
        hamt_read_no_key = await HAMT.build(
            cas=kubo_cas, root_node_id=root_cid, values_are_bytes=True, read_only=True
        )
        # Use the *non-encrypted* store
        zhs_read_no_key = ZarrHAMTStore(hamt_read_no_key, read_only=True)

        # Expect failure when trying to parse encrypted 'zarr.json' as JSON
        with pytest.raises(Exception) as excinfo:
            xr.open_zarr(store=zhs_read_no_key)
        # Check if it's a JSON decode error or similar Zarr error.
        # The exact error might vary, so checking for ValueError (from encryption)
        # or JSONDecodeError or Zarr errors covers likely scenarios.
        assert isinstance(
            excinfo.value,
            (
                ValueError,
                json.JSONDecodeError,
            ),
        )
        print(f"Read failed as expected ({type(excinfo.value).__name__}).")

        # --- Basic Coverage Tests ---
        assert await ezhs_read_ok.exists("zarr.json")
        assert not await ezhs_read_ok.exists("non_existent_key")
        assert ezhs_read_ok == ezhs_read_ok  # Test __eq__
        assert ezhs_read_ok != ezhs_read_bad_key
        assert not ezhs_read_ok.supports_writes

        keys_list = [k async for k in ezhs_read_ok.list()]
        assert "zarr.json" in keys_list
        assert "temp/c/0/0/0" in keys_list  # Check for a known chunk key

        prefix_keys = [k async for k in ezhs_read_ok.list_prefix("temp/")]
        assert all(k.startswith("temp/") for k in prefix_keys)
        assert len(prefix_keys) > 0

        dir_list = [k async for k in ezhs_read_ok.list_dir("")]
        assert "zarr.json" in dir_list
        assert "temp" in dir_list

        with pytest.raises(NotImplementedError):
            await ezhs_read_ok.set_partial_values([])

        with pytest.raises(NotImplementedError):
            await ezhs_read_ok.get_partial_values(
                zarr.core.buffer.default_buffer_prototype(), []
            )

        with pytest.raises(Exception):
            await ezhs_read_ok.set("new_key", np.array([b"a"], dtype=np.bytes_))

        with pytest.raises(Exception):
            await ezhs_read_ok.delete("zarr.json")
