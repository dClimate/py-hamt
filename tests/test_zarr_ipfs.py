import os
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import xarray as xr
import pytest
import zarr.core.buffer

from py_hamt import HAMT, IPFSStore, IPFSZarr3, create_zarr_encryption_transformers


@pytest.fixture(scope="module")
def random_zarr_dataset():
    """Creates a random xarray Dataset and saves it to a temporary zarr store.

    Returns:
        tuple: (dataset_path, expected_data)
            - dataset_path: Path to the zarr store
            - expected_data: The original xarray Dataset for comparison
    """
    # Create temporary directory for zarr store
    temp_dir = tempfile.mkdtemp()
    zarr_path = os.path.join(temp_dir, "test.zarr")

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

    ds.to_zarr(zarr_path, mode="w")

    yield zarr_path, ds

    # Cleanup
    shutil.rmtree(temp_dir)


# This test also collects miscellaneous statistics about performance, run with pytest -s to see these statistics being printed out
@pytest.mark.asyncio
async def test_write_read(random_zarr_dataset: tuple[str, xr.Dataset]):
    _, test_ds = random_zarr_dataset
    print("=== Writing this xarray Dataset to a Zarr v3 on IPFS ===")
    print(test_ds)

    ipfsstore = IPFSStore(debug=True)
    hamt = HAMT(store=ipfsstore)
    ipfszarr3 = IPFSZarr3(hamt)
    assert ipfszarr3.supports_writes
    start = time.perf_counter()
    # Do an initial write along with an append
    test_ds.to_zarr(store=ipfszarr3)  # type: ignore
    test_ds.to_zarr(store=ipfszarr3, mode="a", append_dim="time")  # type: ignore
    end = time.perf_counter()
    elapsed = end - start
    print("=== Write Stats")
    print(f"Total time in seconds: {elapsed:.2f}")
    print(f"Sent bytes: {ipfsstore.total_sent}")
    print(f"Received bytes: {ipfsstore.total_received}")
    print("=== Root CID")
    cid = hamt.root_node_id
    print(cid)

    print("=== Reading data back in and checking if identical")
    ipfsstore = IPFSStore(debug=True)
    hamt = HAMT(store=ipfsstore, root_node_id=cid)
    start = time.perf_counter()
    ipfs_ds: xr.Dataset
    ipfszarr3 = IPFSZarr3(hamt, read_only=True)
    ipfs_ds = xr.open_zarr(store=ipfszarr3)
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
    print(f"Sent bytes: {ipfsstore.total_sent}")
    print(f"Received bytes: {ipfsstore.total_received}")

    # Tests for code coverage's sake
    assert await ipfszarr3.exists("zarr.json")
    # __eq__
    assert ipfszarr3 == ipfszarr3
    assert ipfszarr3 != hamt
    assert not ipfszarr3.supports_writes
    assert not ipfszarr3.supports_partial_writes
    assert not ipfszarr3.supports_deletes

    hamt_keys = set(ipfszarr3.hamt.keys())
    ipfszarr3_keys: set[str] = set()
    async for k in ipfszarr3.list():
        ipfszarr3_keys.add(k)
    assert hamt_keys == ipfszarr3_keys

    ipfszarr3_keys: set[str] = set()
    async for k in ipfszarr3.list():
        ipfszarr3_keys.add(k)
    assert hamt_keys == ipfszarr3_keys

    ipfszarr3_keys: set[str] = set()
    async for k in ipfszarr3.list_prefix(""):
        ipfszarr3_keys.add(k)
    assert hamt_keys == ipfszarr3_keys

    with pytest.raises(NotImplementedError):
        await ipfszarr3.set_partial_values([])

    with pytest.raises(NotImplementedError):
        await ipfszarr3.get_partial_values(
            zarr.core.buffer.default_buffer_prototype(), []
        )

    previous_zarr_json = await ipfszarr3.get(
        "zarr.json", zarr.core.buffer.default_buffer_prototype()
    )
    assert previous_zarr_json is not None
    # Setting a metadata file that should always exist should not change anything
    await ipfszarr3.set_if_not_exists("zarr.json", np.array([b"a"], dtype=np.bytes_))  # type: ignore np.arrays, if dtype is bytes, is usable as a zarr buffer
    zarr_json_now = await ipfszarr3.get(
        "zarr.json", zarr.core.buffer.default_buffer_prototype()
    )
    assert zarr_json_now is not None
    assert previous_zarr_json.to_bytes() == zarr_json_now.to_bytes()

    # now remove that metadata file and then add it back
    ipfszarr3 = IPFSZarr3(ipfszarr3.hamt, read_only=False)  # make a writable version
    await ipfszarr3.delete("zarr.json")
    ipfszarr3_keys: set[str] = set()
    async for k in ipfszarr3.list():
        ipfszarr3_keys.add(k)
    assert hamt_keys != ipfszarr3_keys
    assert "zarr.json" not in ipfszarr3_keys

    await ipfszarr3.set_if_not_exists("zarr.json", previous_zarr_json)
    zarr_json_now = await ipfszarr3.get(
        "zarr.json", zarr.core.buffer.default_buffer_prototype()
    )
    assert zarr_json_now is not None
    assert previous_zarr_json.to_bytes() == zarr_json_now.to_bytes()


def test_encryption(random_zarr_dataset: tuple[str, xr.Dataset]):
    _, test_ds = random_zarr_dataset

    with pytest.raises(ValueError, match="Encryption key is not 32 bytes"):
        create_zarr_encryption_transformers(bytes(), bytes())

    encryption_key = bytes(32)
    # Encrypt only precipitation, not temperature or the coordinate variables
    encrypt, decrypt = create_zarr_encryption_transformers(
        encryption_key,
        header="sample-header".encode(),
        exclude_vars=["lat", "lon", "time", "temp"],
    )
    hamt = HAMT(
        store=IPFSStore(), transformer_encode=encrypt, transformer_decode=decrypt
    )
    ipfszarr3 = IPFSZarr3(hamt)
    test_ds.to_zarr(store=ipfszarr3)  # type: ignore

    ipfs_ds = xr.open_zarr(store=ipfszarr3)
    xr.testing.assert_identical(ipfs_ds, test_ds)

    # Now trying to load without a decryptor, xarray should be able to read the metadata and still perform operations on the unencrypted variable
    print("=== Attempting to read and print metadata of partially encrypted zarr")

    ds = xr.open_zarr(
        store=IPFSZarr3(
            HAMT(store=IPFSStore(), root_node_id=hamt.root_node_id, read_only=True)
        )
    )
    print(ds)
    assert ds.temp.sum() == test_ds.temp.sum()
    # We should be unable to read precipitation values which are still encrypted
    with pytest.raises(Exception):
        ds.precip.sum()


# This test assumes the other zarr ipfs tests are working fine, so if other things are breaking check those first
def test_authenticated_gateway(random_zarr_dataset: tuple[str, xr.Dataset]):
    _, test_ds = random_zarr_dataset

    def write_and_check(store: IPFSStore) -> bool:
        try:
            store.rpc_uri_stem = "http://127.0.0.1:5002"  # 5002 is the port configured in the run-checks.yaml actions file for nginx to serve the proxy on
            hamt = HAMT(store=store)
            ipfszarr3 = IPFSZarr3(hamt)
            test_ds.to_zarr(store=ipfszarr3, mode="w")  # type: ignore
            loaded_ds = xr.open_zarr(store=ipfszarr3)
            xr.testing.assert_identical(test_ds, loaded_ds)
            return True
        except Exception as _:
            return False

    # Test with API Key
    api_key_store = IPFSStore(api_key="test")
    assert write_and_check(api_key_store)

    # Test that wrong API Key fails
    bad_api_key_store = IPFSStore(api_key="badKey")
    assert not write_and_check(bad_api_key_store)

    # Test just bearer token
    bearer_ipfs_store = IPFSStore(bearer_token="test")
    assert write_and_check(bearer_ipfs_store)

    # Test with wrong bearer
    bad_bearer_store = IPFSStore(bearer_token="wrongBearer")
    assert not write_and_check(bad_bearer_store)

    # Test with just basic auth
    basic_auth_store = IPFSStore(basic_auth=("test", "test"))
    assert write_and_check(basic_auth_store)

    # Test with wrong basic auth
    bad_basic_auth_store = IPFSStore(basic_auth=("wrong", "wrong"))
    assert not write_and_check(bad_basic_auth_store)
