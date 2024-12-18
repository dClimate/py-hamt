import os
import shutil
import tempfile

from multiformats import CID
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import time

from py_hamt import HAMT, IPFSStore


@pytest.fixture
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


def test_upload_then_read(random_zarr_dataset: tuple[str, xr.Dataset]):
    zarr_path, expected_ds = random_zarr_dataset
    test_ds = xr.open_zarr(zarr_path)

    print("Writing this xarray Dataset to IPFS")
    print(test_ds)

    start_time = time.time()
    hamt1 = HAMT(store=IPFSStore(pin_on_add=False))
    test_ds.to_zarr(store=hamt1, mode="w")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Adding without pinning took {total_time:.2f} seconds")

    start_time = time.time()
    hamt2 = HAMT(store=IPFSStore(pin_on_add=True))
    test_ds.to_zarr(store=hamt2, mode="w")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Adding with pinning took {total_time:.2f} seconds")

    hamt1_root: CID = hamt1.root_node_id  # type: ignore
    hamt2_root: CID = hamt2.root_node_id  # type: ignore
    print(f"No pin on add root CID: {hamt1_root}")
    print(f"Pin on add root CID: {hamt2_root}")

    print("Reading in from IPFS")
    hamt1_read = HAMT(store=IPFSStore(), root_node_id=hamt1_root, read_only=True)
    hamt2_read = HAMT(store=IPFSStore(), root_node_id=hamt2_root, read_only=True)
    start_time = time.time()
    loaded_ds1 = xr.open_zarr(store=hamt1_read)
    loaded_ds2 = xr.open_zarr(store=hamt2_read)
    end_time = time.time()
    xr.testing.assert_identical(loaded_ds1, loaded_ds2)
    xr.testing.assert_identical(loaded_ds1, expected_ds)
    total_time = (end_time - start_time) / 2
    print(
        f"Took {total_time:.2f} seconds on average to load between the two loaded datasets"
    )

    assert "temp" in loaded_ds1
    assert "precip" in loaded_ds1
    assert loaded_ds1.temp.attrs["units"] == "celsius"

    assert loaded_ds1.temp.shape == expected_ds.temp.shape

    assert "temp" in loaded_ds2
    assert "precip" in loaded_ds2
    assert loaded_ds2.temp.attrs["units"] == "celsius"

    assert loaded_ds2.temp.shape == expected_ds.temp.shape
