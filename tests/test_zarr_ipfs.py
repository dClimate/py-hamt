import os
import shutil
import tempfile

from multiformats import CID
import numpy as np
import pandas as pd
import xarray as xr
import pytest

from py_hamt import HAMT, IPFSStore
from py_hamt.ensemble_hamt import EnsembleHAMT


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

    print("Working with this xarray Dataset")
    print(ds)

    yield zarr_path, ds

    # Cleanup
    shutil.rmtree(temp_dir)


def test_upload_then_read(random_zarr_dataset: tuple[str, xr.Dataset]):
    zarr_path, expected_ds = random_zarr_dataset
    test_ds = expected_ds

    hamt = HAMT(store=IPFSStore(pin_on_add=False))
    test_ds.to_zarr(store=hamt, mode="w")

    hamt_root: CID = hamt.root_node_id  # type: ignore
    # Useful to print out sometimes for exploring in IPLD Explorer
    # https://explore.ipld.io/
    print(f"root CID: {hamt_root}")

    print("Reading in from IPFS with a new HAMT")
    hamt_read = HAMT(store=IPFSStore(), root_node_id=hamt_root, read_only=True)
    loaded_ds1 = xr.open_zarr(store=hamt_read)
    xr.testing.assert_identical(loaded_ds1, expected_ds)

    assert "temp" in loaded_ds1
    assert "precip" in loaded_ds1
    assert loaded_ds1.temp.attrs["units"] == "celsius"

    assert loaded_ds1.temp.shape == expected_ds.temp.shape


def test_ensemble(random_zarr_dataset: tuple[str, xr.Dataset]):
    zarr_path, expected_ds = random_zarr_dataset
    test_ds = expected_ds

    ensemble_size = 8
    ensemble_hamts = []
    for i in range(0, ensemble_size):
        ensemble_hamts.append(HAMT(store=IPFSStore()))
    ensemble = EnsembleHAMT(ensemble=ensemble_hamts)
    test_ds.to_zarr(store=ensemble)
    consolidated_hamt = HAMT(store=IPFSStore())
    ensemble.consolidate(consolidated_hamt)

    print("Reading in from IPFS to verify correctness...")
    loaded_ds = xr.open_zarr(store=consolidated_hamt)
    xr.testing.assert_identical(expected_ds, loaded_ds)
