import os
import shutil
import tempfile

from multiformats import CID
import numpy as np
import pandas as pd
import xarray as xr
import pytest
import time

import io

from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

from py_hamt import HamtFactory, IPFSStore


class EncryptionFilter:
    """An encryption filter for ZARR data.
    This class is serialized and stored along with the Zarr it is used with, so instead
    of storing the encryption key, we store the hash of the encryption key, so it can be
    looked up in the key registry at run time as needed.
    Parameters
    ----------
    key_hash: str
        The hex digest of an encryption key. A key may be generated using
        :func:`generate_encryption_key`. The hex digest is obtained by registering the
        key using :func:`register_encryption_key`.
    """

    codec_id = "xchacha20poly1305"
    header = b"dClimate-Zarr"

    def __init__(self, encryption_key: str):
        self.encryption_key = encryption_key

    def encode(self, buf):
        raw = io.BytesIO()
        raw.write(buf)
        nonce = get_random_bytes(24)  # XChaCha uses 24 byte (192 bit) nonce
        cipher = ChaCha20_Poly1305.new(key=self.encryption_key, nonce=nonce)
        cipher.update(self.header)
        ciphertext, tag = cipher.encrypt_and_digest(raw.getbuffer())

        return nonce + tag + ciphertext

    def decode(self, buf, out=None):
        nonce, tag, ciphertext = buf[:24], buf[24:40], buf[40:]
        cipher = ChaCha20_Poly1305.new(key=self.encryption_key, nonce=nonce)
        cipher.update(self.header)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)

        if out is not None:
            outbuf = io.BytesIO(plaintext)
            outbuf.readinto(out)
            return out

        return plaintext


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
    encryption_key = get_random_bytes(32)
    hamt1 = HamtFactory.create(
        store=IPFSStore(pin_on_add=True),
        transformer=EncryptionFilter(encryption_key=encryption_key),
    )
    test_ds.to_zarr(store=hamt1, mode="w")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Adding with encryption took {total_time:.2f} seconds")

    start_time = time.time()
    hamt2 = HamtFactory.create(
        store=IPFSStore(pin_on_add=True),
    )
    test_ds.to_zarr(store=hamt2, mode="w")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Adding without encryption took {total_time:.2f} seconds")

    hamt1_root: CID = hamt1.root_node_id  # type: ignore
    hamt2_root: CID = hamt2.root_node_id  # type: ignore
    print(f"No pin on add root CID: {hamt1_root}")
    print(f"Pin on add root CID: {hamt2_root}")

    print("Reading in from IPFS")
    hamt1_read = HamtFactory.create(
        store=IPFSStore(),
        root_node_id=hamt1_root,
        read_only=True,
        transformer=EncryptionFilter(encryption_key=encryption_key),
    )
    hamt2_read = HamtFactory.create(
        store=IPFSStore(),
        root_node_id=hamt2_root,
        read_only=True,
    )
    start_time = time.time()
    loaded_ds1 = xr.open_zarr(store=hamt1_read)
    print(loaded_ds1)
    loaded_ds2 = xr.open_zarr(store=hamt2_read)
    end_time = time.time()
    xr.testing.assert_identical(loaded_ds1, loaded_ds2)
    xr.testing.assert_identical(loaded_ds1, expected_ds)
    total_time = (end_time - start_time) / 2
    print(
        f"Took {total_time:.2f} seconds on average to load between the two loaded datasets"
    )

    # Test with bad encryption key
    hamt1_read_bad = HamtFactory.create(
        store=IPFSStore(),
        root_node_id=hamt1_root,
        read_only=True,
        transformer=EncryptionFilter(encryption_key=get_random_bytes(32)),
    )
    with pytest.raises(Exception):
        xr.open_zarr(store=hamt1_read_bad)

    assert "temp" in loaded_ds1
    assert "precip" in loaded_ds1
    assert loaded_ds1.temp.attrs["units"] == "celsius"

    assert loaded_ds1.temp.shape == expected_ds.temp.shape

    assert "temp" in loaded_ds2
    assert "precip" in loaded_ds2
    assert loaded_ds2.temp.attrs["units"] == "celsius"

    assert loaded_ds2.temp.shape == expected_ds.temp.shape
