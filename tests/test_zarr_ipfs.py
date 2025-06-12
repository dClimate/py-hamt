import time

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr
import zarr.core.buffer
from dag_cbor.ipld import IPLDKind

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
    create_ipfs: tuple[str, str],
    random_zarr_dataset: xr.Dataset,
):  # noqa for fixture which is imported above but then "redefined"
    rpc_base_url, gateway_base_url = create_ipfs
    test_ds = random_zarr_dataset
    print("=== Writing this xarray Dataset to a Zarr v3 on IPFS ===")
    print(test_ds)

    async with KuboCAS(  # <-- own and auto-close session
        rpc_base_url=rpc_base_url,
        gateway_base_url=gateway_base_url,
    ) as kubo_cas:
        hamt = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
        zhs = ZarrHAMTStore(hamt)
        assert zhs.supports_writes
        start = time.perf_counter()
        # Do an initial write along with an append which is a common xarray/zarr operation
        test_ds.to_zarr(store=zhs)  # type: ignore
        test_ds.to_zarr(store=zhs, mode="a", append_dim="time", zarr_format=3)  # type: ignore
        end = time.perf_counter()
        elapsed = end - start
        print("=== Write Stats")
        print(f"Total time in seconds: {elapsed:.2f}")
        print("=== Root CID")
        await hamt.make_read_only()
        cid: IPLDKind = hamt.root_node_id
        print(cid)

        print("=== Reading data back in and checking if identical")
        hamt = await HAMT.build(
            cas=kubo_cas, root_node_id=cid, values_are_bytes=True, read_only=True
        )
        start = time.perf_counter()
        ipfs_ds: xr.Dataset
        zhs = ZarrHAMTStore(hamt, read_only=True)
        ipfs_ds = xr.open_zarr(store=zhs)
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

        # Tests for code coverage's sake
        assert await zhs.exists("zarr.json")
        # __eq__
        assert zhs == zhs
        assert zhs != hamt
        assert not zhs.supports_writes
        assert not zhs.supports_partial_writes
        assert not zhs.supports_deletes

        hamt_keys: set[str] = set()
        async for k in zhs.hamt.keys():
            hamt_keys.add(k)

        zhs_keys_1: set[str] = set()
        async for k in zhs.list():
            zhs_keys_1.add(k)
        assert hamt_keys == zhs_keys_1

        zhs_keys_2: set[str] = set()
        async for k in zhs.list():
            zhs_keys_2.add(k)
        assert hamt_keys == zhs_keys_2

        zhs_keys_3: set[str] = set()
        async for k in zhs.list_prefix(""):
            zhs_keys_3.add(k)
        assert hamt_keys == zhs_keys_3

        with pytest.raises(NotImplementedError):
            await zhs.set_partial_values([])

        with pytest.raises(NotImplementedError):
            await zhs.get_partial_values(
                zarr.core.buffer.default_buffer_prototype(), []
            )

        previous_zarr_json: zarr.core.buffer.Buffer | None = await zhs.get(
            "zarr.json", zarr.core.buffer.default_buffer_prototype()
        )
        assert previous_zarr_json is not None
        # Setting a metadata file that should always exist should not change anything
        await zhs.set_if_not_exists(
            "zarr.json",
            np.array([b"a"], dtype=np.bytes_),  # type: ignore[arg-type]
        )  # np.arrays, if dtype is bytes, is usable as a zarr buffer
        zarr_json_now: zarr.core.buffer.Buffer | None = await zhs.get(
            "zarr.json", zarr.core.buffer.default_buffer_prototype()
        )
        assert zarr_json_now is not None
        assert previous_zarr_json.to_bytes() == zarr_json_now.to_bytes()

        # now remove that metadata file and then add it back
        await zhs.hamt.enable_write()
        zhs = ZarrHAMTStore(zhs.hamt, read_only=False)  # make a writable version
        await zhs.delete("zarr.json")
        # doing a duplicate delete should not result in an error
        await zhs.delete("zarr.json")
        zhs_keys_4: set[str] = set()
        async for k in zhs.list():
            zhs_keys_4.add(k)
        assert hamt_keys != zhs_keys_4
        assert "zarr.json" not in zhs_keys_4

        await zhs.set_if_not_exists("zarr.json", previous_zarr_json)
        zarr_json_now = await zhs.get(
            "zarr.json", zarr.core.buffer.default_buffer_prototype()
        )
        assert zarr_json_now is not None
        assert previous_zarr_json.to_bytes() == zarr_json_now.to_bytes()


@pytest.mark.asyncio
async def test_list_dir_dedup():
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    zhs = ZarrHAMTStore(hamt)

    await hamt.set("foo/bar/0", b"")
    await hamt.set("foo/bar/1", b"")
    results = [n async for n in zhs.list_dir("foo/")]
    assert results == ["bar"]  # no duplicates
