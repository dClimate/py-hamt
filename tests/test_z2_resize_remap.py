import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import zarr.core.buffer
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore


@pytest.mark.asyncio
async def test_append_along_non_first_dimension_preserves_existing_chunks() -> None:
    """Changing a trailing chunk-grid dimension must remap existing CIDs."""
    times = pd.date_range("2024-01-01", periods=2)
    initial = xr.Dataset(
        {"temp": (["time", "lon"], np.arange(8.0).reshape(2, 4))},
        coords={"time": times, "lon": np.arange(4.0)},
    ).chunk({"time": 1, "lon": 2})
    appended = xr.Dataset(
        {"temp": (["time", "lon"], 100 + np.arange(4.0).reshape(2, 2))},
        coords={"time": times, "lon": np.arange(4.0, 6.0)},
    ).chunk({"time": 1, "lon": 2})

    cas = CIDInMemoryCAS()
    write_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 4),
        chunk_shape=(1, 2),
        chunks_per_shard=16,
    )
    initial.to_zarr(store=write_store, mode="w")
    appended.to_zarr(store=write_store, append_dim="lon")
    root_cid = await write_store.flush()

    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    actual = xr.open_zarr(store=read_store).compute()
    expected = xr.concat([initial, appended], dim="lon").compute()

    np.testing.assert_array_equal(
        actual["temp"].values,
        expected["temp"].values,
        err_msg="append along non-first dimension scrambled existing chunks",
    )


@pytest.mark.asyncio
async def test_secondary_array_metadata_does_not_resize_primary_geometry() -> None:
    """Only the primary array's zarr.json may resize the sharded geometry."""
    initial = xr.Dataset(
        {"temp": (["time", "lon"], np.arange(8.0).reshape(2, 4))},
        coords={"time": np.arange(2), "lon": np.arange(4)},
    ).chunk({"time": 1, "lon": 2})

    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 4),
        chunk_shape=(1, 2),
        chunks_per_shard=16,
    )
    initial.to_zarr(store=store, mode="w")

    prototype = zarr.core.buffer.default_buffer_prototype()
    secondary_metadata = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2, 6],
        "data_type": "float64",
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": [1, 2]},
        },
        "chunk_key_encoding": {
            "name": "default",
            "configuration": {"separator": "/"},
        },
        "fill_value": 0.0,
        "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}],
        "attributes": {"_ARRAY_DIMENSIONS": ["time", "lon"]},
        "dimension_names": ["time", "lon"],
    }
    await store.set(
        "other/zarr.json",
        prototype.buffer.from_bytes(json.dumps(secondary_metadata).encode()),
    )

    assert store._array_shape == (2, 4), (
        "secondary array metadata spuriously resized the primary geometry"
    )
    root_cid = await store.flush()
    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    actual = xr.open_zarr(store=read_store)["temp"].compute()
    xr.testing.assert_identical(initial["temp"].compute(), actual)


@pytest.mark.asyncio
async def test_shrink_non_first_dimension_preserves_surviving_chunks() -> None:
    """Shrinking a trailing dimension must retain chunks at surviving coordinates."""
    initial = xr.Dataset(
        {"temp": (["time", "lon"], np.arange(12.0).reshape(2, 6))},
        coords={"time": np.arange(2), "lon": np.arange(6)},
    ).chunk({"time": 1, "lon": 2})

    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(2, 6),
        chunk_shape=(1, 2),
        chunks_per_shard=16,
    )
    initial.to_zarr(store=store, mode="w")
    await store.resize_store((2, 4))
    await store.resize_variable("temp", (2, 4))
    await store.resize_variable("lon", (4,))
    root_cid = await store.flush()

    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    actual = xr.open_zarr(store=read_store, consolidated=False)["temp"].compute()
    expected = initial["temp"].isel(lon=slice(0, 4)).compute()

    np.testing.assert_array_equal(
        actual.values,
        expected.values,
        err_msg="shrinking a non-first dimension scrambled surviving chunks",
    )
