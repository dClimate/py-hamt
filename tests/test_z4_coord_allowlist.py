import numpy as np
import pandas as pd
import pytest
import xarray as xr
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore


@pytest.mark.asyncio
async def test_non_allowlisted_coordinate_names_round_trip() -> None:
    """Coordinate arrays must not be indexed using the primary array geometry."""
    times = pd.date_range("2024-01-01", periods=4)
    dataset = xr.Dataset(
        {
            "temp": (
                ["time", "y", "x"],
                np.arange(4 * 4 * 6, dtype=np.float64).reshape(4, 4, 6),
            )
        },
        coords={
            "time": times,
            "y": np.arange(4, dtype=np.float64),
            "x": np.arange(6, dtype=np.float64),
        },
    ).chunk({"time": 2, "y": 4, "x": 6})

    cas = CIDInMemoryCAS()
    write_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4, 6),
        chunk_shape=(2, 4, 6),
        chunks_per_shard=8,
    )
    dataset.to_zarr(store=write_store, mode="w")
    root_cid = await write_store.flush()

    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    actual = xr.open_zarr(store=read_store)

    xr.testing.assert_identical(dataset.compute(), actual.compute())
