import dag_cbor
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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data_variable", "coordinate"),
    [("data", "lat"), ("streamflow", "time")],
)
async def test_legacy_manifest_derives_primary_array_structurally(
    data_variable: str, coordinate: str
) -> None:
    """Legacy roots must reject same-dimensional auxiliary array candidates."""
    dataset = xr.Dataset(
        {data_variable: ([coordinate], np.linspace(10.0, 19.0, 10))},
        coords={coordinate: np.arange(10)},
    ).chunk({coordinate: 5})

    cas = CIDInMemoryCAS()
    write_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=8,
    )
    # Write the data variable first to reproduce the layout produced by the
    # pre-fix name allowlist: primary chunks are sharded, coordinate chunks are
    # retained in the root metadata dictionary.
    dataset.drop_vars(coordinate).to_zarr(store=write_store, mode="w")
    dataset[[coordinate]].to_zarr(store=write_store, mode="a")
    root_cid = await write_store.flush()

    root_obj = dag_cbor.decode(await cas.load(root_cid))
    assert isinstance(root_obj, dict)
    root_obj["chunks"].pop("primary_array_path")
    legacy_root_cid = await cas.save(dag_cbor.encode(root_obj), codec="dag-cbor")

    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=str(legacy_root_cid),
    )
    assert read_store._primary_array_path == data_variable

    actual = xr.open_zarr(store=read_store)
    xr.testing.assert_identical(dataset.compute(), actual.compute())


@pytest.mark.asyncio
async def test_primary_array_metadata_guards() -> None:
    """Non-array keys and non-object metadata cannot identify a primary array."""
    store = await ShardedZarrStore.open(
        cas=CIDInMemoryCAS(),
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=8,
    )

    assert not store._record_primary_array_path("unrelated.json", {"shape": [10]})
    assert not store._record_primary_array_path("data/zarr.json", ["not", "a", "dict"])
    assert store._primary_array_path is None

    assert not store._metadata_matches_store_geometry({})
    assert not store._metadata_matches_store_geometry({
        "chunk_grid": {"name": "regular", "configuration": []}
    })


@pytest.mark.asyncio
async def test_legacy_derivation_ignores_malformed_metadata() -> None:
    """Malformed legacy entries are ignored and ndim-only fallback still works."""
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(10,),
        chunk_shape=(5,),
        chunks_per_shard=8,
    )

    store._root_obj["metadata"] = []
    await store._derive_primary_array_path()
    assert store._primary_array_path is None

    invalid_json_cid = await cas.save(b"{", codec="raw")
    non_object_cid = await cas.save(b"[]", codec="raw")
    fallback_cid = await cas.save(b'{"shape": [9]}', codec="raw")
    store._root_obj["metadata"] = {
        1: fallback_cid,
        "invalid/zarr.json": invalid_json_cid,
        "non-object/zarr.json": non_object_cid,
        "fallback/zarr.json": fallback_cid,
    }

    await store._derive_primary_array_path()

    assert store._primary_array_path == "fallback"
