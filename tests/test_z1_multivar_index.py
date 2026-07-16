import base64
import json
from pathlib import Path

import dag_cbor
import numpy as np
import pytest
import xarray as xr
from testing_utils import CIDInMemoryCAS

from py_hamt import HAMT, ShardedZarrStore
from py_hamt.hamt_to_sharded_converter import convert_hamt_to_sharded
from py_hamt.sharded_zarr_store import SHARDED_ZARR_V2
from py_hamt.zarr_hamt_store import ZarrHAMTStore

FIXTURE_DIR = Path(__file__).parent / "fixtures"
DATA_VARIABLES = ("temp", "precip")
ARRAY_SHAPE = (4, 4, 6)


def _multivar_dataset(*, different_chunk_grids: bool) -> xr.Dataset:
    dataset = xr.Dataset(
        {
            "temp": (
                ("time", "lat", "lon"),
                np.full(ARRAY_SHAPE, 1.0, dtype=np.float64),
            ),
            "precip": (
                ("time", "lat", "lon"),
                np.full(ARRAY_SHAPE, 2.0, dtype=np.float64),
            ),
        },
        coords={
            "time": np.arange(ARRAY_SHAPE[0]),
            "lat": np.linspace(-90.0, 90.0, ARRAY_SHAPE[1]),
            "lon": np.linspace(-180.0, 180.0, ARRAY_SHAPE[2]),
        },
    )
    dataset["temp"].encoding["chunks"] = (2, 4, 6)
    dataset["precip"].encoding["chunks"] = (
        (2, 2, 6) if different_chunk_grids else (2, 4, 6)
    )
    return dataset


def _single_var_dataset() -> xr.Dataset:
    values = np.arange(np.prod(ARRAY_SHAPE), dtype=np.float64).reshape(ARRAY_SHAPE)
    dataset = xr.Dataset(
        {"temp": (("time", "lat", "lon"), values)},
        coords={
            "time": np.arange(ARRAY_SHAPE[0]),
            "lat": np.linspace(-90.0, 90.0, ARRAY_SHAPE[1]),
            "lon": np.linspace(-180.0, 180.0, ARRAY_SHAPE[2]),
        },
    )
    dataset["temp"].encoding["chunks"] = (2, 4, 6)
    return dataset


async def _decode_root(cas: CIDInMemoryCAS, root_cid: str) -> dict[str, object]:
    root = dag_cbor.decode(await cas.load(root_cid))
    assert isinstance(root, dict)
    return root


def _assert_data_chunks_are_indexed(
    root_obj: dict[str, object], array_paths: tuple[str, ...] = DATA_VARIABLES
) -> None:
    metadata = root_obj["metadata"]
    assert isinstance(metadata, dict)

    for array_path in array_paths:
        chunk_prefix = f"{array_path}/c/"
        offending_keys = sorted(
            key
            for key in metadata
            if isinstance(key, str) and key.startswith(chunk_prefix)
        )
        assert not offending_keys, (
            f"data-variable chunks for {array_path!r} must use a per-array "
            f"shard index, not root metadata entries; offending keys: {offending_keys}"
        )

    data_variable_keys = sorted(
        key
        for key in metadata
        if isinstance(key, str)
        and any(key == path or key.startswith(f"{path}/") for path in array_paths)
    )
    non_metadata_keys = [
        key for key in data_variable_keys if not key.endswith("zarr.json")
    ]
    assert not non_metadata_keys, (
        "root metadata may contain only zarr.json entries for data variables; "
        f"offending keys: {non_metadata_keys}"
    )


async def _open_sharded_dataset(
    cas: CIDInMemoryCAS, root_cid: str, *, group: str | None = None
) -> xr.Dataset:
    store = await ShardedZarrStore.open(cas=cas, read_only=True, root_cid=root_cid)
    return xr.open_zarr(store=store, group=group)


@pytest.mark.asyncio
@pytest.mark.parametrize("different_chunk_grids", [False, True])
async def test_multivar_chunks_have_per_array_indexes(
    different_chunk_grids: bool,
) -> None:
    # Path-aware indexing is the v2 contract; shape-based creation intentionally
    # remains the deprecated v1 compatibility path on the current base.
    # Distinct per-array chunk grids exercise the core multi-array corruption
    # scenario: reuse of the primary array's geometry would corrupt "precip".
    expected = _multivar_dataset(different_chunk_grids=different_chunk_grids)
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        chunks_per_shard=8,
        manifest_version=SHARDED_ZARR_V2,
    )

    expected.to_zarr(store=store, mode="w", group="0")
    root_cid = await store.flush()

    actual = await _open_sharded_dataset(cas, root_cid, group="0")
    np.testing.assert_array_equal(actual["temp"].values, expected["temp"].values)
    np.testing.assert_array_equal(actual["precip"].values, expected["precip"].values)
    _assert_data_chunks_are_indexed(
        await _decode_root(cas, root_cid), ("0/temp", "0/precip")
    )


@pytest.mark.asyncio
async def test_converter_multivar_builds_per_array_indexes() -> None:
    expected = _multivar_dataset(different_chunk_grids=True)
    cas = CIDInMemoryCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    source_store = ZarrHAMTStore(hamt, read_only=False)

    expected.to_zarr(store=source_store, mode="w")
    await hamt.make_read_only()
    hamt_root_cid = str(hamt.root_node_id)

    # The converter's type hint is narrower than what it needs (KuboCAS vs any
    # ContentAddressedStore); any CID-producing CAS works at runtime.
    sharded_root_cid = await convert_hamt_to_sharded(
        cas=cas,  # type: ignore[arg-type]
        hamt_root_cid=hamt_root_cid,
        chunks_per_shard=8,
    )

    actual = await _open_sharded_dataset(cas, sharded_root_cid)
    np.testing.assert_array_equal(actual["temp"].values, expected["temp"].values)
    np.testing.assert_array_equal(actual["precip"].values, expected["precip"].values)
    _assert_data_chunks_are_indexed(await _decode_root(cas, sharded_root_cid))


@pytest.mark.asyncio
async def test_root_manifest_has_explicit_version_field() -> None:
    expected = _single_var_dataset()
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=ARRAY_SHAPE,
        chunk_shape=(2, 4, 6),
        chunks_per_shard=8,
    )

    expected.to_zarr(store=store, mode="w")
    root_obj = await _decode_root(cas, await store.flush())

    assert "manifest_version" in root_obj
    assert isinstance(root_obj["manifest_version"], str)
    assert root_obj["manifest_version"]


async def _load_fixture(filename: str) -> tuple[CIDInMemoryCAS, str]:
    serialized = json.loads((FIXTURE_DIR / filename).read_text())
    cas = CIDInMemoryCAS()
    cas.store.update({
        cid: base64.b64decode(encoded_block)
        for cid, encoded_block in serialized["blocks"].items()
    })
    return cas, serialized["root_cid"]


@pytest.mark.asyncio
async def test_legacy_single_var_fixture_readable() -> None:
    # Generated at HEAD 790f5a4 by z1_make_legacy_fixture.py. Never regenerate this
    # fixture after the format change: it is the legacy compatibility contract.
    cas, root_cid = await _load_fixture("z1_legacy_single_var_store.json")

    actual = await _open_sharded_dataset(cas, root_cid)
    np.testing.assert_array_equal(
        actual["temp"].values, _single_var_dataset()["temp"].values
    )


@pytest.mark.asyncio
async def test_legacy_multivar_fixture_readable() -> None:
    # Generated at HEAD 790f5a4 by z1_make_legacy_fixture.py. It preserves the
    # current mixed shard-index/flat-metadata multi-variable layout for reading.
    cas, root_cid = await _load_fixture("z1_legacy_multivar_store.json")

    actual = await _open_sharded_dataset(cas, root_cid)
    expected = _multivar_dataset(different_chunk_grids=False)
    np.testing.assert_array_equal(actual["temp"].values, expected["temp"].values)
    np.testing.assert_array_equal(actual["precip"].values, expected["precip"].values)
