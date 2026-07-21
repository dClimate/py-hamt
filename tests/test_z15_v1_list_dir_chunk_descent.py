import numpy as np
import pytest
import xarray as xr
from testing_utils import CIDInMemoryCAS

from py_hamt import ShardedZarrStore


@pytest.mark.asyncio
async def test_v1_list_dir_descends_into_chunk_directory() -> None:
    """V1 ``list_dir("<primary>/c")`` must enumerate coordinate directories.

    Previously only V2 recognised a chunk prefix, so a V1 store reported "c"
    under the primary but returned nothing when listed one level deeper.
    """
    ds = xr.Dataset(
        {"temp": (["y", "x"], np.arange(16.0).reshape(4, 4))},
        coords={"y": np.arange(4), "x": np.arange(4)},
    ).chunk({"y": 2, "x": 2})
    cas = CIDInMemoryCAS()
    store = await ShardedZarrStore.open(
        cas=cas,
        read_only=False,
        array_shape=(4, 4),
        chunk_shape=(2, 2),
        chunks_per_shard=16,
    )
    ds.to_zarr(store=store, mode="w")
    root_cid = await store.flush()

    read_store = await ShardedZarrStore.open(
        cas=cas,
        read_only=True,
        root_cid=root_cid,
    )
    primary = read_store._primary_array_path
    assert primary is not None

    # "c" appears directly under the primary...
    top = {key async for key in read_store.list_dir(primary)}
    assert "c" in top

    # ...and listing one level deeper enumerates the first chunk coordinate.
    chunk_prefix = "c" if primary == "" else f"{primary}/c"
    descended = {key async for key in read_store.list_dir(chunk_prefix)}

    expected = {
        key[len(chunk_prefix) + 1 :].split("/", 1)[0]
        async for key in read_store.list()
        if key.startswith(f"{chunk_prefix}/")
    }
    assert expected, "test setup produced no chunk keys under the primary"
    assert descended == expected
