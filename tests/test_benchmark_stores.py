"""Performance benchmarks for the Zarr store backends (issue #58).

This file replaces a commented-out scratch script. That script could not run:
it imported ``FlatZarrStore`` (no longer exported), pointed at the production
``ipfs-gateway.dclimate.net`` with a blank ``X-API-Key``, used
``Dataset.dims.values()`` (removed in modern xarray), and ``print``ed timings
without asserting anything -- so it could not fail, and therefore protected
nothing.

What is asserted here is **request counts**, not wall-clock time. Counts are
deterministic, so they hold up in CI, and they are the quantity the performance
work actually moved: PR #88's speedups and ``ShardedZarrStore`` both work by
collapsing HAMT traversal depth into fewer network round trips. Timings are
still measured and reported via ``--benchmark-report`` for humans, but nothing
fails on them.

Run:
    pytest tests/test_benchmark_stores.py --ipfs
    pytest tests/test_benchmark_stores.py --ipfs --benchmark-report -s
"""

import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from py_hamt import HAMT, KuboCAS, ShardedZarrStore, ZarrHAMTStore
from py_hamt import instrumentation as instr

pytestmark = pytest.mark.ipfs


# --------------------------------------------------------------------------
# Harness
# --------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    """One measured phase: how long it took and what it cost in requests."""

    label: str
    seconds: float
    counters: dict[str, float] = field(default_factory=dict)
    unique_cids: int = 0
    duplicate_requests: int = 0

    @property
    def cas_loads(self) -> float:
        return self.counters.get("cas_load.total", 0.0)

    @property
    def retries(self) -> float:
        return self.counters.get("cas_load.retries", 0.0)

    @property
    def shard_loads(self) -> float:
        return self.counters.get("sharded_store.shard_cache.total", 0.0)

    @property
    def work_units(self) -> float:
        """Total storage operations, whichever backend performed them.

        ``ShardedZarrStore`` writes buffer into shards and flush, so they emit
        ``sharded_store.*`` rather than ``cas_load.*``. Comparing raw
        ``cas_load`` totals across the two backends would therefore read a
        sharded write as "no work done".
        """
        return self.cas_loads + self.shard_loads

    def report(self) -> str:
        return (
            f"  {self.label:<34} {self.seconds:7.2f}s  "
            f"cas_loads={self.cas_loads:<7.0f} "
            f"unique={self.unique_cids:<7} "
            f"dupes={self.duplicate_requests:<5} "
            f"retries={self.retries:.0f}"
        )


class _Phase:
    """Context manager timing a block and snapshotting instrumentation."""

    def __init__(self, label: str) -> None:
        self.label = label
        self.result: Optional[BenchmarkResult] = None

    def __enter__(self) -> "_Phase":
        instr.reset(self.label)
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc: Any) -> None:
        elapsed = time.perf_counter() - self._start
        snap = instr.snapshot()
        self.result = BenchmarkResult(
            label=self.label,
            seconds=elapsed,
            counters=dict(snap.get("counters", {})),
            unique_cids=int(snap.get("cas_load_unique_cids", 0)),
            duplicate_requests=int(snap.get("cas_load_duplicate_requests", 0)),
        )


@pytest.fixture(autouse=True)
def _trace_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Instrumentation is opt-in via env var; benchmarks always need it."""
    monkeypatch.setenv("PY_HAMT_TRACE", "1")


@pytest.fixture
def report(request: pytest.FixtureRequest) -> Any:
    """Collect phase results and print them when --benchmark-report is passed."""
    collected: list[BenchmarkResult] = []

    def add(result: Optional[BenchmarkResult]) -> BenchmarkResult:
        assert result is not None, "phase did not complete"
        collected.append(result)
        return result

    yield add

    if request.config.getoption("--benchmark-report") and collected:
        print(f"\n{'=' * 78}\n{request.node.name}\n{'=' * 78}")
        for result in collected:
            print(result.report())


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------


def _make_dataset(
    variable: str, *, periods: int = 100, time_chunk: int = 20
) -> xr.Dataset:
    """A small, deterministically-seeded gridded dataset.

    Seeded so chunk bytes -- and therefore CIDs and request counts -- are stable
    across runs. An unseeded dataset would make the count assertions flaky.
    """
    rng = np.random.default_rng(0)
    times = pd.date_range("2024-01-01", periods=periods)
    lats = np.linspace(-90, 90, 18)
    lons = np.linspace(-180, 180, 36)

    data = rng.standard_normal((len(times), len(lats), len(lons)))
    ds = xr.Dataset(
        {variable: (["time", "lat", "lon"], data)},
        coords={"time": times, "lat": lats, "lon": lons},
    )
    return ds.chunk({"time": time_chunk, "lat": 18, "lon": 36})


@pytest.fixture(scope="module")
def hamt_dataset() -> xr.Dataset:
    return _make_dataset("temp")


@pytest.fixture(scope="module")
def shard_dataset() -> xr.Dataset:
    return _make_dataset("precip")


def _chunk_shape(ds: xr.Dataset) -> tuple[int, ...]:
    """Chunk shape in dimension order, via .sizes (``.dims`` is deprecated)."""
    return tuple(
        ds.chunks[dim][0] if dim in ds.chunks else ds.sizes[dim] for dim in ds.sizes
    )


def _appended_shape(ds: xr.Dataset) -> tuple[int, ...]:
    """Array shape after one append along ``time``."""
    return tuple(size * 2 if dim == "time" else size for dim, size in ds.sizes.items())


# --------------------------------------------------------------------------
# Benchmarks
# --------------------------------------------------------------------------


async def test_benchmark_hamt_store(
    create_ipfs: tuple[str, str], hamt_dataset: xr.Dataset, report: Any
) -> None:
    """Write-then-append-then-read through ZarrHAMTStore."""
    rpc, gateway = create_ipfs
    ds = hamt_dataset

    async with KuboCAS(rpc_base_url=rpc, gateway_base_url=gateway) as cas:
        with _Phase("ZarrHAMTStore write+append") as phase:
            hamt = await HAMT.build(cas=cas, values_are_bytes=True)
            store = ZarrHAMTStore(hamt)
            ds.to_zarr(store=store, mode="w")
            ds.to_zarr(store=store, mode="a", append_dim="time")
            await hamt.make_read_only()
        write = report(phase.result)

        root = hamt.root_node_id

        with _Phase("ZarrHAMTStore read") as phase:
            read_hamt = await HAMT.build(
                cas=cas, root_node_id=root, values_are_bytes=True, read_only=True
            )
            actual = xr.open_zarr(store=ZarrHAMTStore(read_hamt, read_only=True))
            actual.load()
        read = report(phase.result)

    xr.testing.assert_identical(xr.concat([ds, ds], dim="time"), actual)

    assert write.cas_loads > 0, "write did no content-store work"
    assert read.cas_loads > 0, "read did no content-store work"
    # A clean local daemon should never need to retry.
    assert write.retries == 0
    assert read.retries == 0


async def test_benchmark_sharded_store(
    create_ipfs: tuple[str, str], shard_dataset: xr.Dataset, report: Any
) -> None:
    """Write-then-append-then-read through ShardedZarrStore."""
    rpc, gateway = create_ipfs
    ds = shard_dataset

    async with KuboCAS(rpc_base_url=rpc, gateway_base_url=gateway) as cas:
        with _Phase("ShardedZarrStore write+append") as phase:
            store = await ShardedZarrStore.open(
                cas=cas,
                read_only=False,
                array_shape=_appended_shape(ds),
                chunk_shape=_chunk_shape(ds),
                chunks_per_shard=50,
            )
            ds.to_zarr(store=store, mode="w")
            ds.to_zarr(store=store, mode="a", append_dim="time")
            root_cid = await store.flush()
        write = report(phase.result)

        with _Phase("ShardedZarrStore read") as phase:
            read_store = await ShardedZarrStore.open(
                cas=cas, read_only=True, root_cid=root_cid
            )
            actual = xr.open_zarr(store=read_store)
            actual.load()
        read = report(phase.result)

    xr.testing.assert_identical(xr.concat([ds, ds], dim="time"), actual)

    assert write.work_units > 0, "write did no storage work"
    assert read.work_units > 0, "read did no storage work"
    assert read.retries == 0


@pytest.mark.xfail(
    reason=(
        "Sharded full-scan reads currently cost ~2x the HAMT walk and refetch "
        "roughly one duplicate block per chunk (320 chunks -> 313 duplicate "
        "requests, vs 5 for HAMT). cas_load totals are also identical for "
        "chunks_per_shard of 50, 500 and 5000, and shard_cache.hit/miss are "
        "never emitted -- so this read path is not consulting the shard cache "
        "at lines 1770-1820 at all. Tracked separately; this test documents the "
        "intended contract and will pass once the read path uses the cache."
    ),
    strict=True,
)
async def test_sharded_read_uses_fewer_round_trips_than_hamt(
    create_ipfs: tuple[str, str], report: Any
) -> None:
    """The sharded store's reason for existing, asserted rather than assumed.

    Sharding batches many chunk references into one shard object, so a full read
    should cost materially fewer content-store loads than the HAMT walk. This is
    the claim behind PR #88 and issue #56; without it in a test, a regression
    that reintroduced per-chunk round trips passes CI silently -- which is
    exactly what was happening.

    The dataset here is deliberately larger than ``chunks_per_shard``. Below that
    threshold both backends coincidentally tie, so a smaller dataset would let
    this assertion pass without demonstrating anything.
    """
    rpc, gateway = create_ipfs
    # 100 time steps at a chunk of 5 => 80 chunks, comfortably above the
    # chunks_per_shard=50 used below.
    ds = _make_dataset("temp", periods=100, time_chunk=5)

    async with KuboCAS(rpc_base_url=rpc, gateway_base_url=gateway) as cas:
        hamt = await HAMT.build(cas=cas, values_are_bytes=True)
        ds.to_zarr(store=ZarrHAMTStore(hamt), mode="w")
        await hamt.make_read_only()

        with _Phase("HAMT read") as phase:
            read_hamt = await HAMT.build(
                cas=cas,
                root_node_id=hamt.root_node_id,
                values_are_bytes=True,
                read_only=True,
            )
            hamt_result = xr.open_zarr(store=ZarrHAMTStore(read_hamt, read_only=True))
            hamt_result.load()
        hamt_read = report(phase.result)

        sharded = await ShardedZarrStore.open(
            cas=cas,
            read_only=False,
            array_shape=tuple(ds.sizes.values()),
            chunk_shape=_chunk_shape(ds),
            chunks_per_shard=50,
        )
        ds.to_zarr(store=sharded, mode="w")
        root_cid = await sharded.flush()

        with _Phase("Sharded read") as phase:
            read_store = await ShardedZarrStore.open(
                cas=cas, read_only=True, root_cid=root_cid
            )
            sharded_result = xr.open_zarr(store=read_store)
            sharded_result.load()
        sharded_read = report(phase.result)

    # Both must return the same data, or the comparison is meaningless.
    xr.testing.assert_identical(hamt_result, sharded_result)

    # Network round trips specifically: a shard fetch that hits the in-process
    # cache costs nothing remotely, so cas_load is the honest measure on both
    # sides here (unlike the write path, which buffers -- see work_units).
    assert sharded_read.cas_loads <= hamt_read.cas_loads, (
        f"sharded read cost {sharded_read.cas_loads:.0f} CAS loads vs HAMT's "
        f"{hamt_read.cas_loads:.0f}; sharding is supposed to reduce round trips"
    )


async def test_read_cache_prevents_duplicate_cid_fetches(
    create_ipfs: tuple[str, str], report: Any
) -> None:
    """Re-reading the same data must be served from cache, not refetched.

    Guards the read-cache behaviour that ``test_h5_read_cache_growth`` bounds the
    memory of: that test proves the cache does not grow without limit, this one
    proves it actually saves requests.
    """
    rpc, gateway = create_ipfs
    ds = _make_dataset("temp", periods=40)

    async with KuboCAS(rpc_base_url=rpc, gateway_base_url=gateway) as cas:
        hamt = await HAMT.build(cas=cas, values_are_bytes=True)
        ds.to_zarr(store=ZarrHAMTStore(hamt), mode="w")
        await hamt.make_read_only()

        read_hamt = await HAMT.build(
            cas=cas,
            root_node_id=hamt.root_node_id,
            values_are_bytes=True,
            read_only=True,
        )
        store = ZarrHAMTStore(read_hamt, read_only=True)

        with _Phase("first read (cold)") as phase:
            xr.open_zarr(store=store).load()
        cold = report(phase.result)

        with _Phase("second read (warm)") as phase:
            xr.open_zarr(store=store).load()
        warm = report(phase.result)

    assert cold.cas_loads > 0
    assert warm.cas_loads < cold.cas_loads, (
        f"warm read cost {warm.cas_loads:.0f} loads vs cold {cold.cas_loads:.0f}; "
        "the read cache is not being used"
    )


@pytest.mark.parametrize("concurrency", [1, 8, 32])
async def test_benchmark_concurrency_levels(
    create_ipfs: tuple[str, str], concurrency: int, report: Any
) -> None:
    """Read throughput across concurrency settings (also covers issue #59).

    Correctness must hold at every level, and the work done -- the number of
    distinct blocks fetched -- must not depend on how many requests are allowed
    in flight. Wall-clock is reported but not asserted on.
    """
    rpc, gateway = create_ipfs
    ds = _make_dataset("temp", periods=40)

    async with KuboCAS(rpc_base_url=rpc, gateway_base_url=gateway) as cas:
        hamt = await HAMT.build(cas=cas, values_are_bytes=True)
        ds.to_zarr(store=ZarrHAMTStore(hamt), mode="w")
        await hamt.make_read_only()
        root = hamt.root_node_id

    async with KuboCAS(
        rpc_base_url=rpc, gateway_base_url=gateway, concurrency=concurrency
    ) as cas:
        with _Phase(f"read @ concurrency={concurrency}") as phase:
            read_hamt = await HAMT.build(
                cas=cas, root_node_id=root, values_are_bytes=True, read_only=True
            )
            actual = xr.open_zarr(store=ZarrHAMTStore(read_hamt, read_only=True))
            actual.load()
        result = report(phase.result)

    xr.testing.assert_identical(ds, actual)
    assert result.unique_cids > 0
    assert result.retries == 0, (
        f"concurrency={concurrency} forced {result.retries:.0f} retries"
    )
