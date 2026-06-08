import os
import statistics
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from opentelemetry import metrics, trace
from opentelemetry.trace import Span, Status, StatusCode

TRUE_VALUES = {"True", "true", "1", "yes", "Y", "T"}
OTEL_ATTRIBUTE_VALUE = str | bool | int | float

_TRACER = trace.get_tracer("py_hamt")
_METER = metrics.get_meter("py_hamt")
_CAS_LOAD_COUNTER = _METER.create_counter(
    "py_hamt.cas.load.requests",
    unit="1",
    description="Number of content-addressed store load requests.",
)
_CAS_LOAD_BYTES = _METER.create_counter(
    "py_hamt.cas.load.response_bytes",
    unit="By",
    description="Bytes returned by content-addressed store load requests.",
)
_CAS_LOAD_RETRIES = _METER.create_counter(
    "py_hamt.cas.load.retries",
    unit="1",
    description="Retry attempts used by content-addressed store load requests.",
)
_CAS_LOAD_DURATION = _METER.create_histogram(
    "py_hamt.cas.load.duration",
    unit="s",
    description="Content-addressed store load latency.",
)
_HAMT_LOOKUP_COUNTER = _METER.create_counter(
    "py_hamt.hamt.lookup.requests",
    unit="1",
    description="HAMT key lookup requests.",
)
_HAMT_LOOKUP_DURATION = _METER.create_histogram(
    "py_hamt.hamt.lookup.duration",
    unit="s",
    description="HAMT key lookup latency.",
)
_HAMT_LOOKUP_DEPTH = _METER.create_histogram(
    "py_hamt.hamt.lookup.depth",
    unit="1",
    description="HAMT traversal depth per key lookup.",
)
_HAMT_NODE_LOAD_COUNTER = _METER.create_counter(
    "py_hamt.hamt.node_load.requests",
    unit="1",
    description="HAMT node load requests.",
)
_HAMT_NODE_LOAD_BYTES = _METER.create_counter(
    "py_hamt.hamt.node_load.bytes",
    unit="By",
    description="Bytes returned by HAMT node loads.",
)
_HAMT_NODE_LOAD_DURATION = _METER.create_histogram(
    "py_hamt.hamt.node_load.duration",
    unit="s",
    description="HAMT node load latency.",
)
_ZARR_GET_COUNTER = _METER.create_counter(
    "py_hamt.zarr.get.requests",
    unit="1",
    description="Zarr store get requests.",
)
_ZARR_GET_DURATION = _METER.create_histogram(
    "py_hamt.zarr.get.duration",
    unit="s",
    description="Zarr store get latency.",
)
_SHARD_LOAD_COUNTER = _METER.create_counter(
    "py_hamt.sharded_store.shard_load.requests",
    unit="1",
    description="Sharded Zarr shard load requests.",
)
_SHARD_LOAD_DURATION = _METER.create_histogram(
    "py_hamt.sharded_store.shard_load.duration",
    unit="s",
    description="Sharded Zarr shard load latency.",
)
_SHARD_ENTRIES = _METER.create_histogram(
    "py_hamt.sharded_store.shard_entries",
    unit="1",
    description="Number of entries loaded per shard.",
)


@dataclass
class TraceContext:
    started_at: float
    span: Span
    span_context_manager: Any
    metric_attributes: dict[str, OTEL_ATTRIBUTE_VALUE]
    custom_metrics_enabled: bool


def _attributes(
    attributes: dict[str, Any] | None = None,
) -> dict[str, OTEL_ATTRIBUTE_VALUE]:
    if not attributes:
        return {}
    cleaned: dict[str, OTEL_ATTRIBUTE_VALUE] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (str, bool, int, float)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


@contextmanager
def span(
    name: str, attributes: dict[str, Any] | None = None
) -> Generator[Span, None, None]:
    with _TRACER.start_as_current_span(
        name, attributes=_attributes(attributes)
    ) as active_span:
        try:
            yield active_span
        except Exception as exc:
            if active_span.is_recording():
                active_span.record_exception(exc)
                active_span.set_status(Status(StatusCode.ERROR, str(exc)))
            raise


def _current_span() -> Span:
    return trace.get_current_span()


def _set_current_span_attributes(attributes: dict[str, Any]) -> None:
    active_span = _current_span()
    if not active_span.is_recording():
        return
    for key, value in _attributes(attributes).items():
        active_span.set_attribute(key, value)


def enabled() -> bool:
    return (
        os.getenv("IPFS_RETRIEVAL_TRACE", "False") in TRUE_VALUES
        or os.getenv("PY_HAMT_TRACE", "False") in TRUE_VALUES
    )


def _sample_limit() -> int:
    try:
        return max(0, int(os.getenv("IPFS_RETRIEVAL_TRACE_SAMPLE_LIMIT", "2000")))
    except ValueError:
        return 2000


class _Metrics:
    def __init__(self, label: str | None = None) -> None:
        self.label = label
        self.started_at = time.perf_counter()
        self.counters: defaultdict[str, float] = defaultdict(float)
        self.bytes: defaultdict[str, float] = defaultdict(float)
        self.timing_totals: defaultdict[str, float] = defaultdict(float)
        self.timing_counts: defaultdict[str, int] = defaultdict(int)
        self.samples: dict[str, list[float]] = defaultdict(list)
        self.cid_counts: Counter[str] = Counter()
        self.key_counts: Counter[str] = Counter()
        self.inflight_cas_loads = 0
        self.max_inflight_cas_loads = 0
        self.sample_limit = _sample_limit()

    def observe(self, name: str, value: float) -> None:
        self.timing_totals[name] += value
        self.timing_counts[name] += 1
        if len(self.samples[name]) < self.sample_limit:
            self.samples[name].append(value)

    def value(self, name: str, value: float) -> None:
        self.counters[f"{name}.total"] += value
        self.counters[f"{name}.count"] += 1
        if len(self.samples[name]) < self.sample_limit:
            self.samples[name].append(value)


_LOCK = threading.RLock()
_METRICS = _Metrics()


def reset(label: str | None = None) -> None:
    if not enabled():
        return
    global _METRICS
    with _LOCK:
        _METRICS = _Metrics(label=label)


def increment(name: str, amount: int = 1) -> None:
    if not enabled():
        return
    with _LOCK:
        _METRICS.counters[name] += amount


def add_bytes(name: str, amount: int) -> None:
    if not enabled():
        return
    with _LOCK:
        _METRICS.bytes[name] += amount


def observe(name: str, seconds: float) -> None:
    if not enabled():
        return
    with _LOCK:
        _METRICS.observe(name, seconds)


def value(name: str, measured_value: float) -> None:
    if not enabled():
        return
    with _LOCK:
        _METRICS.value(name, measured_value)


def begin_cas_load(cid: Any, has_range: bool) -> TraceContext:
    started_at = time.perf_counter()
    metric_attributes: dict[str, OTEL_ATTRIBUTE_VALUE] = {
        "py_hamt.cas.has_range": has_range
    }
    custom_metrics_enabled = enabled()
    span_context_manager = _TRACER.start_as_current_span(
        "py_hamt.cas.load",
        attributes=_attributes({
            "py_hamt.cid": str(cid),
            **metric_attributes,
        }),
    )
    active_span = span_context_manager.__enter__()
    if custom_metrics_enabled:
        with _LOCK:
            _METRICS.counters["cas_load.range" if has_range else "cas_load.full"] += 1
            _METRICS.counters["cas_load.total"] += 1
            _METRICS.cid_counts[str(cid)] += 1
            _METRICS.inflight_cas_loads += 1
            _METRICS.max_inflight_cas_loads = max(
                _METRICS.max_inflight_cas_loads, _METRICS.inflight_cas_loads
            )
    return TraceContext(
        started_at=started_at,
        span=active_span,
        span_context_manager=span_context_manager,
        metric_attributes=metric_attributes,
        custom_metrics_enabled=custom_metrics_enabled,
    )


def end_cas_load(
    trace_context: TraceContext | None,
    *,
    byte_count: int = 0,
    retries: int = 0,
    status: str = "ok",
) -> None:
    if trace_context is None:
        return
    elapsed = time.perf_counter() - trace_context.started_at
    attributes = {**trace_context.metric_attributes, "py_hamt.cas.status": status}
    _CAS_LOAD_COUNTER.add(1, attributes)
    _CAS_LOAD_BYTES.add(byte_count, attributes)
    _CAS_LOAD_RETRIES.add(retries, attributes)
    _CAS_LOAD_DURATION.record(elapsed, attributes)
    if trace_context.span.is_recording():
        trace_context.span.set_attribute("py_hamt.cas.status", status)
        trace_context.span.set_attribute("py_hamt.cas.response_bytes", byte_count)
        trace_context.span.set_attribute("py_hamt.cas.retries", retries)
        if status != "ok":
            trace_context.span.set_status(Status(StatusCode.ERROR, status))
    if trace_context.custom_metrics_enabled:
        with _LOCK:
            _METRICS.inflight_cas_loads = max(0, _METRICS.inflight_cas_loads - 1)
            _METRICS.counters[f"cas_load.status.{status}"] += 1
            _METRICS.counters["cas_load.retries"] += retries
            _METRICS.bytes["cas_load.response_bytes"] += byte_count
            _METRICS.observe("cas_load.latency_seconds", elapsed)
    trace_context.span_context_manager.__exit__(None, None, None)


def record_hamt_node_load(
    *, cache_hit: bool, seconds: float, byte_count: int = 0
) -> None:
    attributes = {"py_hamt.cache_hit": cache_hit}
    _HAMT_NODE_LOAD_COUNTER.add(1, attributes)
    _HAMT_NODE_LOAD_BYTES.add(byte_count, attributes)
    _HAMT_NODE_LOAD_DURATION.record(seconds, attributes)
    _set_current_span_attributes({
        "py_hamt.hamt.node_load.cache_hit": cache_hit,
        "py_hamt.hamt.node_load.bytes": byte_count,
    })
    if not enabled():
        return
    with _LOCK:
        _METRICS.counters[
            "hamt_node_load.cache_hit" if cache_hit else "hamt_node_load.cache_miss"
        ] += 1
        _METRICS.counters["hamt_node_load.total"] += 1
        _METRICS.bytes["hamt_node_load.bytes"] += byte_count
        _METRICS.observe("hamt_node_load.latency_seconds", seconds)


def record_hamt_lookup(
    key: str,
    *,
    depth: int,
    node_loads: int,
    node_cache_hits: int,
    found: bool,
    seconds: float,
) -> None:
    attributes = {"py_hamt.hamt.lookup.found": found}
    _HAMT_LOOKUP_COUNTER.add(1, attributes)
    _HAMT_LOOKUP_DURATION.record(seconds, attributes)
    _HAMT_LOOKUP_DEPTH.record(depth, attributes)
    _set_current_span_attributes({
        "py_hamt.hamt.lookup.key": key,
        "py_hamt.hamt.lookup.depth": depth,
        "py_hamt.hamt.lookup.node_loads": node_loads,
        "py_hamt.hamt.lookup.node_cache_hits": node_cache_hits,
        "py_hamt.hamt.lookup.found": found,
    })
    if not enabled():
        return
    with _LOCK:
        _METRICS.counters["hamt_lookup.total"] += 1
        _METRICS.counters["hamt_lookup.found" if found else "hamt_lookup.missing"] += 1
        _METRICS.observe("hamt_lookup.latency_seconds", seconds)
        _METRICS.value("hamt_lookup.depth", depth)
        _METRICS.value("hamt_lookup.node_loads", node_loads)
        _METRICS.value("hamt_lookup.node_cache_hits", node_cache_hits)
        _METRICS.key_counts[f"hamt:{key}"] += 1


def record_zarr_get(
    *,
    store: str,
    key: str,
    kind: str,
    hit: bool,
    seconds: float,
    byte_range: bool = False,
    shard_idx: int | None = None,
) -> None:
    attributes = {
        "py_hamt.zarr.store": store,
        "py_hamt.zarr.kind": kind,
        "py_hamt.zarr.hit": hit,
        "py_hamt.zarr.byte_range": byte_range,
    }
    _ZARR_GET_COUNTER.add(1, attributes)
    _ZARR_GET_DURATION.record(seconds, attributes)
    _set_current_span_attributes({
        "py_hamt.zarr.store": store,
        "py_hamt.zarr.key": key,
        "py_hamt.zarr.kind": kind,
        "py_hamt.zarr.hit": hit,
        "py_hamt.zarr.byte_range": byte_range,
        "py_hamt.zarr.shard_idx": shard_idx,
    })
    if not enabled():
        return
    prefix = f"{store}.get.{kind}"
    with _LOCK:
        _METRICS.counters[f"{prefix}.total"] += 1
        _METRICS.counters[f"{prefix}.hit" if hit else f"{prefix}.miss"] += 1
        if byte_range:
            _METRICS.counters[f"{prefix}.byte_range"] += 1
        if shard_idx is not None:
            _METRICS.counters[f"{store}.shard.{shard_idx}.requests"] += 1
        _METRICS.observe(f"{prefix}.latency_seconds", seconds)
        _METRICS.key_counts[f"{store}:{key}"] += 1


def record_shard_load(
    *, shard_idx: int, cache_hit: bool, seconds: float, entries: int = 0
) -> None:
    attributes = {"py_hamt.cache_hit": cache_hit}
    _SHARD_LOAD_COUNTER.add(1, attributes)
    _SHARD_LOAD_DURATION.record(seconds, attributes)
    if entries:
        _SHARD_ENTRIES.record(entries, attributes)
    _set_current_span_attributes({
        "py_hamt.sharded_store.shard_idx": shard_idx,
        "py_hamt.sharded_store.shard_cache_hit": cache_hit,
        "py_hamt.sharded_store.shard_entries": entries,
    })
    if not enabled():
        return
    with _LOCK:
        _METRICS.counters[
            "sharded_store.shard_cache.hit"
            if cache_hit
            else "sharded_store.shard_cache.miss"
        ] += 1
        _METRICS.counters["sharded_store.shard_cache.total"] += 1
        _METRICS.observe("sharded_store.shard_cache.latency_seconds", seconds)
        if entries:
            _METRICS.value("sharded_store.shard_entries", entries)
        _METRICS.counters[f"sharded_store.shard.{shard_idx}.loads"] += 1


def _distribution(
    values: list[float], total: float, count: int
) -> dict[str, float | int]:
    if count == 0:
        return {"count": 0}
    sorted_values = sorted(values)

    def percentile(pct: float) -> float:
        index = min(
            len(sorted_values) - 1, max(0, round((len(sorted_values) - 1) * pct))
        )
        return sorted_values[index]

    result: dict[str, float | int] = {
        "count": count,
        "avg": total / count,
        "sample_count": len(sorted_values),
    }
    if sorted_values:
        result.update({
            "min": sorted_values[0],
            "p50": statistics.median(sorted_values),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": sorted_values[-1],
        })
    return result


def _counter_distribution(metrics: _Metrics, name: str) -> dict[str, float | int]:
    count = int(metrics.counters.get(f"{name}.count", 0))
    total = float(metrics.counters.get(f"{name}.total", 0.0))
    return _distribution(metrics.samples.get(name, []), total, count)


def snapshot() -> dict[str, Any]:
    if not enabled():
        return {}
    with _LOCK:
        metrics = _METRICS
        duplicate_cid_requests = sum(
            count - 1 for count in metrics.cid_counts.values() if count > 1
        )
        timing_summary = {
            name: _distribution(
                metrics.samples.get(name, []),
                float(total),
                int(metrics.timing_counts[name]),
            )
            for name, total in metrics.timing_totals.items()
        }
        value_summary = {
            name: _counter_distribution(metrics, name)
            for name in {
                key.rsplit(".", 1)[0]
                for key in metrics.counters
                if key.endswith(".count")
                and f"{key.rsplit('.', 1)[0]}.total" in metrics.counters
            }
        }
        return {
            "enabled": True,
            "label": metrics.label,
            "elapsed_seconds": time.perf_counter() - metrics.started_at,
            "counters": dict(metrics.counters),
            "bytes": dict(metrics.bytes),
            "timings": timing_summary,
            "values": value_summary,
            "cas_load_unique_cids": len(metrics.cid_counts),
            "cas_load_duplicate_requests": duplicate_cid_requests,
            "cas_load_top_cids": metrics.cid_counts.most_common(20),
            "zarr_top_keys": metrics.key_counts.most_common(50),
            "max_inflight_cas_loads": metrics.max_inflight_cas_loads,
            "sample_limit": metrics.sample_limit,
        }
