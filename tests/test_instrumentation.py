import pytest

from py_hamt import instrumentation


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, object] = {}
        self.exceptions: list[Exception] = []
        self.statuses: list[object] = []

    def is_recording(self) -> bool:
        return True

    def record_exception(self, exc: Exception) -> None:
        self.exceptions.append(exc)

    def set_status(self, status: object) -> None:
        self.statuses.append(status)

    def set_attribute(self, key: str, value: object) -> None:
        self.attributes[key] = value


class _FakeSpanContextManager:
    def __init__(self, span: _FakeSpan) -> None:
        self.span = span
        self.exits: list[tuple[object, object, object]] = []

    def __enter__(self) -> _FakeSpan:
        return self.span

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.exits.append((exc_type, exc, traceback))


class _FakeTracer:
    def __init__(self) -> None:
        self.spans: list[_FakeSpan] = []
        self.context_managers: list[_FakeSpanContextManager] = []

    def start_as_current_span(
        self, name: str, attributes: dict[str, object] | None = None
    ) -> _FakeSpanContextManager:
        span = _FakeSpan()
        if attributes:
            span.attributes.update(attributes)
        self.spans.append(span)
        context_manager = _FakeSpanContextManager(span)
        self.context_managers.append(context_manager)
        return context_manager


def _disable_tracing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IPFS_RETRIEVAL_TRACE", raising=False)
    monkeypatch.delenv("PY_HAMT_TRACE", raising=False)
    monkeypatch.delenv("IPFS_RETRIEVAL_TRACE_SAMPLE_LIMIT", raising=False)


def test_instrumentation_noop_when_trace_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_tracing(monkeypatch)

    instrumentation.reset("ignored")
    instrumentation.increment("counter")
    instrumentation.add_bytes("bytes", 3)
    instrumentation.observe("timing", 0.1)
    instrumentation.value("value", 2.0)
    instrumentation.record_hamt_node_load(cache_hit=False, seconds=0.01, byte_count=1)
    instrumentation.record_hamt_lookup(
        "disabled",
        depth=0,
        node_loads=0,
        node_cache_hits=0,
        found=False,
        seconds=0.01,
    )
    instrumentation.record_zarr_get(
        store="disabled",
        key="missing",
        kind="metadata",
        hit=False,
        seconds=0.01,
    )
    instrumentation.record_shard_load(
        shard_idx=0, cache_hit=False, seconds=0.01, entries=0
    )

    assert instrumentation.snapshot() == {}


def test_attributes_and_recording_span(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _FakeTracer()
    monkeypatch.setattr(instrumentation, "_TRACER", tracer)

    assert instrumentation._attributes(None) == {}
    assert instrumentation._attributes({"a": None, "b": object(), "c": 1})["c"] == 1

    with pytest.raises(RuntimeError, match="boom"):
        with instrumentation.span("test.span", {"object": object()}):
            raise RuntimeError("boom")

    assert tracer.spans[0].exceptions
    assert tracer.spans[0].statuses

    active_span = _FakeSpan()
    monkeypatch.setattr(instrumentation.trace, "get_current_span", lambda: active_span)
    instrumentation._set_current_span_attributes({"value": object(), "skip": None})
    assert "value" in active_span.attributes
    assert "skip" not in active_span.attributes


def test_snapshot_records_env_gated_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    tracer = _FakeTracer()
    monkeypatch.setattr(instrumentation, "_TRACER", tracer)

    monkeypatch.setenv("PY_HAMT_TRACE", "true")
    monkeypatch.setenv("IPFS_RETRIEVAL_TRACE_SAMPLE_LIMIT", "not-an-int")
    instrumentation.reset("invalid-limit")
    assert instrumentation.snapshot()["sample_limit"] == 2000

    monkeypatch.setenv("IPFS_RETRIEVAL_TRACE_SAMPLE_LIMIT", "1")
    instrumentation.reset("instrumented")

    instrumentation.increment("custom.counter", 2)
    instrumentation.add_bytes("custom.bytes", 5)
    instrumentation.observe("custom.timing", 0.1)
    instrumentation.observe("custom.timing", 0.2)
    instrumentation.value("custom.value", 3.0)
    instrumentation.value("custom.value", 4.0)

    instrumentation.end_cas_load(None)

    trace_context = instrumentation.begin_cas_load("cid-a", has_range=True)
    instrumentation.end_cas_load(
        trace_context, byte_count=10, retries=1, status="timeout"
    )
    trace_context = instrumentation.begin_cas_load("cid-a", has_range=False)
    instrumentation.end_cas_load(trace_context, byte_count=20, status="ok")

    instrumentation.record_hamt_node_load(cache_hit=True, seconds=0.01, byte_count=1)
    instrumentation.record_hamt_node_load(cache_hit=False, seconds=0.02, byte_count=2)
    instrumentation.record_hamt_lookup(
        "found",
        depth=2,
        node_loads=3,
        node_cache_hits=1,
        found=True,
        seconds=0.03,
    )
    instrumentation.record_hamt_lookup(
        "missing",
        depth=1,
        node_loads=1,
        node_cache_hits=0,
        found=False,
        seconds=0.04,
    )
    instrumentation.record_zarr_get(
        store="store",
        key="metadata/zarr.json",
        kind="metadata",
        hit=True,
        seconds=0.05,
        byte_range=True,
        shard_idx=7,
    )
    instrumentation.record_zarr_get(
        store="store",
        key="missing",
        kind="chunk",
        hit=False,
        seconds=0.06,
    )
    instrumentation.record_shard_load(
        shard_idx=3, cache_hit=True, seconds=0.07, entries=4
    )
    instrumentation.record_shard_load(
        shard_idx=4, cache_hit=False, seconds=0.08, entries=0
    )

    snapshot = instrumentation.snapshot()

    assert snapshot["enabled"] is True
    assert snapshot["label"] == "instrumented"
    assert snapshot["sample_limit"] == 1
    assert snapshot["counters"]["cas_load.total"] == 2
    assert snapshot["counters"]["cas_load.status.timeout"] == 1
    assert snapshot["bytes"]["cas_load.response_bytes"] == 30
    assert snapshot["timings"]["custom.timing"]["sample_count"] == 1
    assert snapshot["values"]["custom.value"]["count"] == 2
    assert snapshot["cas_load_unique_cids"] == 1
    assert snapshot["cas_load_duplicate_requests"] == 1
    assert snapshot["cas_load_top_cids"][0] == ("cid-a", 2)
    assert snapshot["zarr_top_keys"]
    assert snapshot["max_inflight_cas_loads"] == 1
    assert tracer.spans[0].statuses


def test_distribution_without_samples() -> None:
    assert instrumentation._distribution([], 0.0, 0) == {"count": 0}
    assert instrumentation._distribution([], 4.0, 2) == {
        "count": 2,
        "avg": 2.0,
        "sample_count": 0,
    }
