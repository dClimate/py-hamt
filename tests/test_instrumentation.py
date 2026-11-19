import asyncio
from collections.abc import Generator

import httpx
import pytest
from multiformats import CID, multihash
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from py_hamt import KuboCAS, instrumentation


@pytest.fixture
def otel_spans(
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[tuple[TracerProvider, InMemorySpanExporter], None, None]:
    provider = TracerProvider()
    exporter = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(instrumentation, "_TRACER", provider.get_tracer("py_hamt"))
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", "diagnostic")
    monkeypatch.setenv("PY_HAMT_OTEL_SLOW_SECONDS", "5")
    try:
        yield provider, exporter
    finally:
        provider.shutdown()


@pytest.mark.parametrize("mode", ["diagnostic", "off"])
def test_quiet_modes_preserve_parent_context_without_parent_attribute_pollution(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    mode: str,
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", mode)
    provider, exporter = otel_spans
    with provider.get_tracer("caller").start_as_current_span("job") as parent:
        with instrumentation.span(
            "py_hamt.hamt.lookup", {"py_hamt.hamt.lookup.key": "key"}
        ):
            instrumentation.record_hamt_lookup(
                "key",
                depth=1,
                node_loads=1,
                node_cache_hits=0,
                found=True,
                seconds=0.01,
            )
            load = instrumentation.begin_cas_load("cid", has_range=False)
            instrumentation.end_cas_load(load, byte_count=123)
            assert trace.get_current_span() is parent
    (exported,) = exporter.get_finished_spans()
    assert exported.name == "job"
    assert not exported.attributes
    assert instrumentation._DEFERRED_ATTRIBUTES.get() is None


def test_full_mode_preserves_nested_detail(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", "full")
    provider, exporter = otel_spans
    with provider.get_tracer("caller").start_as_current_span("job"):
        with instrumentation.span("py_hamt.hamt.lookup"):
            load = instrumentation.begin_cas_load("cid", has_range=False)
            instrumentation.end_cas_load(load, byte_count=123)
    load_span, lookup_span, parent = exporter.get_finished_spans()
    assert load_span.parent == lookup_span.context
    assert lookup_span.parent == parent.context
    assert load_span.attributes["py_hamt.cid"] == "cid"
    assert load_span.attributes["py_hamt.cas.response_bytes"] == 123


@pytest.mark.parametrize("initial_mode", ["full", "diagnostic", "off"])
@pytest.mark.parametrize("completion_mode", ["full", "diagnostic", "off"])
def test_cas_completion_keeps_the_operation_start_mode(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    initial_mode: str,
    completion_mode: str,
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", initial_mode)
    provider, exporter = otel_spans
    with provider.get_tracer("caller").start_as_current_span("job"):
        load = instrumentation.begin_cas_load("cid", has_range=False)
        monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", completion_mode)
        instrumentation.end_cas_load(load, byte_count=123, retries=1, status="timeout")
    spans = exporter.get_finished_spans()
    if initial_mode == "off":
        assert [span.name for span in spans] == ["job"]
    else:
        load_span, parent = spans
        assert load_span.parent == parent.context
        assert load_span.status.status_code == trace.StatusCode.ERROR
        assert load_span.attributes["py_hamt.cas.status"] == "timeout"
        assert load_span.attributes["py_hamt.cas.response_bytes"] == 123
        assert load_span.attributes["py_hamt.cas.retries"] == 1
    assert instrumentation._DEFERRED_ATTRIBUTES.get() is None
    assert not instrumentation._FULL_SPAN_ACTIVE.get()


def test_full_child_does_not_write_into_an_outer_diagnostic_buffer(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_SLOW_SECONDS", "0")
    _, exporter = otel_spans
    with instrumentation.span("outer"):
        monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", "full")
        with instrumentation.span("inner"):
            instrumentation._set_current_span_attributes({"owner": "inner"})
        instrumentation._set_current_span_attributes({"owner": "outer"})
    inner, outer = exporter.get_finished_spans()
    assert inner.attributes["owner"] == "inner"
    assert outer.attributes["owner"] == "outer"
    assert instrumentation._DEFERRED_ATTRIBUTES.get() is None
    assert not instrumentation._FULL_SPAN_ACTIVE.get()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["full", "diagnostic", "off"])
@pytest.mark.parametrize("error_type", [RuntimeError, asyncio.CancelledError])
@pytest.mark.parametrize("retry_first", [False, True])
async def test_cas_load_preserves_actual_exceptions_and_records_metrics_once(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    mode: str,
    error_type: type[BaseException],
    retry_first: bool,
) -> None:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", mode)
    provider, exporter = otel_spans
    reader = InMemoryMetricReader()
    meter_provider = MeterProvider(metric_readers=[reader])
    meter = meter_provider.get_meter("py_hamt")
    monkeypatch.setattr(
        instrumentation, "_CAS_LOAD_COUNTER", meter.create_counter("requests")
    )
    monkeypatch.setattr(
        instrumentation, "_CAS_LOAD_RETRIES", meter.create_counter("retries")
    )
    monkeypatch.setattr(
        instrumentation, "_CAS_LOAD_DURATION", meter.create_histogram("duration")
    )
    error = error_type("CAS interrupted")
    attempts = 0

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if retry_first and attempts == 1:
            raise httpx.ReadTimeout("retry", request=request)
        raise error

    cid = CID("base32", 1, "raw", multihash.digest(b"block", "sha2-256"))
    try:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            async with KuboCAS(
                client=client,
                gateway_base_url="https://gateway.test",
                max_retries=1,
                initial_delay=0.001,
            ) as cas:
                with provider.get_tracer("caller").start_as_current_span(
                    "job"
                ) as parent:
                    with pytest.raises(error_type) as raised:
                        await cas.load(cid)
                    assert raised.value is error
                    assert trace.get_current_span() is parent
        spans = exporter.get_finished_spans()
        if mode == "off":
            assert [span.name for span in spans] == ["job"]
        else:
            load_span, job_span = spans
            assert load_span.parent == job_span.context
            assert load_span.status.status_code == trace.StatusCode.ERROR
            assert load_span.attributes["py_hamt.cas.retries"] == int(retry_first)
            assert load_span.attributes["py_hamt.cas.status"] == (
                "cancelled" if error_type is asyncio.CancelledError else "error"
            )
            exception_events = [
                event for event in load_span.events if event.name == "exception"
            ]
            assert len(exception_events) == 1
            assert (
                "CAS interrupted"
                in exception_events[0].attributes["exception.stacktrace"]
            )
        values = {
            metric.name: metric.data.data_points[0]
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        assert values["requests"].value == 1
        assert values["retries"].value == int(retry_first)
        assert values["duration"].count == 1
        assert attempts == 1 + int(retry_first)
        assert instrumentation._DEFERRED_ATTRIBUTES.get() is None
        assert not instrumentation._FULL_SPAN_ACTIVE.get()
    finally:
        meter_provider.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("status,body", [(404, b""), (200, b"corrupt"), (503, b"")])
async def test_diagnostic_spans_retain_successful_gateway_failover(
    otel_spans: tuple[TracerProvider, InMemorySpanExporter], status: int, body: bytes
) -> None:
    provider, exporter = otel_spans
    data = b"block"
    cid = CID("base32", 1, "raw", multihash.digest(data, "sha2-256"))
    attempts: list[str] = []

    async def handle(request: httpx.Request) -> httpx.Response:
        attempts.append(request.url.host)
        return (
            httpx.Response(status, content=body)
            if request.url.host == "primary.test"
            else httpx.Response(200, content=data)
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
        async with KuboCAS(
            client=client,
            gateway_base_urls=["https://primary.test", "https://secondary.test"],
            max_retries=0,
            verify_content=True,
        ) as cas:
            with provider.get_tracer("caller").start_as_current_span("job"):
                assert await cas.load(cid) == data
    load_span, parent = exporter.get_finished_spans()
    assert attempts == ["primary.test", "secondary.test"]
    assert load_span.parent == parent.context
    assert load_span.attributes["py_hamt.cas.gateway_failures"] == 1
    assert load_span.attributes["py_hamt.cas.retries"] == 0
    assert load_span.attributes["py_hamt.cas.status"] == "ok"
    assert load_span.attributes["py_hamt.cas.response_bytes"] == len(data)
    assert load_span.status.status_code == trace.StatusCode.UNSET


@pytest.mark.parametrize("mode", ["diagnostic", "off"])
def test_quiet_metric_helpers_do_not_annotate_the_caller_outside_a_library_span(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    mode: str,
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", mode)
    provider, exporter = otel_spans
    with provider.get_tracer("caller").start_as_current_span("job"):
        instrumentation.record_hamt_lookup(
            "key", depth=1, node_loads=1, node_cache_hits=0, found=True, seconds=0.01
        )
    (parent,) = exporter.get_finished_spans()
    assert not parent.attributes


@pytest.mark.parametrize(
    "error", [RuntimeError("failure"), asyncio.CancelledError("cancelled")]
)
@pytest.mark.parametrize("mode", ["full", "diagnostic"])
def test_exceptions_preserve_stack_and_job_context(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    error: BaseException,
    mode: str,
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", mode)
    provider, exporter = otel_spans
    with provider.get_tracer("caller").start_as_current_span("job"):
        with pytest.raises(type(error)) as raised:
            with instrumentation.span(
                "py_hamt.hamt.lookup", {"py_hamt.hamt.lookup.key": "key"}
            ):
                raise error
        assert raised.value is error
    operation, parent = exporter.get_finished_spans()
    assert operation.parent == parent.context
    assert operation.context.trace_id == parent.context.trace_id
    assert operation.status.status_code == trace.StatusCode.ERROR
    assert operation.attributes["py_hamt.hamt.lookup.key"] == "key"
    exception_events = [
        event for event in operation.events if event.name == "exception"
    ]
    assert len(exception_events) == 1
    assert "Traceback" in exception_events[0].attributes["exception.stacktrace"]
    assert operation.end_time >= operation.start_time
    assert instrumentation._DEFERRED_ATTRIBUTES.get() is None


@pytest.mark.parametrize("elapsed,kept", [(4.999, False), (5, True)])
def test_diagnostic_slow_threshold(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    elapsed: float,
    kept: bool,
) -> None:
    _, exporter = otel_spans
    clock = iter([0.0, elapsed])
    monkeypatch.setattr(instrumentation.time, "perf_counter", lambda: next(clock))
    with instrumentation.span("py_hamt.sharded_store.get"):
        instrumentation._set_current_span_attributes({"py_hamt.zarr.key": "chunk"})
    spans = exporter.get_finished_spans()
    assert len(spans) == int(kept)
    if kept:
        assert spans[0].attributes["py_hamt.zarr.key"] == "chunk"


@pytest.mark.parametrize("status,retries", [("timeout", 0), ("ok", 1)])
def test_diagnostic_cas_failure_and_retry_attributes(
    otel_spans: tuple[TracerProvider, InMemorySpanExporter], status: str, retries: int
) -> None:
    _, exporter = otel_spans
    load = instrumentation.begin_cas_load("cid", has_range=True)
    instrumentation.end_cas_load(load, byte_count=12, retries=retries, status=status)
    (span,) = exporter.get_finished_spans()
    assert span.attributes["py_hamt.cid"] == "cid"
    assert span.attributes["py_hamt.cas.status"] == status
    assert span.attributes["py_hamt.cas.retries"] == retries
    assert span.status.status_code == (
        trace.StatusCode.ERROR if status != "ok" else trace.StatusCode.UNSET
    )


def test_diagnostic_missing_lookup_is_retained(
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    _, exporter = otel_spans
    with instrumentation.span("py_hamt.hamt.lookup"):
        instrumentation.record_hamt_lookup(
            "missing",
            depth=1,
            node_loads=1,
            node_cache_hits=0,
            found=False,
            seconds=0.01,
        )
    (span,) = exporter.get_finished_spans()
    assert span.attributes["py_hamt.hamt.lookup.found"] is False
    assert span.attributes["py_hamt.hamt.lookup.key"] == "missing"


@pytest.mark.parametrize("mode", ["diagnostic", "off"])
def test_quiet_modes_keep_aggregate_metrics(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
    mode: str,
) -> None:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader

    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", mode)
    reader = InMemoryMetricReader()
    provider = MeterProvider(metric_readers=[reader])
    meter = provider.get_meter("py_hamt")
    monkeypatch.setattr(
        instrumentation, "_CAS_LOAD_COUNTER", meter.create_counter("requests")
    )
    monkeypatch.setattr(
        instrumentation, "_CAS_LOAD_BYTES", meter.create_counter("bytes")
    )
    monkeypatch.setattr(
        instrumentation, "_CAS_LOAD_DURATION", meter.create_histogram("duration")
    )
    try:
        load = instrumentation.begin_cas_load("cid", has_range=False)
        instrumentation.end_cas_load(load, byte_count=123)
        values = {
            metric.name: metric.data.data_points[0]
            for resource in reader.get_metrics_data().resource_metrics
            for scope in resource.scope_metrics
            for metric in scope.metrics
        }
        assert values["requests"].value == 1
        assert values["bytes"].value == 123
        assert values["duration"].count == 1
        assert not otel_spans[1].get_finished_spans()
    finally:
        provider.shutdown()


@pytest.mark.asyncio
async def test_diagnostic_context_is_isolated_between_async_tasks(
    monkeypatch: pytest.MonkeyPatch,
    otel_spans: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_SLOW_SECONDS", "0")
    provider, exporter = otel_spans

    async def read(key: str) -> None:
        with provider.get_tracer("caller").start_as_current_span(key):
            with instrumentation.span("py_hamt.hamt.lookup", {"key": key}):
                await asyncio.sleep(0)
                instrumentation._set_current_span_attributes({"result": key})

    await asyncio.gather(read("first"), read("second"))
    spans = exporter.get_finished_spans()
    parents = {
        span.context.span_id: span.name
        for span in spans
        if span.name != "py_hamt.hamt.lookup"
    }
    for span in spans:
        if span.name == "py_hamt.hamt.lookup":
            assert (
                span.attributes["key"]
                == span.attributes["result"]
                == parents[span.parent.span_id]
            )
    assert instrumentation._DEFERRED_ATTRIBUTES.get() is None


@pytest.mark.parametrize("threshold", ["-1", "nan", "inf", "bad"])
def test_invalid_diagnostic_threshold_is_rejected(
    monkeypatch: pytest.MonkeyPatch, threshold: str
) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", "diagnostic")
    monkeypatch.setenv("PY_HAMT_OTEL_SLOW_SECONDS", threshold)
    with pytest.raises(ValueError):
        with instrumentation.span("test"):
            pass


def test_invalid_mode_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", "typo")
    with pytest.raises(ValueError, match="PY_HAMT_OTEL_TRACE_MODE"):
        with instrumentation.span("test"):
            pass


def test_off_mode_skips_diagnostic_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PY_HAMT_OTEL_TRACE_MODE", "off")
    monkeypatch.setenv("PY_HAMT_OTEL_SLOW_SECONDS", "unused")
    with instrumentation.span("test") as span:
        assert not span.is_recording()


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
        self,
        name: str,
        attributes: dict[str, object] | None = None,
        *,
        record_exception: bool = True,
        set_status_on_exception: bool = True,
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
