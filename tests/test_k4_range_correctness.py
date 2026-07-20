from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TypeAlias

import httpx
import pytest

from py_hamt import InMemoryCAS, KuboCAS

BODY = bytes(range(100))
RecordedRanges: TypeAlias = dict[str, list[str | None]]


@pytest.fixture
def range_gateway() -> Iterator[tuple[str, RecordedRanges]]:
    """Run an offline gateway that can honor or ignore Range requests."""
    recorded_ranges: RecordedRanges = {}

    class RangeHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            cid = self.path.rsplit("/", 1)[-1].split("?", 1)[0]
            range_header = self.headers.get("Range")
            recorded_ranges.setdefault(cid, []).append(range_header)

            if (
                cid in ("range-not-satisfiable", "malformed-416")
                and range_header is not None
            ):
                # Compliant gateway rejects a range starting at/past EOF with
                # 416 + ``Content-Range: bytes */N``. ``malformed-416`` omits
                # the header to exercise the non-empty-slice fallback.
                self.send_response(416)
                if cid == "range-not-satisfiable":
                    self.send_header("Content-Range", f"bytes */{len(BODY)}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if cid == "honors-range" and range_header is not None:
                unit, requested_range = range_header.split("=", 1)
                start_text, end_text = requested_range.split("-", 1)
                assert unit == "bytes"
                start = int(start_text)
                end = int(end_text)
                response_body = BODY[start : end + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {start}-{end}/{len(BODY)}")
            else:
                # This is the defective-gateway behavior under test: ignore Range.
                response_body = BODY
                self.send_response(200)

            self.send_header("Content-Length", str(len(response_body)))
            self.end_headers()
            self.wfile.write(response_body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RangeHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    gateway_url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        yield gateway_url, recorded_ranges
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


def make_kubo_cas(gateway_url: str) -> KuboCAS:
    return KuboCAS(
        gateway_base_url=gateway_url,
        rpc_base_url=gateway_url,
        max_retries=0,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cid", "load_kwargs", "expected"),
    [
        ("ignores-bounded", {"offset": 5, "length": 10}, BODY[5:15]),
        ("ignores-open-ended", {"offset": 5}, BODY[5:]),
        ("ignores-suffix", {"suffix": 7}, BODY[-7:]),
    ],
    ids=["bounded", "open-ended", "suffix"],
)
async def test_kubocas_slices_full_body_when_gateway_ignores_range(
    range_gateway: tuple[str, RecordedRanges],
    cid: str,
    load_kwargs: dict[str, int],
    expected: bytes,
) -> None:
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        actual = await cas.load(cid, **load_kwargs)
    finally:
        await cas.aclose()

    assert actual == expected, (
        f"gateway returned {len(actual)} bytes; expected {len(expected)} "
        f"for {load_kwargs}"
    )


@pytest.mark.asyncio
async def test_kubocas_zero_length_returns_empty_without_request(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    gateway_url, recorded_ranges = range_gateway
    cid = "zero-length"
    cas = make_kubo_cas(gateway_url)
    try:
        actual = await cas.load(cid, offset=5, length=0)
    finally:
        await cas.aclose()

    observed_ranges = recorded_ranges.get(cid, [])
    assert (actual, observed_ranges) == (b"", []), (
        f"zero-length load returned {len(actual)} bytes and sent Range headers "
        f"{observed_ranges!r}; expected 0 bytes and no request"
    )


@pytest.mark.asyncio
async def test_inmemorycas_zero_length_returns_empty() -> None:
    cas = InMemoryCAS()
    key = await cas.save(BODY, "raw")

    assert await cas.load(key, offset=5, length=0) == b""


@pytest.mark.asyncio
async def test_inmemorycas_zero_suffix_returns_empty() -> None:
    cas = InMemoryCAS()
    key = await cas.save(BODY, "raw")

    actual = await cas.load(key, suffix=0)

    assert actual == b"", (
        f"suffix=0 returned the full {len(actual)}-byte object; expected b''"
    )


@pytest.mark.asyncio
async def test_kubocas_zero_suffix_returns_empty_without_request(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    gateway_url, recorded_ranges = range_gateway
    cid = "zero-suffix"
    cas = make_kubo_cas(gateway_url)
    try:
        actual = await cas.load(cid, suffix=0)
    finally:
        await cas.aclose()

    observed_ranges = recorded_ranges.get(cid, [])
    assert (actual, observed_ranges) == (b"", []), (
        f"suffix=0 load returned {len(actual)} bytes and sent Range headers "
        f"{observed_ranges!r}; expected 0 bytes and no request"
    )


@pytest.mark.asyncio
async def test_kubocas_proper_206_range_is_unchanged(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    gateway_url, recorded_ranges = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        actual = await cas.load("honors-range", offset=5, length=10)
    finally:
        await cas.aclose()

    assert actual == BODY[5:15]
    assert recorded_ranges["honors-range"] == ["bytes=5-14"]


@pytest.mark.asyncio
async def test_inmemorycas_normal_ranges_use_python_slice_semantics() -> None:
    cas = InMemoryCAS()
    key = await cas.save(BODY, "raw")

    assert await cas.load(key, offset=5, length=10) == BODY[5:15]
    assert await cas.load(key, offset=5) == BODY[5:]
    assert await cas.load(key, suffix=7) == BODY[-7:]


@pytest.mark.asyncio
async def test_inmemorycas_offset_past_eof_returns_empty() -> None:
    """Python-slice semantics: a read starting at/past EOF yields b""."""
    cas = InMemoryCAS()
    key = await cas.save(BODY, "raw")

    assert await cas.load(key, offset=len(BODY)) == b""
    assert await cas.load(key, offset=len(BODY) + 10) == b""
    assert await cas.load(key, offset=len(BODY) + 10, length=5) == b""


@pytest.mark.asyncio
async def test_kubocas_offset_past_eof_returns_empty(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        actual = await cas.load("range-not-satisfiable", offset=len(BODY) + 10)
    finally:
        await cas.aclose()

    assert actual == b"", (
        f"offset past EOF returned {len(actual)} bytes; expected b'' to match "
        "Python-slice / InMemoryCAS semantics"
    )


@pytest.mark.asyncio
async def test_kubocas_in_bounds_416_is_surfaced_as_error(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    # A 416 whose ``*/N`` still covers the requested offset is a genuine
    # (unexpected) error, not an empty read: it must not be swallowed.
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await cas.load("range-not-satisfiable", offset=5)
    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_kubocas_416_without_content_range_is_surfaced_as_error(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    # Without a parseable ``Content-Range`` we cannot prove the read is empty,
    # so the 416 is raised rather than treated as b"".
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await cas.load("malformed-416", offset=len(BODY) + 10)
    finally:
        await cas.aclose()
