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

            if cid == "empty-suffix-416" and range_header is not None:
                # Zero-length object: a suffix range is unsatisfiable, so a
                # compliant gateway answers 416 + ``Content-Range: bytes */0``.
                self.send_response(416)
                self.send_header("Content-Range", "bytes */0")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            if cid == "unexpected-2xx" and range_header is not None:
                # A non-206/200 success (203) carrying the full body: the byte
                # window is unknowable, so KuboCAS must reject it.
                self.send_response(203)
                self.send_header("Content-Length", str(len(BODY)))
                self.end_headers()
                self.wfile.write(BODY)
                return

            if cid.startswith("bad-206") and range_header is not None:
                # A 206 whose Content-Range is absent, garbled, or inconsistent
                # with the body/request. httpx's raise_for_status accepts these,
                # so KuboCAS must validate them itself.
                self.send_response(206)
                if cid == "bad-206-no-range":
                    body = BODY[5:15]  # no Content-Range header at all
                elif cid == "bad-206-star-total":
                    body = BODY[5:15]
                    self.send_header("Content-Range", "bytes 5-14/*")
                elif cid == "bad-206-bad-length":
                    body = BODY[5:10]  # 5 bytes...
                    self.send_header("Content-Range", "bytes 5-14/100")  # claims 10
                else:  # bad-206-wrong-start
                    body = BODY[0:10]  # consistent window, but wrong start
                    self.send_header("Content-Range", "bytes 0-9/100")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if cid in ("honors-range", "honors-suffix") and range_header is not None:
                unit, requested_range = range_header.split("=", 1)
                start_text, end_text = requested_range.split("-", 1)
                assert unit == "bytes"
                if cid == "honors-suffix":
                    # bytes=-N maps to the last N bytes of the object.
                    start = len(BODY) - int(end_text)
                    end = len(BODY) - 1
                else:
                    start = int(start_text)
                    # bytes=start- (open-ended) runs to the end of the object.
                    end = int(end_text) if end_text else len(BODY) - 1
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


@pytest.mark.asyncio
async def test_kubocas_suffix_on_empty_object_returns_empty(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    # A suffix range against a zero-length object is unsatisfiable (416 bytes
    # */0). Python slicing of b"" yields b"", so KuboCAS must too rather than
    # raising -- mirroring InMemoryCAS and the offset-past-EOF case.
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        actual = await cas.load("empty-suffix-416", suffix=7)
    finally:
        await cas.aclose()

    assert actual == b"", (
        f"suffix read of an empty object returned {len(actual)} bytes; "
        "expected b'' to match Python-slice / InMemoryCAS semantics"
    )


@pytest.mark.asyncio
async def test_inmemorycas_suffix_on_empty_object_returns_empty() -> None:
    cas = InMemoryCAS()
    key = await cas.save(b"", "raw")

    assert await cas.load(key, suffix=7) == b""


@pytest.mark.asyncio
async def test_kubocas_honored_open_ended_and_suffix_206(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    # Valid 206 responses for an open-ended offset read and a suffix read must
    # pass validation and return the exact requested window.
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        open_ended = await cas.load("honors-range", offset=5)
        suffix = await cas.load("honors-suffix", suffix=7)
    finally:
        await cas.aclose()

    assert open_ended == BODY[5:]
    assert suffix == BODY[-7:]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cid",
    ["bad-206-no-range", "bad-206-star-total", "bad-206-bad-length"],
)
async def test_kubocas_rejects_inconsistent_206(
    range_gateway: tuple[str, RecordedRanges], cid: str
) -> None:
    # A 206 with a missing/unparseable Content-Range, or one whose declared
    # window disagrees with the body length, is untrustworthy and must raise
    # rather than silently corrupt the read.
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await cas.load(cid, offset=5, length=10)
    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_kubocas_rejects_206_with_wrong_window(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    # An internally-consistent 206 whose window does not start where we asked
    # would hand back the wrong bytes; it must be rejected.
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await cas.load("bad-206-wrong-start", offset=5, length=10)
    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_kubocas_rejects_unexpected_success_status_for_range(
    range_gateway: tuple[str, RecordedRanges],
) -> None:
    # A non-200/206 success (here 203) to a Range request carries an unknown
    # byte window and must be rejected rather than returned as-is.
    gateway_url, _ = range_gateway
    cas = make_kubo_cas(gateway_url)
    try:
        with pytest.raises(httpx.HTTPStatusError):
            await cas.load("unexpected-2xx", offset=5, length=10)
    finally:
        await cas.aclose()
