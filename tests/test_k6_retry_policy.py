import socket
import threading
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from multiformats import CID

from py_hamt import KuboCAS, store_httpx

EXPECTED_BODY = b"gateway response after transient failures"
TEST_CID = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")


@pytest.fixture
def retrying_kubo_server() -> Iterator[tuple[str, dict[str, int]]]:
    """Serve deterministic transient and permanent Kubo HTTP responses."""
    hit_counts: dict[str, int] = {}

    class RetryHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass

        def _record_hit(self, key: str) -> int:
            hit_counts[key] = hit_counts.get(key, 0) + 1
            return hit_counts[key]

        def _send_response(
            self,
            status: int,
            body: bytes = b"",
            headers: dict[str, str] | None = None,
        ) -> None:
            self.send_response(status)
            for name, value in (headers or {}).items():
                self.send_header(name, value)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

        def do_GET(self) -> None:
            cid = self.path.rsplit("/", 1)[-1].split("?", 1)[0]
            attempt = self._record_hit(cid)

            if cid == "transient-500" and attempt <= 2:
                self._send_response(500)
                return
            if cid == "rate-limited" and attempt == 1:
                self._send_response(429, headers={"Retry-After": "0"})
                return
            if cid == "missing":
                self._send_response(404)
                return
            self._send_response(200, EXPECTED_BODY)

        def do_POST(self) -> None:
            content_length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(content_length)
            attempt = self._record_hit("POST")
            if attempt <= 2:
                self._send_response(500)
                return

            response_body = ('{"Hash":"' + str(TEST_CID) + '"}').encode()
            self._send_response(
                200,
                response_body,
                headers={"Content-Type": "application/json"},
            )

    server = ThreadingHTTPServer(("127.0.0.1", 0), RetryHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"

    try:
        yield url, hit_counts
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


def make_cas(url: str, *, max_retries: int = 3) -> KuboCAS:
    return KuboCAS(
        gateway_base_url=url,
        rpc_base_url=url,
        max_retries=max_retries,
        initial_delay=0.01,
    )


def test_retry_delay_parses_http_dates_and_ignores_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # parsedate_to_datetime yields a naive datetime when the header lacks a
    # timezone; _retry_delay treats such values as UTC, so anchor the fake
    # future time to UTC (not local) to stay timezone-independent.
    future_naive_datetime = datetime.now(timezone.utc).replace(
        tzinfo=None
    ) + timedelta(seconds=30)
    monkeypatch.setattr(
        store_httpx,
        "parsedate_to_datetime",
        lambda _: future_naive_datetime,
    )
    dated_response = httpx.Response(429, headers={"Retry-After": "future-date"})
    assert store_httpx._retry_delay(10, 2, 1, dated_response) == 10

    def reject_retry_after(_: str) -> datetime:
        raise ValueError("invalid Retry-After")

    monkeypatch.setattr(store_httpx, "parsedate_to_datetime", reject_retry_after)
    monkeypatch.setattr(store_httpx.random, "random", lambda: 0.5)
    invalid_response = httpx.Response(429, headers={"Retry-After": "invalid"})
    assert store_httpx._retry_delay(10, 2, 1, invalid_response) == 10


def test_slice_requested_range_handles_zero_suffix() -> None:
    assert store_httpx._slice_requested_range(b"content", None, None, 0) == b""


def test_kubo_cas_rejects_negative_concurrency() -> None:
    with pytest.raises(ValueError, match="Semaphore initial value must be >= 0"):
        KuboCAS(concurrency=-1)


@pytest.mark.asyncio
async def test_owned_kubo_cas_reopens_its_semaphore_after_close() -> None:
    cas = KuboCAS()
    cas._closed = True

    semaphore = cas._loop_semaphore()

    assert isinstance(semaphore, store_httpx.asyncio.Semaphore)
    assert cas._closed is False
    await cas.aclose()


@pytest.mark.asyncio
async def test_closed_kubo_cas_with_supplied_client_cannot_reopen() -> None:
    client = httpx.AsyncClient()
    cas = KuboCAS(client=client)
    await cas.aclose()
    try:
        with pytest.raises(RuntimeError, match="KuboCAS is closed"):
            cas._loop_semaphore()
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_save_does_not_retry_nonretryable_status() -> None:
    request_count = 0

    async def reject_save(request: httpx.Request) -> httpx.Response:
        nonlocal request_count
        request_count += 1
        return httpx.Response(400, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(reject_save))
    cas = KuboCAS(client=client, max_retries=3)
    try:
        with pytest.raises(httpx.HTTPStatusError) as error:
            await cas.save(b"invalid upload", codec="raw")
    finally:
        await cas.aclose()
        await client.aclose()

    assert error.value.response.status_code == 400
    assert request_count == 1


@pytest.mark.asyncio
async def test_load_retries_transient_500_responses(
    retrying_kubo_server: tuple[str, dict[str, int]],
) -> None:
    url, hit_counts = retrying_kubo_server
    cas = make_cas(url)
    try:
        result = await cas.load("transient-500")
    finally:
        await cas.aclose()

    assert result == EXPECTED_BODY
    assert hit_counts["transient-500"] == 3


@pytest.mark.asyncio
async def test_load_retries_429_with_retry_after(
    retrying_kubo_server: tuple[str, dict[str, int]],
) -> None:
    url, hit_counts = retrying_kubo_server
    cas = make_cas(url)
    try:
        result = await cas.load("rate-limited")
    finally:
        await cas.aclose()

    assert result == EXPECTED_BODY
    assert hit_counts["rate-limited"] == 2


@pytest.mark.asyncio
async def test_load_preserves_connect_error_after_retries() -> None:
    with socket.socket() as ephemeral_socket:
        ephemeral_socket.bind(("127.0.0.1", 0))
        dead_port = ephemeral_socket.getsockname()[1]

    dead_url = f"http://127.0.0.1:{dead_port}"
    cas = make_cas(dead_url, max_retries=2)
    try:
        with pytest.raises(httpx.ConnectError):
            await cas.load("unreachable")
    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_load_does_not_retry_404(
    retrying_kubo_server: tuple[str, dict[str, int]],
) -> None:
    url, hit_counts = retrying_kubo_server
    cas = make_cas(url)
    try:
        with pytest.raises(httpx.HTTPStatusError) as error:
            await cas.load("missing")
    finally:
        await cas.aclose()

    assert error.value.response.status_code == 404
    assert hit_counts["missing"] == 1


@pytest.mark.asyncio
async def test_save_retries_transient_500_responses(
    retrying_kubo_server: tuple[str, dict[str, int]],
) -> None:
    url, hit_counts = retrying_kubo_server
    cas = make_cas(url)
    try:
        result = await cas.save(b"content-addressed upload", codec="raw")
    finally:
        await cas.aclose()

    assert isinstance(result, CID)
    assert hit_counts["POST"] == 3
