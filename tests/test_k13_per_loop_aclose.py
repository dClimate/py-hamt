import asyncio
import logging
import queue
import threading
import warnings
from collections.abc import Iterator

import httpx
import pytest

from py_hamt import KuboCAS


@pytest.fixture(autouse=True)
def preserve_current_event_loop() -> Iterator[None]:
    """Restore the current loop after tests that use ``asyncio.run``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        try:
            previous_loop = asyncio.get_event_loop()
        except RuntimeError:
            previous_loop = None

    try:
        yield
    finally:
        if previous_loop is not None:
            asyncio.set_event_loop(previous_loop)


class LoopBoundRecordingTransport(httpx.AsyncBaseTransport):
    """Record whether cleanup reached the transport on its owning loop."""

    def __init__(self) -> None:
        self.owner_loop = asyncio.get_running_loop()
        self.aclose_calls = 0
        self.close_calls = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def aclose(self) -> None:
        self.aclose_calls += 1
        if asyncio.get_running_loop() is not self.owner_loop:
            raise RuntimeError("transport closed from a foreign event loop")
        self.closed = True

    def close(self) -> None:
        """Provide an observable synchronous fallback for a dead owner loop."""
        self.close_calls += 1
        self.closed = True


class UncloseableTransport(httpx.AsyncBaseTransport):
    """Model a transport for which cleanup genuinely cannot complete."""

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    async def aclose(self) -> None:
        raise RuntimeError("transport cleanup failed")


def test_aclose_releases_transports_owned_by_closed_loops() -> None:
    """Sequential ``asyncio.run`` clients must not lose their transports."""
    transports: list[LoopBoundRecordingTransport] = []
    clients: list[httpx.AsyncClient] = []

    def client_factory() -> httpx.AsyncClient:
        transport = LoopBoundRecordingTransport()
        client = httpx.AsyncClient(transport=transport)
        transports.append(transport)
        clients.append(client)
        return client

    cas = KuboCAS(client_factory=client_factory)

    async def create_client() -> None:
        cas._loop_client()

    asyncio.run(create_client())
    asyncio.run(create_client())

    assert len(transports) == 2
    assert all(transport.owner_loop.is_closed() for transport in transports)

    asyncio.run(cas.aclose())

    # AsyncClient marks itself closed before awaiting its transport, so the
    # transport state is the resource-release assertion that matters here.
    assert all(client.is_closed for client in clients)
    assert [transport.closed for transport in transports] == [True, True]


def test_aclose_warns_for_stock_transport_owned_by_closed_loop() -> None:
    """A stock transport on a dead loop degrades to warned best-effort cleanup."""
    server_ports: queue.Queue[int] = queue.Queue()
    server_errors: queue.Queue[BaseException] = queue.Queue()
    stop_server = threading.Event()

    async def handle_request(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await reader.readuntil(b"\r\n\r\n")
            writer.write(
                b"HTTP/1.1 200 OK\r\n"
                b"Content-Length: 2\r\n"
                b"Connection: keep-alive\r\n"
                b"\r\n"
                b"OK"
            )
            await writer.drain()
            await reader.read()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def serve() -> None:
        server = await asyncio.start_server(handle_request, "127.0.0.1", 0)
        assert server.sockets is not None
        server_ports.put(server.sockets[0].getsockname()[1])
        try:
            while not stop_server.is_set():
                await asyncio.sleep(0.01)
        finally:
            server.close()
            await server.wait_closed()

    def run_server() -> None:
        try:
            asyncio.run(serve())
        except BaseException as exc:
            server_errors.put(exc)

    server_thread = threading.Thread(target=run_server)
    server_thread.start()
    server_port = server_ports.get(timeout=5)
    cas = KuboCAS()

    async def make_keep_alive_request() -> None:
        response = await cas._loop_client().get(f"http://127.0.0.1:{server_port}/")
        assert response.status_code == 200

    try:
        asyncio.run(make_keep_alive_request())
        owner_loop = next(iter(cas._client_per_loop))
        assert owner_loop.is_closed()

        with pytest.warns(
            RuntimeWarning,
            match="Failed to close an internally created HTTP client",
        ):
            asyncio.run(cas.aclose())

        assert cas._client_per_loop == {}
        assert cas._closed
    finally:
        stop_server.set()
        server_thread.join(timeout=5)

    assert not server_thread.is_alive()
    if not server_errors.empty():
        raise server_errors.get()


def test_aclose_reports_a_transport_cleanup_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unrecoverable close failure must not disappear without a warning."""
    cas = KuboCAS(
        client_factory=lambda: httpx.AsyncClient(transport=UncloseableTransport())
    )

    async def create_and_close() -> None:
        cas._loop_client()
        await cas.aclose()

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        with caplog.at_level(logging.WARNING):
            asyncio.run(create_and_close())

    diagnostics = [str(item.message) for item in caught_warnings]
    diagnostics.extend(record.getMessage() for record in caplog.records)
    assert diagnostics, "aclose() silently swallowed the transport cleanup failure"


def test_aclose_awaits_current_loop_transport_normally() -> None:
    """A transport owned by the active loop keeps the normal async path."""
    transports: list[LoopBoundRecordingTransport] = []

    def client_factory() -> httpx.AsyncClient:
        transport = LoopBoundRecordingTransport()
        transports.append(transport)
        return httpx.AsyncClient(transport=transport)

    cas = KuboCAS(client_factory=client_factory)

    async def create_and_close() -> None:
        cas._loop_client()
        await cas.aclose()

    asyncio.run(create_and_close())

    assert len(transports) == 1
    assert transports[0].closed
    assert transports[0].aclose_calls == 1
    assert transports[0].close_calls == 0
