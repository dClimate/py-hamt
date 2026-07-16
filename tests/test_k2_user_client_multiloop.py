import asyncio
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TypeAlias

import httpx
import pytest

from py_hamt import KuboCAS

RecordedHeaders: TypeAlias = dict[str, dict[str, str]]


@pytest.fixture(autouse=True)
def preserve_current_event_loop() -> Iterator[None]:
    """Restore the current loop after tests that use ``asyncio.run``."""
    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None

    try:
        yield
    finally:
        if previous_loop is not None:
            asyncio.set_event_loop(previous_loop)


@pytest.fixture
def recording_gateway() -> Iterator[tuple[str, RecordedHeaders]]:
    """Run an offline HTTP gateway that records headers by requested CID."""
    recorded_headers: RecordedHeaders = {}

    class RecordingHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            cid = self.path.removeprefix("/ipfs/")
            recorded_headers[cid] = dict(self.headers.items())
            body = b"gateway response"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message_format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), RecordingHandler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]

    try:
        yield f"http://127.0.0.1:{port}", recorded_headers
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


def _build_cas_on_first_loop(gateway_url: str) -> tuple[KuboCAS, httpx.AsyncClient]:
    cas_holder: list[KuboCAS] = []
    client_holder: list[httpx.AsyncClient] = []

    async def phase_one() -> None:
        supplied_client = httpx.AsyncClient(headers={"Authorization": "Bearer secret"})
        cas = KuboCAS(
            client=supplied_client,
            gateway_base_url=gateway_url,
            rpc_base_url=gateway_url,
        )
        await cas.load("auth1")
        cas_holder.append(cas)
        client_holder.append(supplied_client)

    asyncio.run(phase_one())
    return cas_holder[0], client_holder[0]


def _close_all_clients(cas: KuboCAS, supplied_client: httpx.AsyncClient) -> None:
    async def close_clients() -> None:
        for client in {*cas._client_per_loop.values(), supplied_client}:
            if not client.is_closed:
                await client.aclose()

    asyncio.run(close_clients())


def test_user_client_auth_is_propagated_to_second_loop(
    recording_gateway: tuple[str, RecordedHeaders],
) -> None:
    gateway_url, recorded_headers = recording_gateway
    cas, supplied_client = _build_cas_on_first_loop(gateway_url)

    try:
        asyncio.run(cas.load("auth2"))

        assert recorded_headers["auth1"].get("Authorization") == "Bearer secret"
        assert recorded_headers["auth2"].get("Authorization") == "Bearer secret"
    finally:
        _close_all_clients(cas, supplied_client)


def test_aclose_closes_only_internally_created_second_loop_client(
    recording_gateway: tuple[str, RecordedHeaders],
) -> None:
    gateway_url, _ = recording_gateway
    cas, supplied_client = _build_cas_on_first_loop(gateway_url)

    internally_created_clients: list[httpx.AsyncClient] = []

    async def phase_two() -> None:
        await cas.load("auth2")
        # Snapshot internally-created clients BEFORE aclose(), since a
        # correct aclose() implementation may clear the per-loop mapping.
        internally_created_clients.extend(
            client
            for client in cas._client_per_loop.values()
            if client is not supplied_client
        )
        await cas.aclose()

    try:
        asyncio.run(phase_two())

        assert not supplied_client.is_closed
        assert internally_created_clients
        assert all(client.is_closed for client in internally_created_clients)
    finally:
        _close_all_clients(cas, supplied_client)
