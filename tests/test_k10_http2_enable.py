import asyncio
import ssl
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Thread
from typing import cast

import pytest
import trustme
from h2 import events
from h2.config import H2Configuration
from h2.connection import H2Connection
from multiformats import CID

from py_hamt import KuboCAS

EXPECTED_BODY = b"gateway response across HTTP/2 GOAWAY"
TEST_CID = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")
MAX_STREAMS = 4
CONCURRENT_LOADS = 12
GOAWAY_CLOSE_DELAY = 0.05


@dataclass
class TLSTestGateway:
    url: str
    active_transports: set[asyncio.WriteTransport] = field(default_factory=set)


class GoAwayGatewayProtocol(asyncio.Protocol):
    """Serve HTTP/1.1 or raw HTTP/2 according to the negotiated ALPN protocol."""

    def __init__(self, gateway: TLSTestGateway) -> None:
        self.gateway = gateway
        self.transport: asyncio.WriteTransport | None = None
        self.http1_buffer = bytearray()
        self.http1_response_sent = False
        self.h2_connection = H2Connection(
            config=H2Configuration(client_side=False, header_encoding="utf-8")
        )
        self.h2_streams_served = 0
        self.goaway_sent = False
        self.negotiated_protocol: str | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self.transport = cast(asyncio.WriteTransport, transport)
        self.gateway.active_transports.add(self.transport)
        ssl_object = cast(ssl.SSLObject, transport.get_extra_info("ssl_object"))
        self.negotiated_protocol = ssl_object.selected_alpn_protocol()

        if self.negotiated_protocol == "h2":
            self.h2_connection.initiate_connection()
            self.transport.write(self.h2_connection.data_to_send())

    def data_received(self, data: bytes) -> None:
        if self.negotiated_protocol == "h2":
            self._receive_http2(data)
        else:
            self._receive_http1(data)

    def connection_lost(self, exc: Exception | None) -> None:
        if self.transport is not None:
            self.gateway.active_transports.discard(self.transport)
        self.transport = None

    def _receive_http1(self, data: bytes) -> None:
        if self.transport is None or self.http1_response_sent:
            return
        self.http1_buffer.extend(data)
        if b"\r\n\r\n" not in self.http1_buffer:
            return

        self.http1_response_sent = True
        response = (
            b"HTTP/1.1 200 OK\r\n"
            + f"Content-Length: {len(EXPECTED_BODY)}\r\n".encode()
            + b"Content-Type: application/octet-stream\r\n"
            + b"Connection: close\r\n\r\n"
            + EXPECTED_BODY
        )
        self.transport.write(response)
        self.transport.close()

    def _receive_http2(self, data: bytes) -> None:
        if self.transport is None:
            return
        try:
            received_events = self.h2_connection.receive_data(data)
        except Exception:
            self.transport.close()
            return

        for event in received_events:
            if isinstance(event, events.RequestReceived):
                self._serve_http2_stream(event.stream_id)

        pending_data = self.h2_connection.data_to_send()
        if pending_data:
            self.transport.write(pending_data)

    def _serve_http2_stream(self, stream_id: int) -> None:
        if self.goaway_sent:
            return

        self.h2_streams_served += 1
        self.h2_connection.send_headers(
            stream_id,
            [
                (":status", "200"),
                ("content-length", str(len(EXPECTED_BODY))),
                ("content-type", "application/octet-stream"),
            ],
        )
        self.h2_connection.send_data(stream_id, EXPECTED_BODY, end_stream=True)

        if self.h2_streams_served >= MAX_STREAMS:
            self.goaway_sent = True
            self.h2_connection.close_connection(last_stream_id=stream_id)
            asyncio.get_running_loop().call_later(
                GOAWAY_CLOSE_DELAY, self._close_transport
            )

    def _close_transport(self) -> None:
        if self.transport is not None:
            self.transport.close()


@pytest.fixture
def tls_goaway_gateway(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[TLSTestGateway]:
    ca = trustme.CA()
    server_certificate = ca.issue_cert("127.0.0.1")
    ca_path = tmp_path / "ca.pem"
    ca.cert_pem.write_to_path(ca_path)
    monkeypatch.setenv("SSL_CERT_FILE", str(ca_path))

    ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    server_certificate.configure_cert(ssl_context)
    ssl_context.set_alpn_protocols(["h2", "http/1.1"])

    gateway = TLSTestGateway(url="")
    ready = Event()
    server_loop = asyncio.new_event_loop()
    setup_errors: list[BaseException] = []

    def run_server() -> None:
        asyncio.set_event_loop(server_loop)
        server: asyncio.Server | None = None
        try:
            server = server_loop.run_until_complete(
                server_loop.create_server(
                    lambda: GoAwayGatewayProtocol(gateway),
                    host="127.0.0.1",
                    port=0,
                    ssl=ssl_context,
                )
            )
            port = server.sockets[0].getsockname()[1]
            gateway.url = f"https://127.0.0.1:{port}"
            ready.set()
            server_loop.run_forever()
        except BaseException as error:
            setup_errors.append(error)
            ready.set()
        finally:
            if server is not None:
                server.close()
                server_loop.run_until_complete(server.wait_closed())
            for transport in list(gateway.active_transports):
                transport.close()
            server_loop.run_until_complete(asyncio.sleep(0))
            server_loop.close()

    server_thread = Thread(target=run_server, daemon=True)
    server_thread.start()
    if not ready.wait(timeout=5):
        raise RuntimeError("TLS test server did not start")
    if setup_errors:
        raise RuntimeError("TLS test server failed to start") from setup_errors[0]

    try:
        yield gateway
    finally:
        server_loop.call_soon_threadsafe(server_loop.stop)
        server_thread.join(timeout=5)
        if server_thread.is_alive():
            raise RuntimeError("TLS test server did not stop")


@pytest.mark.asyncio
async def test_internal_client_negotiates_http2(
    tls_goaway_gateway: TLSTestGateway,
) -> None:
    cas = KuboCAS(
        gateway_base_url=tls_goaway_gateway.url,
        rpc_base_url="http://127.0.0.1:1",
    )
    try:
        client = cas._loop_client()
        response = await client.get(f"{tls_goaway_gateway.url}/ipfs/{TEST_CID}")
        assert response.http_version == "HTTP/2"
    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_concurrent_loads_survive_http2_goaway(
    tls_goaway_gateway: TLSTestGateway,
) -> None:
    cas = KuboCAS(
        gateway_base_url=tls_goaway_gateway.url,
        rpc_base_url="http://127.0.0.1:1",
        max_retries=4,
        initial_delay=0.05,
        backoff_factor=1.5,
    )
    try:
        results = await asyncio.gather(
            *(cas.load(TEST_CID) for _ in range(CONCURRENT_LOADS))
        )
    finally:
        await cas.aclose()

    assert results == [EXPECTED_BODY] * CONCURRENT_LOADS
