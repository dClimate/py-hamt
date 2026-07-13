import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from py_hamt import KuboCAS

REDIRECT_CID = "redirect"
EXPECTED_BODY = b"loaded after following the gateway redirect"


class _RedirectGatewayHandler(BaseHTTPRequestHandler):
    """Redirect one CID path to a successful local gateway response."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path == f"/ipfs/{REDIRECT_CID}":
            self.send_response(301)
            self.send_header("Location", "/ipfs/plain")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if self.path == "/ipfs/plain":
            self.send_response(200)
            self.send_header("Content-Length", str(len(EXPECTED_BODY)))
            self.end_headers()
            self.wfile.write(EXPECTED_BODY)
            return

        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()


@pytest.fixture
def redirect_gateway_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _RedirectGatewayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


@pytest.mark.asyncio
async def test_internal_client_follows_gateway_redirect(
    redirect_gateway_url: str,
) -> None:
    cas = KuboCAS(
        gateway_base_url=redirect_gateway_url,
        rpc_base_url=redirect_gateway_url,
        max_retries=0,
    )
    try:
        assert await cas.load(REDIRECT_CID) == EXPECTED_BODY
    finally:
        await cas.aclose()
