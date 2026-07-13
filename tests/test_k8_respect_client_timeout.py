import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest
from multiformats import CID

from py_hamt import KuboCAS

SLOW_RESPONSE_DELAY = 0.5
USER_TIMEOUT = 0.05
TEST_CID = CID.decode("bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm")


class _SlowKuboHandler(BaseHTTPRequestHandler):
    """Serve slow gateway and RPC responses without external network access."""

    def log_message(self, format: str, *args: object) -> None:
        pass

    def _send_slow_response(self, body: bytes, content_type: str) -> None:
        time.sleep(SLOW_RESPONSE_DELAY)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # Once the defect is fixed, the client closes the timed-out socket.
            pass

    def do_GET(self) -> None:
        self._send_slow_response(b"slow gateway response", "application/octet-stream")

    def do_POST(self) -> None:
        body = ('{"Hash":"' + str(TEST_CID) + '"}').encode()
        self._send_slow_response(body, "application/json")


@pytest.fixture
def slow_kubo_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _SlowKuboHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


@pytest.mark.asyncio
async def test_load_respects_user_client_timeout(slow_kubo_url: str) -> None:
    async with httpx.AsyncClient(timeout=USER_TIMEOUT) as client:
        cas = KuboCAS(
            client=client,
            gateway_base_url=slow_kubo_url,
            rpc_base_url=slow_kubo_url,
            max_retries=0,
        )

        with pytest.raises(httpx.TimeoutException):
            await cas.load(TEST_CID)


@pytest.mark.asyncio
async def test_save_respects_user_client_timeout(slow_kubo_url: str) -> None:
    async with httpx.AsyncClient(timeout=USER_TIMEOUT) as client:
        cas = KuboCAS(
            client=client,
            gateway_base_url=slow_kubo_url,
            rpc_base_url=slow_kubo_url,
            max_retries=0,
        )

        with pytest.raises(httpx.TimeoutException):
            await cas.save(b"slow RPC request", codec="raw")
