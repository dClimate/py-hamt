import asyncio
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Barrier, Thread

import httpx
import pytest

from py_hamt import KuboCAS


@pytest.fixture
def gateway() -> Iterator[str]:
    """Offline gateway that answers every request with a small 200 body."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, message_format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()


def test_supplied_client_bound_to_one_loop_under_concurrent_first_use(
    gateway: str,
) -> None:
    """Two loops racing on first use must not bind one client to both.

    Before the first-use lock, two event loops on different threads could each
    observe ``_supplied_client`` as non-None and bind the same
    ``httpx.AsyncClient`` to both loops, which fails at request time with a
    "bound to a different event loop" error. The lock serializes consumption so
    the supplied client is used by exactly one loop and the other loop falls
    back to a distinct internally created client.
    """
    supplied_client = httpx.AsyncClient()
    cas = KuboCAS(
        client=supplied_client,
        gateway_base_url=gateway,
        rpc_base_url=gateway,
    )

    barrier = Barrier(2)
    errors: list[BaseException] = []

    def worker(cid: str) -> None:
        async def run() -> None:
            barrier.wait()  # release both threads into first-use together
            await cas.load(cid)

        try:
            asyncio.run(run())
        except Exception as exc:  # pragma: no cover - only trips on regression
            errors.append(exc)

    threads = [Thread(target=worker, args=(f"cid-{i}",)) for i in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    try:
        assert not errors, f"concurrent first-use raised: {errors!r}"

        bound_to_supplied = [
            loop
            for loop, client in cas._client_per_loop.items()
            if client is supplied_client
        ]
        assert len(bound_to_supplied) == 1, (
            "supplied client should be bound to exactly one loop, "
            f"got {len(bound_to_supplied)}"
        )
        assert cas._supplied_client is None
        assert len(cas._client_per_loop) == 2
    finally:

        async def cleanup() -> None:
            for client in {*cas._client_per_loop.values(), supplied_client}:
                if not client.is_closed:
                    await client.aclose()

        asyncio.run(cleanup())
