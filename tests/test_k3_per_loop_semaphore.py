import asyncio
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from py_hamt import KuboCAS


def test_contended_loads_work_across_sequential_event_loops() -> None:
    try:
        previous_loop = asyncio.get_event_loop()
    except RuntimeError:
        previous_loop = None

    expected_body = b"loaded from the mock gateway"
    request_started = threading.Event()
    release_response = threading.Event()

    class GatewayHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            request_started.set()
            release_response.wait()
            self.send_response(200)
            self.send_header("Content-Length", str(len(expected_body)))
            self.end_headers()
            self.wfile.write(expected_body)

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), GatewayHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_address[1]}"
    cas = KuboCAS(
        gateway_base_url=base_url,
        rpc_base_url=base_url,
        concurrency=1,
        max_retries=0,
    )

    async def run_contended_loads() -> list[bytes]:
        request_started.clear()
        release_response.clear()

        first_load = asyncio.create_task(cas.load("plain"))
        await asyncio.to_thread(request_started.wait)
        second_load = asyncio.create_task(cas.load("plain"))
        await asyncio.sleep(0)
        release_response.set()

        return list(await asyncio.gather(first_load, second_load))

    try:
        assert asyncio.run(run_contended_loads()) == [expected_body, expected_body]
        assert asyncio.run(run_contended_loads()) == [expected_body, expected_body]
    finally:
        release_response.set()
        asyncio.run(cas.aclose())
        server.shutdown()
        server.server_close()
        server_thread.join()
        if previous_loop is not None:
            asyncio.set_event_loop(previous_loop)
