import asyncio
import time

import httpx
import pytest

from py_hamt import KuboCAS


@pytest.mark.asyncio
async def test_load_releases_semaphore_during_retry_backoff() -> None:
    """A failing CID's retry delays must not consume the only request slot."""
    expected_body = b"healthy gateway response"

    def handle_request(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bad-cid"):
            raise httpx.ConnectError("simulated connection failure", request=request)
        return httpx.Response(200, content=expected_body, request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handle_request)
    ) as client:
        cas = KuboCAS(
            client=client,
            gateway_base_url="https://gateway.test",
            rpc_base_url="https://rpc.test",
            concurrency=1,
            max_retries=2,
            initial_delay=0.4,
            backoff_factor=2.0,
        )

        async def failing_load() -> bytes:
            return await cas.load("bad-cid")

        async def healthy_load() -> tuple[bytes, float]:
            await asyncio.sleep(0.05)
            started_at = time.perf_counter()
            body = await cas.load("good-cid")
            return body, time.perf_counter() - started_at

        try:
            failure, healthy_result = await asyncio.gather(
                failing_load(), healthy_load(), return_exceptions=True
            )
        finally:
            await cas.aclose()

    assert isinstance(failure, httpx.ConnectError)
    assert not isinstance(healthy_result, BaseException)
    healthy_body, healthy_elapsed = healthy_result
    assert healthy_body == expected_body
    assert healthy_elapsed < 0.5, (
        "healthy load waited behind another load's retry backoff: "
        f"{healthy_elapsed:.3f}s elapsed"
    )
