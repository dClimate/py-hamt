import asyncio

import httpx
from multiformats import CID

from py_hamt import KuboCAS


def test_supplied_client_is_bound_lazily_outside_event_loop() -> None:
    """A supplied client can be constructed synchronously and used later."""
    expected_body = b"loaded through the supplied client"
    requests: list[httpx.Request] = []

    def handle_request(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, content=expected_body, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handle_request))
    try:
        cas = KuboCAS(
            client=client,
            gateway_base_url="http://127.0.0.1:1",
            rpc_base_url="http://127.0.0.1:1",
        )

        async def load_and_close() -> bytes:
            try:
                cid = CID.decode(
                    "bafyreihyrpefhacm6kkp4ql6j6udakdit7g3dmkzfriqfykhjw6cad7lrm"
                )
                return await cas.load(cid)
            finally:
                await client.aclose()

        loaded = asyncio.run(load_and_close())
    finally:
        if not client.is_closed:
            asyncio.run(client.aclose())

    assert loaded == expected_body
    assert len(requests) == 1
    assert requests[0].method == "GET"
