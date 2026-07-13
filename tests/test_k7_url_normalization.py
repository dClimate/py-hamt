import pytest

from py_hamt.store_httpx import KuboCAS


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gateway_base_url",
    [
        "https://example.com",
        "https://example.com/",
        "https://example.com/ipfs",
        "https://example.com/ipfs/",
    ],
)
async def test_gateway_base_url_normalizes_ipfs_path(
    gateway_base_url: str,
) -> None:
    cas = KuboCAS(
        gateway_base_url=gateway_base_url,
        rpc_base_url="https://example.com",
    )

    try:
        assert cas.gateway_base_url == "https://example.com/ipfs/"
    finally:
        await cas.aclose()
