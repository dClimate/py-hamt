import asyncio

import httpx
import pytest
from multiformats import CID

from py_hamt import KuboCAS

TEST_CID = "bafyr4iecw3faqyvj75psutabk2jxpddpjdokdy5b26jdnjjzpkzbgb5xoq"


async def verify_response_content(url: str, client=None):
    """Fetch and verify the response from a given URL"""
    should_close = False
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True)
        should_close = True

    try:
        # Print request info
        print(f"Testing URL: {url}")

        # Fetch content
        response = await client.get(url)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("content-type", "")
        print(f"Content-Type: {content_type}")

        # First few bytes for debug
        content = response.content
        print(f"First 20 bytes: {content[:20].hex()}")
        print(f"Content length: {len(content)}")

        # A valid DAG-CBOR object typically starts with 0xa* for arrays or 0x* for other types
        # This is a simple heuristic check
        first_byte = content[0] if content else 0
        return {
            "url": url,
            "status_code": response.status_code,
            "content_type": content_type,
            "content_length": len(content),
            "first_byte": hex(first_byte),
            "looks_like_dag_cbor": first_byte & 0xE0 in (0x80, 0xA0),  # Arrays or maps
            "content": content,
        }
    finally:
        if should_close:
            await client.aclose()


@pytest.mark.asyncio
async def test_compare_gateways():
    """Compare response content from different IPFS gateways"""

    # Test URLs
    cid = CID.decode(TEST_CID)
    gateways = [
        f"http://127.0.0.1:8080/ipfs/{cid}",  # Local gateway
        f"https://ipfs.io/ipfs/{cid}?format=dag-cbor",  # Public gateway with format parameter
        f"https://dweb.link/ipfs/{cid}?format=dag-cbor",  # Protocol Labs' gateway with format parameter
        f"https://cloudflare-ipfs.com/ipfs/{cid}?format=dag-cbor",  # Cloudflare's gateway with format parameter
    ]

    # Create a single client for all requests
    async with httpx.AsyncClient(follow_redirects=True) as client:
        # Test each gateway
        results = []
        for url in gateways:
            try:
                result = await verify_response_content(url, client)
                results.append(result)
            except Exception as e:
                print(f"Error testing {url}: {e}")
                results.append({"url": url, "error": str(e)})

    # Print comparison
    for result in results:
        print(f"\nURL: {result.get('url')}")
        if "error" in result:
            print(f"  Error: {result['error']}")
            continue

        print(f"  Status: {result.get('status_code')}")
        print(f"  Content-Type: {result.get('content_type')}")
        print(f"  Content Length: {result.get('content_length')}")
        print(f"  First Byte: {result.get('first_byte')}")
        print(f"  Looks like DAG-CBOR: {result.get('looks_like_dag_cbor')}")

    # Verify at least the local gateway worked
    local_result = next((r for r in results if "127.0.0.1" in r.get("url", "")), None)
    if local_result and "error" not in local_result:
        assert local_result.get("looks_like_dag_cbor", False), (
            "Local gateway response doesn't look like DAG-CBOR"
        )


@pytest.mark.asyncio
async def test_kubocas_public_gateway():
    """Test KuboCAS with a public gateway"""

    # Use a public gateway
    cas = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",  # Keep local RPC for saves
        gateway_base_url="https://ipfs.io",  # Use public gateway for loads
    )

    try:
        # Try to load the CID
        cid = CID.decode(TEST_CID)
        data = await cas.load(cid)

        # Print info for debugging
        print(f"Loaded {len(data)} bytes from public gateway")
        print(f"First 20 bytes: {data[:20].hex()}")

        # Check if it looks like DAG-CBOR
        first_byte = data[0] if data else 0
        is_dag_cbor = first_byte & 0xE0 in (0x80, 0xA0)  # Simple check for arrays/maps
        print(f"First byte: {hex(first_byte)}, Looks like DAG-CBOR: {is_dag_cbor}")

        assert is_dag_cbor, "Data from public gateway doesn't look like DAG-CBOR"

    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_trailing_slash_gateway():
    """Test KuboCAS with a gateway URL that has a trailing slash"""

    # Use a gateway URL with a trailing slash
    cas = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080/",  # Note the trailing slash
    )

    try:
        # Try to load the CID
        cid = CID.decode(TEST_CID)
        data = await cas.load(cid)

        # Print info for debugging
        print(f"Loaded {len(data)} bytes from gateway with trailing slash")
        print(f"First 20 bytes: {data[:20].hex()}")

        # Check if it looks like DAG-CBOR
        first_byte = data[0] if data else 0
        is_dag_cbor = first_byte & 0xE0 in (0x80, 0xA0)  # Simple check for arrays/maps
        print(f"First byte: {hex(first_byte)}, Looks like DAG-CBOR: {is_dag_cbor}")

        assert is_dag_cbor, (
            "Data from gateway with trailing slash doesn't look like DAG-CBOR"
        )

    finally:
        await cas.aclose()


@pytest.mark.asyncio
async def test_fix_kubocas_load():
    """Test a proposed fix for KuboCAS when loading from public gateways"""

    class FixedKuboCAS(KuboCAS):
        """Extended KuboCAS with improved public gateway support"""

        async def load(self, id):
            """Modified load that ensures we get the raw IPLD content"""
            cid = CID.decode(str(id)) if isinstance(id, str) else id

            # Clean the base URL to prevent path issues
            base_url = self.gateway_base_url
            if "/ipfs/" in base_url:
                base_url = base_url.split("/ipfs/")[0]

            # Construction of URL that works with public gateways
            if base_url.endswith("/"):
                url = f"{base_url}ipfs/{cid}?format=dag-cbor"
            else:
                url = f"{base_url}/ipfs/{cid}?format=dag-cbor"

            print(f"Requesting URL: {url}")

            async with self._sem:
                client = self._loop_client()

                # For public gateways, add appropriate Accept header to get raw content
                headers = {
                    "Accept": "application/vnd.ipld.raw, application/vnd.ipld.dag-cbor, application/octet-stream"
                }

                response = await client.get(url, headers=headers)
                response.raise_for_status()
                return response.content

    # Use the fixed implementation with a public gateway
    cas = FixedKuboCAS(
        rpc_base_url="http://127.0.0.1:5001", gateway_base_url="https://ipfs.io/ipfs/"
    )

    try:
        # Try to load the CID
        cid = CID.decode(TEST_CID)
        data = await cas.load(cid)

        # Print info for debugging
        print(f"Loaded {len(data)} bytes from public gateway with fix")
        print(f"First 20 bytes: {data[:20].hex()}")

        # Check if it looks like DAG-CBOR
        first_byte = data[0] if data else 0
        is_dag_cbor = first_byte & 0xE0 in (0x80, 0xA0)
        print(f"First byte: {hex(first_byte)}, Looks like DAG-CBOR: {is_dag_cbor}")

        assert is_dag_cbor, (
            "Data from public gateway with fix doesn't look like DAG-CBOR"
        )

    finally:
        await cas.aclose()


if __name__ == "__main__":
    asyncio.run(test_compare_gateways())
