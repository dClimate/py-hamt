import asyncio

import dag_cbor
import httpx
import pytest

from py_hamt import KuboCAS

"""
Tests for IPFS gateway functionality.

Note: The GitHub Actions setup-ipfs creates a fresh, empty IPFS node.
Tests must first add content before trying to retrieve it, or use
well-known CIDs that might be available on public gateways.
"""

# Well-known test CID from IPFS examples (may or may not be available)
TEST_CID = "bafybeifx7yeb55armcsxwwitkymga5xf53dxiarykms3ygqic223w5sk3m"


async def verify_response_content(url: str, client=None, timeout=30.0):
    """Fetch and verify the response from a given URL"""
    should_close = False
    if client is None:
        client = httpx.AsyncClient(follow_redirects=True, timeout=timeout)
        should_close = True

    try:
        # Print request info
        print(f"Testing URL: {url}")

        # Fetch content
        response = await client.get(url, timeout=timeout)
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get("content-type", "")
        print(f"Content-Type: {content_type}")

        # First few bytes for debug
        content = response.content
        print(
            f"First 20 bytes: {content[:20].hex() if len(content) >= 20 else content.hex()}"
        )
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
    except httpx.TimeoutException:
        return {"url": url, "error": "Timeout"}
    except Exception as e:
        return {"url": url, "error": str(e)}
    finally:
        if should_close:
            await client.aclose()


@pytest.mark.asyncio
async def test_compare_gateways():
    """Compare response content from different IPFS gateways"""

    # First, let's create some content on the local node
    cas = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
    )

    test_data = b"Test content for gateway comparison"
    local_cid = None

    try:
        local_cid = await cas.save(test_data, codec="raw")
        print(f"Created local test CID: {local_cid}")
        await asyncio.sleep(0.5)  # Give IPFS time to process
    finally:
        await cas.aclose()

    # Test URLs - use our local CID for local gateway, known CID for public gateways
    gateways = [
        (
            f"http://127.0.0.1:8080/ipfs/{local_cid}",
            "Local gateway",
            True,
        ),  # Should work
        (
            f"https://ipfs.io/ipfs/{TEST_CID}",
            "IPFS.io public gateway",
            False,
        ),  # May or may not work
        (
            f"https://dweb.link/ipfs/{TEST_CID}",
            "Protocol Labs gateway",
            False,
        ),  # May or may not work
    ]

    # Create a single client for all requests
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        # Test each gateway
        results = []
        for url, name, must_succeed in gateways:
            result = await verify_response_content(url, client, timeout=10.0)
            result["name"] = name
            result["must_succeed"] = must_succeed
            results.append(result)

    # Print comparison
    successful_results = []
    failed_required = []

    for result in results:
        print(f"\nGateway: {result.get('name')}")
        print(f"URL: {result.get('url')}")

        if "error" in result:
            print(f"  Error: {result['error']}")
            if result.get("must_succeed", False):
                failed_required.append(result)
        else:
            print(f"  Status: {result.get('status_code')}")
            print(f"  Content-Type: {result.get('content_type')}")
            print(f"  Content Length: {result.get('content_length')}")
            print(f"  First Byte: {result.get('first_byte')}")
            successful_results.append(result)

    # Ensure required gateways succeeded
    if failed_required:
        pytest.fail(f"Required gateways failed: {[r['name'] for r in failed_required]}")

    # We should have at least the local gateway working
    assert len(successful_results) > 0, "No gateways returned successful responses"


@pytest.mark.asyncio
async def test_kubocas_public_gateway():
    """Test KuboCAS with a public gateway"""

    # For this test, we'll use the local daemon to save content,
    # then test loading through a "public" gateway (actually local gateway)
    # This ensures the content exists and tests the gateway functionality

    cas_save = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
        max_retries=0,
    )

    try:
        # First save some test data
        test_data = b"Testing public gateway functionality with known content"
        test_cid = await cas_save.save(test_data, codec="raw")
        print(f"Saved test data with CID: {test_cid}")

        # Give IPFS a moment to make the content available
        await asyncio.sleep(1.0)

    finally:
        await cas_save.aclose()

    # Now test loading through different gateway configurations
    test_gateways = [
        # Test local gateway as if it were a public gateway
        ("http://127.0.0.1:8080", "local gateway"),
        # Could add actual public gateways here, but they're unreliable for CI
        ("https://ipfs.io", "ipfs.io public gateway"),
    ]

    for gateway_url, gateway_name in test_gateways:
        cas = KuboCAS(
            rpc_base_url="http://127.0.0.1:5001",  # Keep local RPC for saves
            gateway_base_url=gateway_url,  # Use specified gateway for loads
        )

        try:
            # Try to load the CID we just saved
            loaded_data = await cas.load(test_cid)

            # Print info for debugging
            print(f"Successfully loaded {len(loaded_data)} bytes from {gateway_name}")
            print(
                f"First 20 bytes: {loaded_data[:20].hex() if len(loaded_data) >= 20 else loaded_data.hex()}"
            )

            # Verify we got the correct data
            assert loaded_data == test_data, f"Data mismatch from {gateway_name}"

            print(f"✓ {gateway_name} test passed")

        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError) as e:
            print(f"✗ {gateway_name} failed: {e}")
            # Don't fail the test if a public gateway is down
            if "ipfs.io" in gateway_url or "dweb.link" in gateway_url:
                pytest.skip(f"{gateway_name} appears to be down: {e}")
            else:
                # Re-raise for local gateway errors
                raise

        finally:
            await cas.aclose()

    print("Public gateway test completed successfully")


@pytest.mark.asyncio
async def test_trailing_slash_gateway():
    """Test KuboCAS with a gateway URL that has a trailing slash"""

    # Use a gateway URL with a trailing slash
    cas = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080/",  # Note the trailing slash
        max_retries=0,
    )

    try:
        # First, let's save some data so we know it exists locally
        test_data = b"Hello from trailing slash test! This tests that URLs are properly constructed."
        test_cid = await cas.save(test_data, codec="raw")
        print(f"Saved test data with CID: {test_cid}")

        # Give IPFS a moment to process the data
        await asyncio.sleep(0.5)

        # Now try to load it back through the gateway
        # This tests that the trailing slash in gateway_base_url is handled correctly
        loaded_data = await cas.load(test_cid)

        # Verify we got the same data back
        assert loaded_data == test_data, "Loaded data doesn't match saved data"

        print(
            f"Successfully loaded {len(loaded_data)} bytes from gateway with trailing slash"
        )

        # Also test that the URL construction is correct by checking the gateway_base_url
        assert cas.gateway_base_url == "http://127.0.0.1:8080/ipfs/", (
            "Gateway URL not properly formatted"
        )

    except httpx.ConnectError:
        pytest.skip("Local IPFS daemon not running - skipping test")
    except httpx.ReadTimeout:
        # This might happen if the gateway is slow to start
        pytest.skip("Local gateway read timeout - may still be starting up")
    except asyncio.TimeoutError:
        pytest.skip("Local gateway timed out - may be under heavy load")
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 504:
            pytest.skip("Local gateway returned 504 - may be starting up")
        elif e.response.status_code == 500:
            pytest.skip(
                "Local gateway returned 500 - internal error, may need more time"
            )
        else:
            raise
    finally:
        await cas.aclose()


async def test_fix_kubocas_load():
    """Test URL construction and loading behavior of KuboCAS"""

    # Test URL construction with various gateway configurations
    test_cases = [
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080/ipfs/"),
        ("http://127.0.0.1:8080/", "http://127.0.0.1:8080/ipfs/"),
        ("https://ipfs.io", "https://ipfs.io/ipfs/"),
        ("https://ipfs.io/", "https://ipfs.io/ipfs/"),
        ("https://gateway.ipfs.io/ipfs/", "https://gateway.ipfs.io/ipfs/"),
    ]

    for input_url, expected_base in test_cases:
        cas = KuboCAS(
            rpc_base_url="http://127.0.0.1:5001",
            gateway_base_url=input_url,
            max_retries=0,
        )
        assert cas.gateway_base_url == expected_base, (
            f"URL construction failed for {input_url}"
        )
        await cas.aclose()

    # Test actual loading with local gateway
    cas = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
        max_retries=0,
    )

    try:
        # Save and load test data
        test_data = b"Testing KuboCAS load functionality"
        cid = await cas.save(test_data, codec="raw")

        # Small delay to ensure data is available
        await asyncio.sleep(0.5)

        loaded_data = await cas.load(cid)
        assert loaded_data == test_data, "Loaded data doesn't match saved data"

        print(f"✓ KuboCAS load test passed - loaded {len(loaded_data)} bytes")

    except httpx.ConnectError:
        pytest.skip("Local IPFS daemon not running")
    finally:
        await cas.aclose()


SMALL_DAG_CBOR_CID = "bafyreibwzifwg3a3z5h6vxxalxdtfv5ihof6j4mhy4cl4kxh3fbxn6v2iq"


@pytest.mark.asyncio
async def test_local_dag_cbor_accept_header():
    """Local gateway should honour Accept: application/vnd.ipld.dag-cbor"""

    # Step 1: Minimal DAG-CBOR object (e.g., [1, 2, 3])
    dag_cbor_data = dag_cbor.encode([1, 2, 3])  # Expected output: b'\x83\x01\x02\x03'

    # Step 2: Store it via Kubo RPC as dag-cbor
    cas = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
    )

    try:
        cid = await cas.save(dag_cbor_data, codec="dag-cbor")
        print(f"Saved DAG-CBOR CID: {cid}")
        await asyncio.sleep(0.5)  # Give IPFS time to index the block
    finally:
        await cas.aclose()

    # Step 3: Fetch using Accept header
    url = f"http://127.0.0.1:8080/ipfs/{cid}"
    headers = {"Accept": "application/vnd.ipld.dag-cbor"}

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        try:
            r = await client.get(url, headers=headers)
        except (httpx.ConnectError, httpx.TimeoutException):
            pytest.skip("Local IPFS gateway not reachable")

    if r.status_code >= 500:
        pytest.skip(f"Local gateway returned {r.status_code}")

    # Step 4: Verify response
    assert r.headers.get("content-type", "").startswith(
        "application/vnd.ipld.dag-cbor"
    ), "Gateway did not honor Accept header"

    assert r.content[:1] == b"\x83", "Response is not a DAG-CBOR array of length 3"


if __name__ == "__main__":
    asyncio.run(test_compare_gateways())
