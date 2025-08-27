import asyncio
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import dag_cbor
import httpx
import pytest
from dag_cbor import IPLDKind
from hypothesis import given, settings
from testing_utils import ipld_strategy  # noqa

from py_hamt import InMemoryCAS, KuboCAS


# Just to cover this one case that isn't covered within test_hamt
@pytest.mark.asyncio
async def test_memory_store_exception():
    s = InMemoryCAS()
    with pytest.raises(KeyError):
        await s.load(bytes())


@pytest.mark.asyncio
async def test_memory_store_invalid_key_type():
    """Test that InMemoryCAS.load raises TypeError for non-bytes keys"""
    s = InMemoryCAS()

    # Test with various non-bytes types
    invalid_keys = [
        "string_key",
        123,
        12.34,
        ["list", "key"],
        {"dict": "key"},
        None,
        True,
    ]

    for invalid_key in invalid_keys:
        with pytest.raises(
            TypeError,
            match=f"InMemoryCAS only supports byte‐hash keys; got {type(invalid_key).__name__}",
        ):
            await s.load(invalid_key)


# Test that always works with Docker or local daemon
@pytest.mark.ipfs
@pytest.mark.asyncio(loop_scope="session")
@given(data=ipld_strategy())
@settings(deadline=1000, print_blob=True)
async def test_kubo_urls_explicit(create_ipfs, global_client_session, data: IPLDKind):
    """
    Tests KuboCAS functionality with explicitly provided URLs.
    Works with both Docker containers and local IPFS daemons.
    """
    rpc_url, gateway_url = create_ipfs

    # Test the same functionality but with explicit URLs
    async with KuboCAS(
        rpc_base_url=rpc_url,
        gateway_base_url=gateway_url,
        client=global_client_session,
    ) as kubo_cas:
        encoded_data = dag_cbor.encode(data)
        for codec in ["raw", "dag-cbor"]:
            codec_typed = cast(Literal["raw", "dag-cbor"], codec)
            cid = await kubo_cas.save(encoded_data, codec=codec_typed)
            loaded_encoded_data = await kubo_cas.load(cid)
            result = dag_cbor.decode(loaded_encoded_data)
            assert data == result


@pytest.mark.ipfs
@pytest.mark.asyncio(loop_scope="session")
@given(data=ipld_strategy())
@settings(deadline=1000, print_blob=True)
async def test_kubo_default_urls(global_client_session, data: IPLDKind):
    """
    Tests KuboCAS using its default URLs and when None is passed for URLs.
    Requires a local IPFS daemon on default ports.
    """
    # Check if local IPFS daemon is available on default ports
    import http.client

    try:
        conn = http.client.HTTPConnection("127.0.0.1", 5001, timeout=1)
        conn.request("POST", "/api/v0/version")
        response = conn.getresponse()
        if response.status != 200:
            pytest.skip("No IPFS daemon running on default ports (127.0.0.1:5001)")
    except Exception:
        pytest.skip("No IPFS daemon running on default ports (127.0.0.1:5001)")

    # Your original test code continues here
    async with KuboCAS(client=global_client_session) as kubo_cas_default:
        encoded_data = dag_cbor.encode(data)
        for codec in ["raw", "dag-cbor"]:
            codec_typed = cast(Literal["raw", "dag-cbor"], codec)
            try:
                cid = await kubo_cas_default.save(encoded_data, codec=codec_typed)
                loaded_encoded_data = await kubo_cas_default.load(cid)
                result = dag_cbor.decode(loaded_encoded_data)
                assert data == result
            except Exception as e:
                pytest.fail(
                    f"Error during KuboCAS default URL test (codec: {codec}): {e}"
                )

    async with KuboCAS(
        rpc_base_url=None, gateway_base_url=None, client=global_client_session
    ) as kubo_cas_none_urls:
        encoded_data = dag_cbor.encode(data)
        for codec in ["raw", "dag-cbor"]:
            codec_typed = cast(Literal["raw", "dag-cbor"], codec)
            try:
                cid = await kubo_cas_none_urls.save(encoded_data, codec=codec_typed)
                loaded_encoded_data = await kubo_cas_none_urls.load(cid)
                result = dag_cbor.decode(loaded_encoded_data)
                assert data == result
            except Exception as e:
                pytest.fail(f"Error during KuboCAS None URL test (codec: {codec}): {e}")


@pytest.mark.asyncio
@given(data=ipld_strategy())
@settings(
    deadline=500
)  # this sometimes takes longer than the default 250 ms in GitHub CI
async def test_kubo_cas(create_ipfs, data: IPLDKind):  # noqa
    rpc_base_url, gateway_base_url = create_ipfs

    # Provide our own async Client, for complete code coverage
    async with httpx.AsyncClient() as client:
        async with KuboCAS(
            rpc_base_url=rpc_base_url,
            gateway_base_url=gateway_base_url,
            client=client,
        ) as kubo_cas:
            # Use proper literal types for codec
            codec_raw: Literal["raw"] = "raw"
            codec_dag_cbor: Literal["dag-cbor"] = "dag-cbor"
            for codec in [codec_raw, codec_dag_cbor]:
                codec_typed = cast(Literal["raw", "dag-cbor"], codec)
                cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec_typed)
                result = dag_cbor.decode(await kubo_cas.load(cid))
                assert data == result


async def test_chunker_valid_patterns():
    valid = ["size-1", "size-1024", "rabin", "rabin-16-32-64"]
    for chunker in valid:
        async with KuboCAS(
            chunker=chunker,
            rpc_base_url="http://127.0.0.1:5001",
            gateway_base_url="http://127.0.0.1:8080",
        ) as cas:
            assert f"chunker={chunker}" in cas.rpc_url


@pytest.mark.parametrize(
    "invalid",
    ["", "size-0", "size--1", "rabin-1-2", "foo", "rabin-1-2-0"],
)
async def test_chunker_invalid_patterns(invalid):
    with pytest.raises(ValueError, match="Invalid chunker specification"):
        async with KuboCAS(chunker=invalid):
            pass


@pytest.mark.asyncio
async def test_kubo_timeout_retries():
    """
    Test that KuboCAS handles timeouts with retries and exponential backoff
    for both save and load operations using unittest.mock.
    """
    timeout_count = 0
    successful_after = 2  # Succeed after 2 timeout attempts
    test_cid = "bafybeigdyrzt5sfp7udm7hu76uh7y26nf3efuylqabf3oclgtqy55fbzdi"

    async def mock_post(url, **kwargs):
        nonlocal timeout_count
        # Manually create a dummy request object
        dummy_request = httpx.Request("POST", url, files=kwargs.get("files"))
        if timeout_count < successful_after:
            timeout_count += 1
            raise httpx.TimeoutException("Simulated timeout", request=dummy_request)
        return httpx.Response(200, json={"Hash": test_cid}, request=dummy_request)

    async def mock_get(url, **kwargs):
        nonlocal timeout_count
        # Manually create a dummy request object
        dummy_request = httpx.Request("GET", url)
        if timeout_count < successful_after:
            timeout_count += 1
            raise httpx.TimeoutException("Simulated timeout", request=dummy_request)
        return httpx.Response(200, content=test_data, request=dummy_request)

    # Patch the httpx.AsyncClient methods
    with patch.object(httpx.AsyncClient, "post", new=AsyncMock(side_effect=mock_post)):
        with patch.object(
            httpx.AsyncClient, "get", new=AsyncMock(side_effect=mock_get)
        ):
            async with httpx.AsyncClient() as client:
                async with KuboCAS(
                    rpc_base_url="http://127.0.0.1:5001",
                    gateway_base_url="http://127.0.0.1:8080",
                    client=client,
                    max_retries=3,
                    initial_delay=0.1,
                    backoff_factor=2.0,
                ) as kubo_cas:
                    # Test save with retries
                    timeout_count = 0
                    test_data = dag_cbor.encode("test")
                    cid = await kubo_cas.save(test_data, codec="dag-cbor")
                    assert timeout_count == successful_after, (
                        "Should have retried twice before success"
                    )
                    assert str(cid) == test_cid

                    # Test load with retries
                    timeout_count = 0
                    result = await kubo_cas.load(cid)
                    assert timeout_count == successful_after, (
                        "Should have retried twice before success"
                    )
                    assert result == test_data

                    # Test failure after max retries
                    async def failing_method(url, **kwargs):
                        dummy_request = httpx.Request(
                            "POST", url
                        )  # Create the dummy request
                        raise httpx.TimeoutException(
                            "Simulated timeout", request=dummy_request
                        )

                    with patch.object(
                        httpx.AsyncClient,
                        "post",
                        new=AsyncMock(side_effect=failing_method),
                    ):
                        with patch.object(
                            httpx.AsyncClient,
                            "get",
                            new=AsyncMock(side_effect=failing_method),
                        ):
                            with pytest.raises(
                                httpx.TimeoutException,
                                match="Failed to save data after 3 retries",
                            ):
                                await kubo_cas.save(test_data, codec="dag-cbor")

                            with pytest.raises(
                                httpx.TimeoutException,
                                match="Failed to load data after 3 retries",
                            ):
                                await kubo_cas.load(cid)


@pytest.mark.asyncio
async def test_kubo_backoff_timing():
    """
    Test that KuboCAS implements exponential backoff with jitter correctly.
    """

    async def timeout_method(url, **kwargs):
        # Manually create a dummy request for the exception
        dummy_request = httpx.Request("POST", url)
        raise httpx.TimeoutException("Simulated timeout", request=dummy_request)

    # Patch sleep to record timing
    original_sleep = asyncio.sleep
    sleep_times = []

    async def mock_sleep(delay):
        sleep_times.append(delay)
        # Call the original sleep function to avoid recursion
        await original_sleep(0)

    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(side_effect=timeout_method)
    ):
        async with httpx.AsyncClient() as client:
            async with KuboCAS(
                rpc_base_url="http://127.0.0.1:5001",
                gateway_base_url="http://127.0.0.1:8080",
                client=client,
                max_retries=3,
                initial_delay=0.1,
                backoff_factor=2.0,
            ) as kubo_cas:
                with patch("asyncio.sleep", side_effect=mock_sleep):
                    with pytest.raises(httpx.TimeoutException):
                        await kubo_cas.save(b"test", codec="dag-cbor")

                    # Verify backoff timing
                    assert len(sleep_times) == 3, "Should have attempted 3 retries"
                    assert 0.09 <= sleep_times[0] <= 0.11, "First retry should be ~0.1s"
                    assert 0.18 <= sleep_times[1] <= 0.22, (
                        "Second retry should be ~0.2s"
                    )
                    assert 0.36 <= sleep_times[2] <= 0.44, "Third retry should be ~0.4s"


@pytest.mark.asyncio
async def test_kubo_http_status_error_no_retry():
    """
    Tests that KuboCAS immediately raises HTTPStatusError without retrying.
    """

    # This mock simulates a server error by returning a 500 status code.
    async def mock_post_server_error(url, **kwargs):
        dummy_request = httpx.Request("POST", url)
        return httpx.Response(
            500, request=dummy_request, content=b"Internal Server Error"
        )

    # Patch the client's post method to always return the 500 error.
    with patch.object(
        httpx.AsyncClient, "post", new=AsyncMock(side_effect=mock_post_server_error)
    ):
        # Also patch asyncio.sleep to verify it's not called (i.e., no retries).
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            async with httpx.AsyncClient() as client:
                async with KuboCAS(client=client) as kubo_cas:
                    # Assert that the specific error is raised.
                    with pytest.raises(httpx.HTTPStatusError) as exc_info:
                        await kubo_cas.save(b"some data", codec="raw")

                    # Verify that the response in the exception has the correct status code.
                    assert exc_info.value.response.status_code == 500
                    # Verify that no retry was attempted.
                    mock_sleep.assert_not_called()


@pytest.mark.asyncio
async def test_kubo_cas_retry_validation():
    """Test validation of retry parameters in KuboCAS constructor"""

    # Test max_retries validation
    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        KuboCAS(max_retries=-1)

    with pytest.raises(ValueError, match="max_retries must be non-negative"):
        KuboCAS(max_retries=-5)

    # Test initial_delay validation
    with pytest.raises(ValueError, match="initial_delay must be positive"):
        KuboCAS(initial_delay=0)

    with pytest.raises(ValueError, match="initial_delay must be positive"):
        KuboCAS(initial_delay=-1.0)

    # Test backoff_factor validation
    with pytest.raises(
        ValueError, match="backoff_factor must be >= 1.0 for exponential backoff"
    ):
        KuboCAS(backoff_factor=0.5)

    with pytest.raises(
        ValueError, match="backoff_factor must be >= 1.0 for exponential backoff"
    ):
        KuboCAS(backoff_factor=0.9)

    # Test valid edge case values
    async with KuboCAS(
        max_retries=0, initial_delay=0.001, backoff_factor=1.0
    ) as kubo_cas:
        assert kubo_cas.max_retries == 0
        assert kubo_cas.initial_delay == 0.001
        assert kubo_cas.backoff_factor == 1.0
