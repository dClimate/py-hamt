from typing import Literal, cast

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
