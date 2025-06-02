from typing import Literal, cast

import aiohttp
import dag_cbor
import pytest
from dag_cbor import IPLDKind
from hypothesis import given, settings
from testing_utils import create_ipfs, ipld_strategy  # noqa

from py_hamt import KuboCAS
from py_hamt.store import InMemoryCAS


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


@pytest.mark.asyncio(loop_scope="session")
@given(data=ipld_strategy())
@settings(
    deadline=1000, print_blob=True
)  # Increased deadline, print_blob for debugging
async def test_kubo_default_urls(
    global_client_session, data: IPLDKind
):  # Inject the session fixture
    """
    Tests KuboCAS using its default URLs and when None is passed for URLs,
    leveraging a globally managed aiohttp.ClientSession.
    """
    # Test Case 1: KuboCAS instantiated without explicit URLs (should use its defaults)
    # We pass the managed global_client_session to it.
    # KuboCAS itself is responsible for having default URLs if none are provided.
    async with KuboCAS(session=global_client_session) as kubo_cas_default:
        # print(f"Testing with default URLs: RPC={kubo_cas_default.rpc_base_url}, Gateway={kubo_cas_default.gateway_base_url}")
        encoded_data = dag_cbor.encode(data)

        for codec in ["raw", "dag-cbor"]:
            # The codec is a string, but we use Literal to ensure type safety
            # where codec_raw = "raw" and codec_dag_cbor = "dag-cbor"
            # necessary because when you iterate over a list of strings,
            # even if they are literal strings, mypy widens the type to just str
            codec_typed = cast(Literal["raw", "dag-cbor"], codec)
            # print(f"Saving with codec: {codec}, data: {data}")
            try:
                cid = await kubo_cas_default.save(encoded_data, codec=codec_typed)
                # print(f"Saved. CID: {cid}")
                loaded_encoded_data = await kubo_cas_default.load(cid)
                # print(f"Loaded encoded data length: {len(loaded_encoded_data)}")
                result = dag_cbor.decode(loaded_encoded_data)
                # print(f"Decoded result: {result}")
                assert data == result, (
                    f"Data mismatch for codec {codec} with default URLs"
                )
            except Exception as e:
                pytest.fail(
                    f"Error during KuboCAS default URL test (codec: {codec}): {e}"
                )

    # Test Case 2: KuboCAS instantiated with None for URLs (should also use its defaults)
    # We pass the managed global_client_session to it.
    async with KuboCAS(
        rpc_base_url=None, gateway_base_url=None, session=global_client_session
    ) as kubo_cas_none_urls:
        # print(f"Testing with None URLs: RPC={kubo_cas_none_urls.rpc_base_url}, Gateway={kubo_cas_none_urls.gateway_base_url}")
        encoded_data = dag_cbor.encode(
            data
        )  # Re-encode just in case, though it's the same data
        for codec in ["raw", "dag-cbor"]:
            # print(f"Saving with codec: {codec}, data: {data}")
            codec_typed = cast(Literal["raw", "dag-cbor"], codec)
            try:
                cid = await kubo_cas_none_urls.save(encoded_data, codec=codec_typed)
                # print(f"Saved. CID: {cid}")
                loaded_encoded_data = await kubo_cas_none_urls.load(cid)
                # print(f"Loaded encoded data length: {len(loaded_encoded_data)}")
                result = dag_cbor.decode(loaded_encoded_data)
                # print(f"Decoded result: {result}")
                assert data == result, f"Data mismatch for codec {codec} with None URLs"
            except Exception as e:
                pytest.fail(f"Error during KuboCAS None URL test (codec: {codec}): {e}")


# @given(data=ipld_strategy())
# @settings(deadline=1000)
# @pytest.mark.asyncio
# async def test_kubo_default_urls(data: IPLDKind):
#     try:
#         async with KuboCAS() as kubo_cas:
#             for codec in ("raw", "dag-cbor"):
#                 cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec)
#                 result = dag_cbor.decode(await kubo_cas.load(cid))
#                 assert data == result

#         async with KuboCAS(gateway_base_url=None, rpc_base_url=None) as kubo_cas:
#             for codec in ("raw", "dag-cbor"):
#                 cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec)
#                 result = dag_cbor.decode(await kubo_cas.load(cid))
#                 assert data == result
#     finally:
#         # if Hypothesis cancels early, make sure every open CAS is closed
#         for obj in list(globals().values()):
#             if isinstance(obj, KuboCAS):
#                 await obj.aclose()


@pytest.mark.asyncio
@given(data=ipld_strategy())
@settings(
    deadline=500
)  # this sometimes takes longer than the default 250 ms in GitHub CI
async def test_kubo_cas(create_ipfs, data: IPLDKind):  # noqa
    rpc_base_url, gateway_base_url = create_ipfs

    # Provide our own async Session, for complete code coverage
    async with aiohttp.ClientSession() as session:
        async with KuboCAS(
            rpc_base_url=rpc_base_url,
            gateway_base_url=gateway_base_url,
            session=session,
        ) as kubo_cas:
            # Use proper literal types for codec
            codec_raw: Literal["raw"] = "raw"
            codec_dag_cbor: Literal["dag-cbor"] = "dag-cbor"
            for codec in [codec_raw, codec_dag_cbor]:
                codec_typed = cast(Literal["raw", "dag-cbor"], codec)
                cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec_typed)
                result = dag_cbor.decode(await kubo_cas.load(cid))
                assert data == result
