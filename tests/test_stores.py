import aiohttp
import dag_cbor
from dag_cbor import IPLDKind
from hypothesis import given
import pytest

from py_hamt import IPFSStore

from py_hamt.store import DictStore
from testing_utils import ipld_strategy, create_ipfs  # noqa


# Just to cover this one case that isn't covered within test_hamt
@pytest.mark.asyncio
async def test_memory_store_exception():
    s = DictStore()
    with pytest.raises(KeyError):
        await s.load(bytes())


@pytest.mark.asyncio
@given(data=ipld_strategy())
async def test_ipfsstore_default_urls(data: IPLDKind):
    ipfsstore = IPFSStore()
    # test that ipfs store close session when both are already missing is a no-op
    await ipfsstore.close_sessions()
    cids = []
    try:
        for codec in ["raw", "dag-cbor"]:
            cid = await ipfsstore.save(dag_cbor.encode(data), codec=codec)  # type: ignore
            cids.append(cid)
            result = dag_cbor.decode(await ipfsstore.load(cid))
            assert data == result
            # We should be able to close right after every result and it will remake its sessions if it needs to
            await ipfsstore.close_sessions()
    finally:
        await ipfsstore.close_sessions()
        # test that it's a no-op again
        await ipfsstore.close_sessions()

    # Now do it but with only loading data, which will be a common use pattern for HAMT readers
    try:
        for cid in cids:
            for codec in ["raw", "dag-cbor"]:
                result = dag_cbor.decode(await ipfsstore.load(cid))
                assert data == result
                await ipfsstore.close_sessions()
    finally:
        await ipfsstore.close_sessions()
        await ipfsstore.close_sessions()


@pytest.mark.asyncio
@given(data=ipld_strategy())
async def test_ipfsstore(create_ipfs, data: IPLDKind):  # noqa
    rpc_uri_stem, gateway_uri_stem = create_ipfs
    gateway_session = aiohttp.ClientSession(base_url=gateway_uri_stem)
    rpc_session = aiohttp.ClientSession(base_url=rpc_uri_stem)

    try:
        ipfsstore = IPFSStore(rpc_session=rpc_session, gateway_session=gateway_session)

        for codec in ["raw", "dag-cbor"]:
            cid = await ipfsstore.save(dag_cbor.encode(data), codec=codec)  # type: ignore
            result = dag_cbor.decode(await ipfsstore.load(cid))
            assert data == result
    finally:
        await rpc_session.close()
        await gateway_session.close()
