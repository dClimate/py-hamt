import aiohttp
import dag_cbor
from dag_cbor import IPLDKind
from hypothesis import given
import pytest

from py_hamt import IPFSStore

from testing_utils import ipld_strategy, create_ipfs  # noqa


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
