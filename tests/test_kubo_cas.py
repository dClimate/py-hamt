import dag_cbor
from dag_cbor import IPLDKind
from hypothesis import given, settings
import pytest
import requests

from py_hamt import KuboCAS

from py_hamt.store import InMemoryCAS
from testing_utils import ipld_strategy, create_ipfs  # noqa


# Just to cover this one case that isn't covered within test_hamt
@pytest.mark.asyncio
async def test_memory_store_exception():
    s = InMemoryCAS()
    with pytest.raises(KeyError):
        await s.load(bytes())


@pytest.mark.asyncio
@given(data=ipld_strategy())
@settings(deadline=500)  # this sometimes takes longer than the default 250 ms
async def test_kubo_default_urls(data: IPLDKind):
    kubo_cas = KuboCAS()
    cids = []
    for codec in ["raw", "dag-cbor"]:
        cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec)  # type: ignore
        cids.append(cid)
        result = dag_cbor.decode(await kubo_cas.load(cid))
        assert data == result

    kubo_cas = KuboCAS(gateway_base_url=None, rpc_base_url=None)
    cids = []
    for codec in ["raw", "dag-cbor"]:
        cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec)  # type: ignore
        cids.append(cid)
        result = dag_cbor.decode(await kubo_cas.load(cid))
        assert data == result


@pytest.mark.asyncio
@given(data=ipld_strategy())
@settings(
    deadline=500
)  # this sometimes takes longer than the default 250 ms in GitHub CI
async def test_kubo_cas(create_ipfs, data: IPLDKind):  # noqa
    rpc_base_url, gateway_base_url = create_ipfs

    # Provide our own requests Session, for complete code coverage
    kubo_cas = KuboCAS(
        rpc_base_url=rpc_base_url,
        gateway_base_url=gateway_base_url,
        requests_session=requests.Session(),
    )

    for codec in ["raw", "dag-cbor"]:
        cid = await kubo_cas.save(dag_cbor.encode(data), codec=codec)  # type: ignore
        result = dag_cbor.decode(await kubo_cas.load(cid))
        assert data == result
