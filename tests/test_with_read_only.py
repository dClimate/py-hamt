import pytest

from py_hamt import HAMT, InMemoryCAS, ZarrHAMTStore


@pytest.mark.asyncio
async def test_with_read_only_roundtrip():
    cas = InMemoryCAS()
    hamt_rw = await HAMT.build(cas=cas, values_are_bytes=True)
    store_rw = ZarrHAMTStore(hamt_rw, read_only=False)

    # clone → RO
    store_ro = store_rw.with_read_only(True)
    assert store_ro.read_only is True
    assert store_ro is not store_rw
    # clone back → RW
    store_rw2 = store_ro.with_read_only(False)
    assert store_rw2.read_only is False
    assert store_rw2.hamt.root_node_id == store_rw.hamt.root_node_id
