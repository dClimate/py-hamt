from typing import Optional

import pytest
from dag_cbor.ipld import IPLDKind

from py_hamt import HAMT, InMemoryCAS


class FlakyCAS(InMemoryCAS):
    def __init__(self) -> None:
        super().__init__()
        self.fail = False

    async def load(
        self,
        id: IPLDKind,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        suffix: Optional[int] = None,
    ) -> bytes:
        if self.fail:
            raise ConnectionError("simulated CAS load failure")
        return await super().load(id, offset=offset, length=length, suffix=suffix)


@pytest.mark.asyncio
async def test_failed_delete_preserves_all_keys() -> None:
    cas = FlakyCAS()
    hamt = await HAMT.build(cas=cas)
    keys = ["1313", "5428", "11835", "17462", "20814", "s45"]

    for key in keys:
        await hamt.set(key, key.encode())

    await hamt.cache_vacate()
    assert await hamt.get("s45") == b"s45"

    cas.fail = True
    with pytest.raises(ConnectionError):
        await hamt.delete("s45")
    cas.fail = False

    for key in keys:
        assert await hamt.get(key) == key.encode()
