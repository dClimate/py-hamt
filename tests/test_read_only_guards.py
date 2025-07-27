# tests/test_read_only_guards.py
import numpy as np
import pytest
from Crypto.Random import get_random_bytes

from py_hamt import HAMT, InMemoryCAS, SimpleEncryptedZarrHAMTStore, ZarrHAMTStore


# ---------- helpers ----------------------------------------------------
async def _rw_plain():
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    return ZarrHAMTStore(hamt, read_only=False)


async def _rw_enc():
    cas = InMemoryCAS()
    hamt = await HAMT.build(cas=cas, values_are_bytes=True)
    key, hdr = get_random_bytes(32), b"hdr"
    return SimpleEncryptedZarrHAMTStore(hamt, False, key, hdr)


# ---------- plain store ------------------------------------------------
@pytest.mark.asyncio
async def test_plain_read_only_guards():
    rw = await _rw_plain()
    ro = rw.with_read_only(True)

    assert ro.read_only is True
    with pytest.raises(Exception):
        await ro.set("k", np.array([1], dtype="u1"))
    with pytest.raises(Exception):
        await ro.delete("k")


@pytest.mark.asyncio
async def test_plain_with_same_flag_returns_self():
    rw = await _rw_plain()
    assert rw.with_read_only(False) is rw  # early‑return path


# ---------- encrypted store -------------------------------------------
@pytest.mark.asyncio
async def test_encrypted_read_only_guards_and_self():
    rw = await _rw_enc()
    assert rw.with_read_only(False) is rw  # same‑flag path
    ro = rw.with_read_only(True)
    with pytest.raises(Exception):
        await ro.set("k", np.array([2], dtype="u1"))
    with pytest.raises(Exception):
        await ro.delete("k")
