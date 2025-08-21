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


@pytest.mark.asyncio
async def test_roundtrip_plain_store():
    rw = await _rw_plain()  # writable store
    ro = rw.with_read_only(True)  # clone → RO
    assert ro.read_only is True
    assert ro.hamt is rw.hamt

    # idempotent: RO→RO returns same object
    assert ro.with_read_only(True) is ro

    # back to RW (new wrapper)
    rw2 = ro.with_read_only(False)
    assert rw2.read_only is False and rw2 is not ro
    assert rw2.hamt is rw.hamt

    # guard: cannot write through RO wrapper
    with pytest.raises(Exception):
        await ro.set("k", np.array([0], dtype="u1"))


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
