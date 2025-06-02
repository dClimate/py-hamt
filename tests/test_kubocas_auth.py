# tests/test_kubocas_auth.py
import inspect

import aiohttp
import pytest

from py_hamt.store import KuboCAS


async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


@pytest.mark.asyncio
async def test_user_supplied_session_headers():
    sess = aiohttp.ClientSession(headers={"Authorization": "Bearer x"})
    cas = KuboCAS(session=sess)
    assert (await _maybe_await(cas._loop_session())).headers[
        "Authorization"
    ] == "Bearer x"
    await cas.aclose()  # must NOT close external session
    assert not sess.closed
    await sess.close()


@pytest.mark.asyncio
async def test_internal_session_headers() -> None:
    cas = KuboCAS(headers={"X-Custom": "yes"})
    sess: aiohttp.ClientSession = await _maybe_await(cas._loop_session())
    assert sess.headers["X-Custom"] == "yes"
    await cas.aclose()
    assert sess.closed


@pytest.mark.asyncio
async def test_user_supplied_session_auth() -> None:
    """
    If the user *provides* an aiohttp.ClientSession that already carries
    authentication, KuboCAS must reuse that session unchanged and must
    **not** close it when `aclose()` is called.
    """
    basic = aiohttp.BasicAuth("alice", "secret")
    external_session = aiohttp.ClientSession(auth=basic)

    cas = KuboCAS(session=external_session)
    sess = await _maybe_await(cas._loop_session())

    # The session KuboCAS returns is *exactly* the one we passed in,
    # and therefore must keep the same BasicAuth object.
    assert sess is external_session
    assert sess.auth == basic

    # Calling aclose() must leave the externally-owned session open
    await cas.aclose()
    assert not external_session.closed
    await external_session.close()


@pytest.mark.asyncio
async def test_internal_session_auth() -> None:
    """
    When the caller passes `auth=` to the constructor (but no session),
    KuboCAS must create a ClientSession that carries that auth object,
    and should close it on `aclose()`.
    """
    basic = aiohttp.BasicAuth("bob", "pa55w0rd")
    cas = KuboCAS(auth=basic)

    sess = await _maybe_await(cas._loop_session())
    assert isinstance(sess, aiohttp.ClientSession)
    assert sess.auth == basic

    # aclose() should shut down the internally-owned session
    await cas.aclose()
    assert sess.closed
