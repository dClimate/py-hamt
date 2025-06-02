# tests/test_kubocas_auth.py
import aiohttp
import pytest
import inspect
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
