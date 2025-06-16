# tests/test_kubocas_auth.py
import inspect

import httpx
import pytest

from py_hamt import KuboCAS


async def _maybe_await(x):
    return await x if inspect.isawaitable(x) else x


@pytest.mark.asyncio
async def test_user_supplied_client_headers():
    client = httpx.AsyncClient(headers={"Authorization": "Bearer x"})
    cas = KuboCAS(client=client)
    assert (await _maybe_await(cas._loop_client())).headers[
        "Authorization"
    ] == "Bearer x"
    await cas.aclose()  # must NOT close external client
    assert not client.is_closed
    await client.aclose()


@pytest.mark.asyncio
async def test_internal_client_headers() -> None:
    cas = KuboCAS(headers={"X-Custom": "yes"})
    client: httpx.AsyncClient = await _maybe_await(cas._loop_client())
    assert client.headers["X-Custom"] == "yes"
    await cas.aclose()
    assert client.is_closed


@pytest.mark.asyncio
async def test_user_supplied_client_auth() -> None:
    """
    If the user *provides* an httpx.AsyncClient that already carries
    authentication, KuboCAS must reuse that client unchanged and must
    **not** close it when `aclose()` is called.
    """
    auth = ("alice", "secret")
    external_client = httpx.AsyncClient(auth=auth)

    cas = KuboCAS(client=external_client)
    client = await _maybe_await(cas._loop_client())

    # The client KuboCAS returns is *exactly* the one we passed in,
    # and therefore must keep the same auth settings.
    assert client is external_client
    assert isinstance(client._auth, httpx.BasicAuth)

    # Calling aclose() must leave the externally-owned client open
    await cas.aclose()
    assert not external_client.is_closed
    await external_client.aclose()


@pytest.mark.asyncio
async def test_internal_client_auth() -> None:
    """
    When the caller passes `auth=` to the constructor (but no client),
    KuboCAS must create an AsyncClient that carries that auth setting,
    and should close it on `aclose()`.
    """
    auth = ("bob", "pa55w0rd")
    cas = KuboCAS(auth=auth)

    client = await _maybe_await(cas._loop_client())
    assert isinstance(client, httpx.AsyncClient)
    assert isinstance(client._auth, httpx.BasicAuth)

    # aclose() should shut down the internally-owned client
    await cas.aclose()
    assert client.is_closed
