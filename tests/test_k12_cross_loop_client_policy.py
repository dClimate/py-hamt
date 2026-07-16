import asyncio
import warnings

import httpx
import pytest

from py_hamt import KuboCAS


def _client_for_current_loop(cas: KuboCAS) -> httpx.AsyncClient:
    async def get_client() -> httpx.AsyncClient:
        return cas._loop_client()

    return asyncio.run(get_client())


def _close_clients(cas: KuboCAS, *clients: httpx.AsyncClient) -> None:
    async def close_clients() -> None:
        for client in {*cas._client_per_loop.values(), *clients}:
            if not client.is_closed:
                await client.aclose()

    asyncio.run(close_clients())


def test_client_factory_builds_per_loop_clients() -> None:
    factory_calls = 0
    created_clients: list[httpx.AsyncClient] = []

    def client_factory() -> httpx.AsyncClient:
        nonlocal factory_calls
        factory_calls += 1
        client = httpx.AsyncClient(headers={"X-Marker": "factory"})
        created_clients.append(client)
        return client

    cas = KuboCAS(client_factory=client_factory)

    try:
        first_client = _client_for_current_loop(cas)
        second_client = _client_for_current_loop(cas)

        assert factory_calls == 2
        assert first_client.headers["X-Marker"] == "factory"
        assert second_client.headers["X-Marker"] == "factory"

        asyncio.run(cas.aclose())

        assert all(client.is_closed for client in created_clients)
    finally:
        _close_clients(cas, *created_clients)


def test_client_and_factory_are_mutually_exclusive() -> None:
    supplied_client = httpx.AsyncClient()

    try:
        with pytest.raises(ValueError):
            KuboCAS(
                client=supplied_client,
                client_factory=lambda: httpx.AsyncClient(),
            )
    finally:
        asyncio.run(supplied_client.aclose())


def test_second_loop_fallback_warns() -> None:
    supplied_client = httpx.AsyncClient()
    cas = KuboCAS(client=supplied_client)

    try:
        assert _client_for_current_loop(cas) is supplied_client

        with pytest.warns(
            RuntimeWarning,
            match="cannot be reused across event loops",
        ):
            _client_for_current_loop(cas)
    finally:
        _close_clients(cas, supplied_client)


def test_second_loop_fallback_preserves_redirect_policy() -> None:
    supplied_client = httpx.AsyncClient(follow_redirects=False)
    cas = KuboCAS(client=supplied_client)

    try:
        assert _client_for_current_loop(cas) is supplied_client

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            fallback_client = _client_for_current_loop(cas)

        assert fallback_client.follow_redirects is False
    finally:
        _close_clients(cas, supplied_client)
