import asyncio
import inspect
import unittest
from threading import Event, Thread

import httpx
import pytest

from py_hamt import KuboCAS


async def _maybe_await(value):
    """Helper that awaits *value* if it is awaitable, otherwise returns it as‑is."""
    if inspect.isawaitable(value):
        return await value
    return value


@pytest.mark.asyncio
async def test_get_client_same_instance_same_loop():
    """Two successive calls in the same event loop must hand back the *same*
    AsyncClient instance and `aclose()` should close the client when the
    store owns it."""
    kubo = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001", gateway_base_url="http://127.0.0.1:8080"
    )

    client1 = await _maybe_await(kubo._loop_client())
    client2 = await _maybe_await(kubo._loop_client())

    assert client1 is client2  # no churn

    # the implementation is free to keep its internal mapping private.
    # We only guarantee behavioural semantics (same object, still open, closes via aclose).
    assert not client1.is_closed

    await kubo.aclose()
    assert client1.is_closed


@pytest.mark.asyncio
async def test_get_client_respects_user_supplied_client(global_client_session):
    """A pre‑created client passed to the constructor must always be reused
    and never closed by `aclose()`."""
    kubo = KuboCAS(
        client=global_client_session,
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
    )

    client = await _maybe_await(kubo._loop_client())
    assert client is global_client_session

    await kubo.aclose()  # should *not* close the external client
    assert not global_client_session.is_closed


async def _client_in_new_loop(kubo: KuboCAS):
    """Spawn a *separate* thread with its own event loop, obtain a client for
    that loop and hand both the client and the thread‑local loop object
    back to the caller."""
    ready = Event()
    container: list = []  # will receive (client, loop)

    def _worker():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        async def _get():
            return await _maybe_await(kubo._loop_client())

        try:
            client = new_loop.run_until_complete(_get())
            container.append((client, new_loop))
        finally:
            ready.set()

    t = Thread(target=_worker, daemon=True)
    t.start()
    ready.wait()
    return container[0]


@pytest.mark.asyncio
async def test_distinct_loops_get_distinct_clients():
    """Different event loops must receive distinct `AsyncClient` objects and
    both must be tracked in `_client_per_loop`."""
    kubo = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001", gateway_base_url="http://127.0.0.1:8080"
    )

    primary_client = await _maybe_await(kubo._loop_client())

    secondary_client, secondary_loop = await _client_in_new_loop(kubo)

    assert primary_client is not secondary_client
    assert len(kubo._client_per_loop) == 2
    assert kubo._client_per_loop[asyncio.get_running_loop()] is primary_client
    assert kubo._client_per_loop[secondary_loop] is secondary_client

    # Clean‑up
    await kubo.aclose()
    assert secondary_client.is_closed


@pytest.mark.asyncio
async def test_del_closes_client():
    """`KuboCAS` should close clients when the instance is garbage collected."""
    kubo = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
    )

    client = await _maybe_await(kubo._loop_client())
    assert not client.is_closed

    # Drop the reference and force garbage collection
    del kubo
    import gc

    gc.collect()
    await asyncio.sleep(0)

    assert client.is_closed


# --------------------------------------------------------------------------- #
# 1.  Early‑return guard – instance missing internal sentinel attributes      #
# --------------------------------------------------------------------------- #
def test_del_missing_internal_attributes(monkeypatch):
    """
    If either ``_owns_client`` or ``_closed`` is absent, __del__ must bail out
    immediately.  We remove one attribute and assert that nothing blows up.
    """
    cas = KuboCAS()  # fully‑initialised object
    del cas._owns_client  # simulate a partially‑constructed instance

    # __del__ should *just return* – no exceptions, no side effects
    cas.__del__()  # noqa:  B023  (explicit dunder call is deliberate)


# --------------------------------------------------------------------------- #
# 2.  Loop present *but* not running  →  asyncio.run(...) branch (317‑322)    #
# --------------------------------------------------------------------------- #
def test_del_loop_not_running_branch(monkeypatch):
    """
    Force __del__ down the branch where an event loop *exists* but is *not*
    running, then make ``asyncio.run()`` raise so the error‑handling block
    is executed as well (two birds, one stone).
    """
    cas = KuboCAS()

    # ------------------------------------------------------------------ #
    # 2a.  Fake "current loop" object whose ``is_running()`` is *False*  #
    # ------------------------------------------------------------------ #
    dummy_loop = unittest.mock.Mock(is_running=lambda: False)

    # Patch *this* thread’s "running loop" to our dummy object
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: dummy_loop)

    # ------------------------------------------------------------------ #
    # 2b.  Make ``asyncio.run()`` raise – triggers the except: block      #
    # ------------------------------------------------------------------ #
    run_called = {"flag": False}

    def fake_run(coro):
        run_called["flag"] = True
        raise RuntimeError("simulated failure inside asyncio.run")

    monkeypatch.setattr(asyncio, "run", fake_run)

    # Inject a *placeholder* client so the clean‑up code has something
    # to clear – avoids importing httpx in sync context.
    cas._client_per_loop[dummy_loop] = object()

    # Preconditions
    assert cas._client_per_loop and not cas._closed

    # -- fire! -----------------------------------------------------------------
    cas.__del__()  # noqa:  B023

    # ------------------------------------------------------------------ #
    # 2c.  Post‑conditions:                                              #
    #      • asyncio.run() was attempted                                 #
    #      • the except‑branch cleared the client cache and marked closed#
    # ------------------------------------------------------------------ #
    assert run_called["flag"] is True
    assert cas._closed is True
    assert len(cas._client_per_loop) == 0


@pytest.mark.asyncio
async def test_loop_client_reopens_after_close():
    """Calling _loop_client() after aclose() recreates a fresh client."""
    cas = KuboCAS()

    first = cas._loop_client()
    await cas.aclose()

    # Should no longer raise; instead a new client is created.
    reopened = cas._loop_client()
    assert isinstance(reopened, httpx.AsyncClient)
    assert reopened is not first
    assert cas._closed is False

    await cas.aclose()


@pytest.mark.asyncio
async def test_loop_client_rejects_reuse_of_external_client(global_client_session):
    """Calling _loop_client() after aclose() raises when client is user-supplied."""
    cas = KuboCAS(
        client=global_client_session,
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
    )
    assert cas._loop_client() is global_client_session

    await cas.aclose()
    cas._closed = True  # simulate closed instance with external client
    with pytest.raises(RuntimeError, match="KuboCAS is closed; create a new instance"):
        cas._loop_client()


def _raise_no_loop():
    """Helper to simulate no running event loop."""
    raise RuntimeError("No running event loop")


@pytest.mark.asyncio
async def test_aclose_sync_path(monkeypatch):
    """
    aclose() must fall back to .close() when no loop is running.
    """
    cas = KuboCAS()
    client = cas._loop_client()  # create internal client

    # Pretend we are in synchronous context
    monkeypatch.setattr(asyncio, "get_running_loop", _raise_no_loop)

    await cas.aclose()  # should *not* raise
    assert client.is_closed
    assert cas._closed
    assert not cas._client_per_loop  # map cleared


# @pytest.mark.asyncio
# async def test_del_from_sync_context(monkeypatch):
#     """
#     __del__ must clean up synchronously when called from a context
#     with no running event loop.
#     """
#     # Arrange: Create a CAS and its client inside the test's event loop.
#     cas = KuboCAS()
#     # _loop_client is a sync method, but it needs a running loop to work.
#     # This test provides one via the @pytest.mark.asyncio decorator.
#     client = cas._loop_client()

#     # Pre-condition: The client is now open.
#     assert not client.is_closed

#     # Act:
#     # 1. Now, patch asyncio to simulate what happens when __del__ is
#     #    called from a truly synchronous context (where no loop is available).
#     monkeypatch.setattr(asyncio, "get_running_loop", _raise_no_loop)
#     # 2. Trigger the destructor's logic directly.
#     cas.__del__()

#     # Assert: The destructor should have caught the RuntimeError and
#     # performed the synchronous cleanup.
#     assert client.is_closed
#     assert cas._closed
#     assert not cas._client_per_loop


# def test_del_sync_cleanup(monkeypatch):
#     cas = KuboCAS()
#     dummy = httpx.AsyncClient()
#     cas._client_per_loop[object()] = dummy  # inject

#     monkeypatch.setattr(asyncio, "get_running_loop", _raise_no_loop)

#     cas.__del__()  # noqa: B023
#     assert dummy.is_closed
