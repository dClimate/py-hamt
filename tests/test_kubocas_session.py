import asyncio
import inspect
from threading import Event, Thread

import pytest

from py_hamt.store import KuboCAS


async def _maybe_await(value):
    """Helper that awaits *value* if it is awaitable, otherwise returns it as‑is."""
    if inspect.isawaitable(value):
        return await value
    return value


@pytest.mark.asyncio
async def test_get_session_same_instance_same_loop():
    """Two successive calls in the same event loop must hand back the *same*
    ClientSession instance and `aclose()` should close the session when the
    store owns it."""
    kubo = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001", gateway_base_url="http://127.0.0.1:8080"
    )

    sess1 = await _maybe_await(kubo._loop_session())
    sess2 = await _maybe_await(kubo._loop_session())

    assert sess1 is sess2  # no churn

    # the implementation is free to keep its internal mapping private.
    # We only guarantee behavioural semantics (same object, still open, closes via aclose).
    assert not sess1.closed

    await kubo.aclose()
    assert sess1.closed


@pytest.mark.asyncio
async def test_get_session_respects_user_supplied_session(global_client_session):
    """A pre‑created session passed to the constructor must always be reused
    and never closed by `aclose()`."""
    kubo = KuboCAS(
        session=global_client_session,
        rpc_base_url="http://127.0.0.1:5001",
        gateway_base_url="http://127.0.0.1:8080",
    )

    sess = await _maybe_await(kubo._loop_session())
    assert sess is global_client_session

    await kubo.aclose()  # should *not* close the external session
    assert not global_client_session.closed


async def _session_in_new_loop(kubo: KuboCAS):
    """Spawn a *separate* thread with its own event loop, obtain a session for
    that loop and hand both the session and the thread‑local loop object
    back to the caller."""
    ready = Event()
    container: list = []  # will receive (session, loop)

    def _worker():
        new_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(new_loop)

        async def _get():
            return await _maybe_await(kubo._loop_session())

        try:
            session = new_loop.run_until_complete(_get())
            container.append((session, new_loop))
        finally:
            ready.set()

    t = Thread(target=_worker, daemon=True)
    t.start()
    ready.wait()
    return container[0]


@pytest.mark.asyncio
async def test_distinct_loops_get_distinct_sessions():
    """Different event loops must receive distinct `ClientSession` objects and
    both must be tracked in `_session_per_loop`."""
    kubo = KuboCAS(
        rpc_base_url="http://127.0.0.1:5001", gateway_base_url="http://127.0.0.1:8080"
    )

    primary_session = await _maybe_await(kubo._loop_session())

    secondary_session, secondary_loop = await _session_in_new_loop(kubo)

    assert primary_session is not secondary_session
    assert len(kubo._session_per_loop) == 2
    assert kubo._session_per_loop[asyncio.get_running_loop()] is primary_session
    assert kubo._session_per_loop[secondary_loop] is secondary_session

    # Clean‑up
    await kubo.aclose()
    if not secondary_session.closed:
        await asyncio.to_thread(secondary_session.close)
