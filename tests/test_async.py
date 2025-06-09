import asyncio

import pytest

from py_hamt import KuboCAS


@pytest.mark.asyncio
async def test_kubocas_cross_loop_close():
    """Test that KuboCAS handles closing when sessions exist in multiple loops."""

    # Create a KuboCAS instance
    cas = KuboCAS()

    # Create a session in the current loop
    _ = cas._loop_session()  # This creates a session for the current loop

    # Create a new event loop and session in a different thread
    other_loop_session = []

    def create_session_in_other_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _create():
            # Force creation of a session in this loop
            sess = cas._loop_session()
            other_loop_session.append((loop, sess))

        loop.run_until_complete(_create())
        # Don't close the loop yet - leave it pending

    import threading

    thread = threading.Thread(target=create_session_in_other_loop)
    thread.start()
    thread.join()

    # Now we have sessions in two different loops
    assert len(cas._session_per_loop) == 2

    # This should not raise an error
    try:
        await cas.aclose()
    except RuntimeError as e:
        if "attached to a different loop" in str(e):
            pytest.fail(f"Cross-loop close error: {e}")
        else:
            raise

    # Verify current loop's session was closed
    current_loop = asyncio.get_running_loop()
    assert (
        current_loop not in cas._session_per_loop
        or cas._session_per_loop[current_loop].closed
    )

    # The other loop's session should be removed from tracking
    assert len(cas._session_per_loop) == 0 or all(
        loop == current_loop for loop in cas._session_per_loop
    )


@pytest.mark.asyncio
async def test_kubocas_context_manager_with_fresh_loop():
    """Test the exact scenario from the user's script."""
    # This simulates what asyncio.run() does - creates a fresh loop

    async def use_kubocas():
        async with KuboCAS() as cas:
            # Simulate some work that creates a session
            _ = cas._loop_session()
            # The __aexit__ should handle cleanup properly

    # This should complete without errors
    # Using asyncio.run() creates a fresh event loop
    try:
        # We're already in an event loop from pytest, so we need to be careful
        # In real usage, this would be: asyncio.run(use_kubocas())
        await use_kubocas()
    except RuntimeError as e:
        if "attached to a different loop" in str(e):
            pytest.fail(f"Context manager close error: {e}")
        else:
            raise
