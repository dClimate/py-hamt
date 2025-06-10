# tests/test_async.py
import asyncio
import threading

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
    other_loop_container = []
    other_session_container = []

    def create_session_in_other_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def _create():
            # Force creation of a session in this loop
            sess = cas._loop_session()
            other_session_container.append(sess)
            other_loop_container.append(loop)

        loop.run_until_complete(_create())
        # Important: Don't close the loop yet - we want to test the multi-loop scenario
        # But we do need to clean it up later

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

    # After aclose, the dictionary should be empty or only contain closed sessions
    # from the current loop (the implementation may choose either approach)
    assert len(cas._session_per_loop) == 0 or all(
        loop == asyncio.get_running_loop() for loop in cas._session_per_loop
    )

    # Clean up the other event loop to prevent warnings
    if other_loop_container:
        other_loop = other_loop_container[0]
        if other_session_container and not other_session_container[0].closed:
            # Schedule the session close in its own loop
            def cleanup():
                async def _close():
                    if not other_session_container[0].closed:
                        await other_session_container[0].close()

                other_loop.run_until_complete(_close())
                other_loop.close()

            cleanup_thread = threading.Thread(target=cleanup)
            cleanup_thread.start()
            cleanup_thread.join()


@pytest.mark.asyncio
async def test_kubocas_context_manager_with_fresh_loop():
    """Test the exact scenario from the user's script."""

    async def use_kubocas():
        async with KuboCAS() as cas:
            # Simulate some work that creates a session
            _ = cas._loop_session()
            # The __aexit__ should handle cleanup properly

    # This should complete without errors
    try:
        await use_kubocas()
    except RuntimeError as e:
        if "attached to a different loop" in str(e):
            pytest.fail(f"Context manager close error: {e}")
        else:
            raise


@pytest.mark.asyncio
async def test_kubocas_no_running_loop_in_aclose():
    """Test aclose behavior when called without a running event loop."""
    cas = KuboCAS()

    # Create a session in the current loop
    _ = cas._loop_session()

    # Simulate calling aclose when there's no event loop
    # We'll mock this by calling the method directly
    import unittest.mock

    with unittest.mock.patch(
        "asyncio.get_running_loop", side_effect=RuntimeError("No running loop")
    ):
        await cas.aclose()

    # The session references should be cleared
    assert len(cas._session_per_loop) == 0


@pytest.mark.asyncio
async def test_kubocas_session_already_closed():
    """Test aclose behavior when session is already closed."""
    cas = KuboCAS()

    # Create and immediately close a session
    sess = cas._loop_session()
    await sess.close()

    # aclose should handle this gracefully
    await cas.aclose()

    # Verify cleanup
    assert len(cas._session_per_loop) == 0
