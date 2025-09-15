import dag_cbor
import pytest

from py_hamt import KuboCAS


@pytest.mark.asyncio(loop_scope="session")
async def test_pinning(create_ipfs, global_client_session):
    """
    Tests pinning a CID using KuboCAS with explicit URLs.
    Verifies that a CID can be pinned and is retrievable after pinning.
    """
    rpc_url, gateway_url = create_ipfs

    async with KuboCAS(
        rpc_base_url=rpc_url,
        gateway_base_url=gateway_url,
        client=global_client_session,
    ) as kubo_cas:
        # Save data to get a CID
        data = b"test1"
        encoded_data = dag_cbor.encode(data)
        cid = await kubo_cas.save(encoded_data, codec="dag-cbor")
        # Pin the CID
        await kubo_cas.pin_cid(cid, target_rpc=rpc_url)
        listed_pins = await kubo_cas.pin_ls(target_rpc=rpc_url)
        # Verify the CID is pinned by querying the pin list
        assert str(cid) in listed_pins, f"CID {cid} was not pinned"

        # Unpine the CID
        await kubo_cas.unpin_cid(cid, target_rpc=rpc_url)

        # Verify the CID is no longer pinned
        listed_pins_after_unpin = await kubo_cas.pin_ls(target_rpc=rpc_url)
        assert str(cid) not in listed_pins_after_unpin, f"CID {cid} was not unpinned"

        # Pin again, then perform a pin update
        await kubo_cas.pin_cid(cid, target_rpc=rpc_url)

        data = b"test2"
        encoded_data = dag_cbor.encode(data)
        new_cid = await kubo_cas.save(encoded_data, codec="dag-cbor")

        # Update the pinned CID
        await kubo_cas.pin_update(cid, new_cid, target_rpc=rpc_url)

        # Verify the old CID is no longer pinned and the new CID is pinned
        listed_pins_after_update = await kubo_cas.pin_ls(target_rpc=rpc_url)
        assert str(cid) not in listed_pins_after_update, (
            f"Old CID {cid} was not unpinned after update"
        )
        assert str(new_cid) in listed_pins_after_update, (
            f"New CID {new_cid} was not pinned after update"
        )

        # unpin the new CID
        await kubo_cas.unpin_cid(new_cid, target_rpc=rpc_url)
        # Verify the new CID is no longer pinned
        listed_pins_after_unpin_update = await kubo_cas.pin_ls(target_rpc=rpc_url)
        assert str(new_cid) not in listed_pins_after_unpin_update, (
            f"New CID {new_cid} was not unpinned after update"
        )
