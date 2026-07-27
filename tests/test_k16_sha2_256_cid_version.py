"""Regression tests for issue #43.

Kubo returns a CIDv0 for any ``/api/v0/add`` that does not explicitly request
version 1, and a CIDv0 is ``dag-pb`` by definition. With a hasher that CIDv0 can
represent -- ``sha2-256`` -- the daemon therefore wrapped the payload in a UnixFS
``dag-pb`` node and returned ``Qm...`` no matter which codec ``save()`` asked
for. Two consequences, both covered here:

* ``save()`` deliberately declines to relabel a ``dag-pb`` CID's codec, because
  the block Kubo stored is the protobuf wrapper rather than the bytes passed in.
  A relabelled CID would not hash to its own block.
* ``_cid_is_verifiable()`` declines to check ``dag-pb`` responses, so a
  ``verify_content=True`` store built on ``sha2-256`` silently returned every
  block unverified.

The fix pins ``cid-version=1`` on the add URL. ``blake3`` -- the default, and not
CIDv0-representable -- was never affected, so it is asserted alongside as a
control.
"""

import httpx
import pytest
from multiformats import CID, multihash

from py_hamt import HAMT, KuboCAS
from py_hamt.store_httpx import _cid_is_verifiable


def test_add_url_requests_cid_version_1() -> None:
    """The add URL must pin cid-version=1 for every hasher."""
    cas = KuboCAS(hasher="sha2-256")

    assert "cid-version=1" in cas.rpc_url


@pytest.mark.parametrize("hasher", ["sha2-256", "blake3"])
def test_add_url_pins_cid_version_for_any_hasher(hasher: str) -> None:
    cas = KuboCAS(hasher=hasher)

    assert "cid-version=1" in cas.rpc_url
    assert f"hash={hasher}" in cas.rpc_url


@pytest.mark.ipfs
@pytest.mark.parametrize(
    ("hasher", "codec"),
    [
        ("sha2-256", "raw"),
        ("sha2-256", "dag-cbor"),
        ("blake3", "raw"),
        ("blake3", "dag-cbor"),
    ],
)
async def test_save_returns_cidv1_with_requested_codec(
    create_ipfs: tuple[str, str], hasher: str, codec: str
) -> None:
    """sha2-256 must yield a CIDv1 carrying the codec save() was asked for."""
    rpc, gw = create_ipfs
    payload = b"py-hamt issue 43: " + hasher.encode() + b"/" + codec.encode()

    async with KuboCAS(
        hasher=hasher, rpc_base_url=rpc, gateway_base_url=gw
    ) as cas:
        cid = await cas.save(payload, codec=codec)

        assert cid.version == 1, f"{hasher} still returns a CIDv0"
        assert cid.codec.name == codec
        assert cid.codec.code != KuboCAS.DAG_PB_MARKER


@pytest.mark.ipfs
@pytest.mark.parametrize("hasher", ["sha2-256", "blake3"])
async def test_saved_cid_digest_matches_stored_block(
    create_ipfs: tuple[str, str], hasher: str
) -> None:
    """The returned CID must hash the bytes passed in, not a dag-pb wrapper.

    This is what a CIDv0 broke: Kubo stored ``0a24 0802 121e <payload> 181e``
    (the UnixFS protobuf) rather than the payload, so the digest committed to by
    the CID was the wrapper's.
    """
    rpc, gw = create_ipfs
    payload = b"issue 43 digest check for " + hasher.encode()

    async with KuboCAS(
        hasher=hasher, rpc_base_url=rpc, gateway_base_url=gw
    ) as cas:
        cid = await cas.save(payload, codec="raw")

    raw_digest = bytes(cid.raw_digest)
    computed = multihash.digest(payload, cid.hashfun.name, size=len(raw_digest))

    assert bytes(multihash.unwrap(computed)) == raw_digest


@pytest.mark.ipfs
@pytest.mark.parametrize("hasher", ["sha2-256", "blake3"])
async def test_kubo_stores_payload_verbatim_under_returned_cid(
    create_ipfs: tuple[str, str], hasher: str
) -> None:
    """Kubo's own block store must hold the payload, unwrapped, at that CID.

    Asserted against the daemon rather than the client so a future regression in
    the add parameters is caught even if py-hamt's own bookkeeping stays
    self-consistent.
    """
    rpc, gw = create_ipfs
    payload = b"issue 43 block-store check for " + hasher.encode()

    async with KuboCAS(
        hasher=hasher, rpc_base_url=rpc, gateway_base_url=gw
    ) as cas:
        cid = await cas.save(payload, codec="raw")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{rpc}/api/v0/block/get?arg={cid}")
        response.raise_for_status()

    assert response.content == payload


@pytest.mark.ipfs
@pytest.mark.parametrize("hasher", ["sha2-256", "blake3"])
async def test_saved_cid_is_verifiable(
    create_ipfs: tuple[str, str], hasher: str
) -> None:
    """verify_content must actually verify, not fall through on dag-pb."""
    rpc, gw = create_ipfs
    payload = b"issue 43 verifiability for " + hasher.encode()

    async with KuboCAS(
        hasher=hasher, rpc_base_url=rpc, gateway_base_url=gw, verify_content=True
    ) as cas:
        cid = await cas.save(payload, codec="raw")

        assert _cid_is_verifiable(cid, None, None), (
            f"{hasher} blocks are unverifiable, so verify_content is a no-op"
        )
        assert await cas.load(cid) == payload


@pytest.mark.ipfs
async def test_hamt_roundtrips_under_sha2_256(create_ipfs: tuple[str, str]) -> None:
    """A whole HAMT must build and read back on sha2-256, verification on."""
    rpc, gw = create_ipfs

    async with KuboCAS(
        hasher="sha2-256", rpc_base_url=rpc, gateway_base_url=gw, verify_content=True
    ) as cas:
        hamt = await HAMT.build(cas=cas)
        for index in range(64):
            await hamt.set(f"key-{index}", index)
        await hamt.make_read_only()
        root = hamt.root_node_id

        assert isinstance(root, CID)
        assert root.version == 1
        assert root.codec.code != KuboCAS.DAG_PB_MARKER

        read = await HAMT.build(cas=cas, root_node_id=root, read_only=True)
        assert await read.len() == 64
        for index in range(64):
            assert await read.get(f"key-{index}") == index
