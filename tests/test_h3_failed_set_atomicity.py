import pytest

from py_hamt import HAMT, InMemoryCAS


def one_byte_colliding_hash(_: bytes) -> bytes:
    return b"\x00"


@pytest.mark.asyncio
async def test_failed_set_preserves_previously_committed_key() -> None:
    hamt = await HAMT.build(
        cas=InMemoryCAS(),
        hash_fn=one_byte_colliding_hash,
        max_bucket_size=1,
    )

    await hamt.set("a", b"value-a")
    assert await hamt.get("a") == b"value-a"

    with pytest.raises(IndexError):
        await hamt.set("b", b"value-b")

    assert await hamt.get("a") == b"value-a"
