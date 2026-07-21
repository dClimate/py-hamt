from typing import Any

import pytest
from blake3 import blake3
from multiformats import multihash

from py_hamt.hamt import blake3_hashfn


@pytest.mark.parametrize(
    ("input_bytes", "expected_hex_digest"),
    [
        (b"", "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"),
        (b"\x00", "2d3adedff11b61f14c886e35afa036736dcd87a74d27b5c1510225d0f592e213"),
        (
            b"hello world",
            "d74981efa70a0c880b8d8c1985d075dbcbf679b99a5f9914e5aaf96b831a9e24",
        ),
        (
            "héllo ünïcöde".encode(),
            "90ede513eb21f2db0fc88dc008be5e07d35304d4628624c614abc5a25991d5f3",
        ),
        (
            bytes(range(256)) * 4096,
            "64479cf7293960210547db8d982359e0c4ce054525ed7086cf93030828fc0533",
        ),
    ],
    ids=["empty", "one-byte", "ascii", "unicode", "one-megabyte-pattern"],
)
def test_blake3_hashfn_preserves_golden_digest(
    input_bytes: bytes, expected_hex_digest: str
) -> None:
    digest = blake3_hashfn(input_bytes)

    assert len(digest) == 32
    assert digest.hex() == expected_hex_digest


def test_blake3_hashfn_bypasses_multiformats_wrappers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = b"structural probe: no multiformats in hot path"
    multiformats_call_count = 0
    original_digest = multihash.Multihash.digest
    original_unwrap = multihash.Multihash.unwrap

    def digest_spy(self: multihash.Multihash, *args: Any, **kwargs: Any) -> bytes:
        nonlocal multiformats_call_count
        multiformats_call_count += 1
        return original_digest(self, *args, **kwargs)

    def unwrap_spy(self: multihash.Multihash, *args: Any, **kwargs: Any) -> bytes:
        nonlocal multiformats_call_count
        multiformats_call_count += 1
        return original_unwrap(self, *args, **kwargs)

    monkeypatch.setattr(multihash.Multihash, "digest", digest_spy)
    monkeypatch.setattr(multihash.Multihash, "unwrap", unwrap_spy)

    digest = blake3_hashfn(data)

    assert digest == blake3(data).digest(length=32)
    assert multiformats_call_count == 0, (
        "blake3_hashfn must not route through multiformats multihash wrappers "
        "(perf: ~134us/hash overhead)"
    )
