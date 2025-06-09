import pytest

from py_hamt.hamt import extract_bits


# Most of these tests were adapted over to python from rvagg's IAMap tests in JS
def test_extract_bits() -> None:
    assert extract_bits(bytes([0b11111111]), 0, 5) == 0b11111
    assert extract_bits(bytes([0b10101010]), 0, 5) == 0b10101
    assert extract_bits(bytes([0b10000000]), 0, 5) == 0b10000
    assert extract_bits(bytes([0b00010000]), 0, 5) == 0b00010
    assert extract_bits(bytes([0b10000100, 0b10010000]), 0, 9) == 0b100001001
    assert extract_bits(bytes([0b10101010, 0b10101010]), 0, 9) == 0b101010101
    assert extract_bits(bytes([0b10000100, 0b10010000]), 1, 5) == 0b10010
    assert extract_bits(bytes([0b10101010, 0b10101010]), 1, 5) == 0b01010
    assert extract_bits(bytes([0b10000100, 0b10010000]), 2, 5) == 0b01000
    assert extract_bits(bytes([0b10101010, 0b10101010]), 2, 5) == 0b10101
    assert (
        extract_bits(bytes([0b10000100, 0b10010000, 0b10000100, 0b10000100]), 3, 5)
        == 0b01000
    )
    assert (
        extract_bits(bytes([0b10101010, 0b10101010, 0b10101010, 0b10101010]), 3, 5)
        == 0b01010
    )
    assert (
        extract_bits(bytes([0b10000100, 0b10010000, 0b10000100, 0b10000100]), 4, 5)
        == 0b01001
    )
    assert (
        extract_bits(bytes([0b10101010, 0b10101010, 0b10101010, 0b10101010]), 4, 5)
        == 0b10101
    )

    with pytest.raises(
        IndexError, match="Arguments extract more bits than remain in the hash bits"
    ):
        extract_bits(bytes([0b1]), 20, 20)
