def test_mask_fun():
    from py_hamt.bit_utils import extract_bits

    assert extract_bits(bytes([0b11111111]), 0, 5) == 0b11111
    assert extract_bits(bytes([0b10101010]), 0, 5) == 0b10101
    assert extract_bits(bytes([0b10000000]), 0, 5) == 0b10000
    assert extract_bits(bytes([0b00010000]), 0, 5) == 0b00010


def test_bitmap_has():
    from py_hamt.bit_utils import bitmap_has

    assert not bitmap_has(bytes([0b0]), 0)
    assert not bitmap_has(bytes([0b0]), 1)
    assert bitmap_has(bytes([0b1]), 0)
    assert not bitmap_has(bytes([0b1]), 1)
    assert not bitmap_has(bytes([0b101010]), 2)
    assert bitmap_has(bytes([0b101010]), 3)
    assert not bitmap_has(bytes([0b101010]), 4)
    assert bitmap_has(bytes([0b101010]), 5)
    assert bitmap_has(bytes([0b100000]), 5)
    assert bitmap_has(bytes([0b0100000]), 5)
    assert bitmap_has(bytes([0b00100000]), 5)


def test_index():
    from py_hamt.bit_utils import rank

    assert rank(bytes([0b111111]), 0) == 0
    assert rank(bytes([0b111111]), 1) == 1
    assert rank(bytes([0b111111]), 2) == 2
    assert rank(bytes([0b111111]), 4) == 4
    assert rank(bytes([0b111100]), 2) == 0
    assert rank(bytes([0b111101]), 4) == 3
    assert rank(bytes([0b111001]), 4) == 2
    assert rank(bytes([0b111000]), 4) == 1
    assert rank(bytes([0b110000]), 4) == 0

    # new node, no bitmask_fun, insertion at the start
    assert rank(bytes([0b000000]), 0) == 0
    assert rank(bytes([0b000000]), 1) == 0
    assert rank(bytes([0b000000]), 2) == 0
    assert rank(bytes([0b000000]), 3) == 0


def test_set_bit():
    from py_hamt.bit_utils import set_bit

    assert set_bit(bytes([0b0]), 0, 1) == bytes([0b00000001])
    assert set_bit(bytes([0b0]), 0, 1) == bytes(([0b00000001]))
    assert set_bit(bytes([0b0]), 1, 1) == bytes(([0b00000010]))
    assert set_bit(bytes([0b0]), 7, 1) == bytes(([0b10000000]))
    assert set_bit(bytes([0b11111111]), 0, 1) == bytes(([0b11111111]))
    assert set_bit(bytes([0b11111111]), 7, 1) == bytes(([0b11111111]))
    assert set_bit(bytes([0b01010101]), 1, 1) == bytes(([0b01010111]))
    assert set_bit(bytes([0b01010101]), 7, 1) == bytes(([0b11010101]))
    assert set_bit(bytes([0b11111111]), 0, 0) == bytes(([0b11111110]))
    assert set_bit(bytes([0b11111111]), 1, 0) == bytes(([0b11111101]))
    assert set_bit(bytes([0b11111111]), 7, 0) == bytes(([0b01111111]))
    assert set_bit(bytes([0b0, 0b11111111]), 8 + 0, 1) == bytes(([0b0, 0b11111111]))
    assert set_bit(bytes([0b0, 0b11111111]), 8 + 7, 1) == bytes(([0b0, 0b11111111]))
    assert set_bit(bytes([0b0, 0b01010101]), 8 + 1, 1) == bytes(([0b0, 0b01010111]))
    assert set_bit(bytes([0b0, 0b01010101]), 8 + 7, 1) == bytes(([0b0, 0b11010101]))
    assert set_bit(bytes([0b0, 0b11111111]), 8 + 0, 0) == bytes(([0b0, 0b11111110]))
    assert set_bit(bytes([0b0, 0b11111111]), 8 + 1, 0) == bytes(([0b0, 0b11111101]))
    assert set_bit(bytes([0b0, 0b11111111]), 8 + 7, 0) == bytes(([0b0, 0b01111111]))
    assert set_bit(bytes([0b0]), 0, 0) == bytes(([0b00000000]))
    assert set_bit(bytes([0b0]), 7, 0) == bytes(([0b00000000]))
    assert set_bit(bytes([0b01010101]), 0, 0) == bytes(([0b01010100]))
    assert set_bit(bytes([0b01010101]), 6, 0) == bytes(([0b00010101]))
    assert set_bit(
        bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 0, 0
    ) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000001]))
    assert set_bit(
        bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 0, 1
    ) == bytes(([0b11000011, 0b11010010, 0b01001010, 0b0000001]))
    assert set_bit(
        bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 12, 0
    ) == bytes(([0b11000010, 0b11000010, 0b01001010, 0b0000001]))
    assert set_bit(
        bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 12, 1
    ) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000001]))
    assert set_bit(
        bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 24, 0
    ) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000000]))
    assert set_bit(
        bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 24, 1
    ) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000001]))
