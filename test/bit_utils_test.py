from py_hamt import bit_utils

# mask_fun tests
assert bit_utils.extract_bits(bytes([0b11111111]), 0, 5) == 0b11111
assert bit_utils.extract_bits(bytes([0b10101010]), 0, 5) == 0b10101
assert bit_utils.extract_bits(bytes([0b10000000]), 0, 5) == 0b10000
assert bit_utils.extract_bits(bytes([0b00010000]), 0, 5) == 0b00010

# bitmap_has tests
assert not bit_utils.bitmap_has(bytes([0b0]), 0)
assert not bit_utils.bitmap_has(bytes([0b0]), 1)
assert bit_utils.bitmap_has(bytes([0b1]), 0)
assert not bit_utils.bitmap_has(bytes([0b1]), 1)
assert not bit_utils.bitmap_has(bytes([0b101010]), 2)
assert bit_utils.bitmap_has(bytes([0b101010]), 3)
assert not bit_utils.bitmap_has(bytes([0b101010]), 4)
assert bit_utils.bitmap_has(bytes([0b101010]), 5)
assert bit_utils.bitmap_has(bytes([0b100000]), 5)
assert bit_utils.bitmap_has(bytes([0b0100000]), 5)
assert bit_utils.bitmap_has(bytes([0b00100000]), 5)
print("bitmap_has tests passed")

# index tests
assert bit_utils.rank(bytes([0b111111]), 0) == 0
assert bit_utils.rank(bytes([0b111111]), 1) == 1
assert bit_utils.rank(bytes([0b111111]), 2) == 2
assert bit_utils.rank(bytes([0b111111]), 4) == 4
assert bit_utils.rank(bytes([0b111100]), 2) == 0
assert bit_utils.rank(bytes([0b111101]), 4) == 3
assert bit_utils.rank(bytes([0b111001]), 4) == 2
assert bit_utils.rank(bytes([0b111000]), 4) == 1
assert bit_utils.rank(bytes([0b110000]), 4) == 0
# new node, no bitmask_fun, insertion at the start
assert bit_utils.rank(bytes([0b000000]), 0) == 0
assert bit_utils.rank(bytes([0b000000]), 1) == 0
assert bit_utils.rank(bytes([0b000000]), 2) == 0
assert bit_utils.rank(bytes([0b000000]), 3) == 0
print("index tests passed")

# set_bit tests
assert bit_utils.set_bit(bytes([0b0]), 0, 1) == bytes([0b00000001])
assert bit_utils.set_bit(bytes([0b0]), 0, 1) == bytes(([0b00000001]))
assert bit_utils.set_bit(bytes([0b0]), 1, 1) == bytes(([0b00000010]))
assert bit_utils.set_bit(bytes([0b0]), 7, 1) == bytes(([0b10000000]))
assert bit_utils.set_bit(bytes([0b11111111]), 0, 1) == bytes(([0b11111111]))
assert bit_utils.set_bit(bytes([0b11111111]), 7, 1) == bytes(([0b11111111]))
assert bit_utils.set_bit(bytes([0b01010101]), 1, 1) == bytes(([0b01010111]))
assert bit_utils.set_bit(bytes([0b01010101]), 7, 1) == bytes(([0b11010101]))
assert bit_utils.set_bit(bytes([0b11111111]), 0, 0) == bytes(([0b11111110]))
assert bit_utils.set_bit(bytes([0b11111111]), 1, 0) == bytes(([0b11111101]))
assert bit_utils.set_bit(bytes([0b11111111]), 7, 0) == bytes(([0b01111111]))
assert bit_utils.set_bit(bytes([0b0, 0b11111111]), 8 + 0, 1) == bytes(
    ([0b0, 0b11111111])
)
assert bit_utils.set_bit(bytes([0b0, 0b11111111]), 8 + 7, 1) == bytes(
    ([0b0, 0b11111111])
)
assert bit_utils.set_bit(bytes([0b0, 0b01010101]), 8 + 1, 1) == bytes(
    ([0b0, 0b01010111])
)
assert bit_utils.set_bit(bytes([0b0, 0b01010101]), 8 + 7, 1) == bytes(
    ([0b0, 0b11010101])
)
assert bit_utils.set_bit(bytes([0b0, 0b11111111]), 8 + 0, 0) == bytes(
    ([0b0, 0b11111110])
)
assert bit_utils.set_bit(bytes([0b0, 0b11111111]), 8 + 1, 0) == bytes(
    ([0b0, 0b11111101])
)
assert bit_utils.set_bit(bytes([0b0, 0b11111111]), 8 + 7, 0) == bytes(
    ([0b0, 0b01111111])
)
assert bit_utils.set_bit(bytes([0b0]), 0, 0) == bytes(([0b00000000]))
assert bit_utils.set_bit(bytes([0b0]), 7, 0) == bytes(([0b00000000]))
assert bit_utils.set_bit(bytes([0b01010101]), 0, 0) == bytes(([0b01010100]))
assert bit_utils.set_bit(bytes([0b01010101]), 6, 0) == bytes(([0b00010101]))
assert bit_utils.set_bit(
    bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 0, 0
) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000001]))
assert bit_utils.set_bit(
    bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 0, 1
) == bytes(([0b11000011, 0b11010010, 0b01001010, 0b0000001]))
assert bit_utils.set_bit(
    bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 12, 0
) == bytes(([0b11000010, 0b11000010, 0b01001010, 0b0000001]))
assert bit_utils.set_bit(
    bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 12, 1
) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000001]))
assert bit_utils.set_bit(
    bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 24, 0
) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000000]))
assert bit_utils.set_bit(
    bytes([0b11000010, 0b11010010, 0b01001010, 0b0000001]), 24, 1
) == bytes(([0b11000010, 0b11010010, 0b01001010, 0b0000001]))
print("all tests passed")
