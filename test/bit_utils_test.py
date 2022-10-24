import sys 
sys.path.append("..")
import bit_utils

#mask tests
print(bit_utils.mask(bytes([0b11111111]), 0, 5) == bytes([0b11111]))
print(bit_utils.mask(bytes([0b10101010]), 0, 5) == bytes([0b10101]))
print(bit_utils.mask(bytes([0b10000000]), 0, 5) == bytes([0b10000]))
print(bit_utils.mask(bytes([0b00010000]), 0, 5) == bytes([0b00010]))
print(bit_utils.mask(bytes([0b10000100, 0b10010000]), 0, 9) == bit_utils.bitstring_to_bytes('0b100001001'))
print(bit_utils.mask(bytes([0b10101010, 0b10101010]), 0, 9) == bit_utils.bitstring_to_bytes('0b101010101'))
print(bit_utils.mask(bytes([0b10000100, 0b10010000]), 1, 5) == bit_utils.bitstring_to_bytes('0b10010'))
print(bit_utils.mask(bytes([0b10101010, 0b10101010]), 1, 5) == bit_utils.bitstring_to_bytes('0b01010'))
print(bit_utils.mask(bytes([0b10000100, 0b10010000]), 2, 5) == bit_utils.bitstring_to_bytes('0b01000'))
print(bit_utils.mask(bytes([0b10101010, 0b10101010]), 2, 5) == bit_utils.bitstring_to_bytes('0b10101'))

#bitmap_has tests 
print(not bit_utils.bitmap_has(bytes([0b0]), 0))
print(not bit_utils.bitmap_has(bytes([0b0]), 1))
print(bit_utils.bitmap_has(bytes([0b1]), 0))
print(not bit_utils.bitmap_has(bytes([0b1]), 1))
print(not bit_utils.bitmap_has(bytes([0b101010]), 2))
print(bit_utils.bitmap_has(bytes([0b101010]), 3))
print(not bit_utils.bitmap_has(bytes([0b101010]), 4))
print(bit_utils.bitmap_has(bytes([0b101010]), 5))
print(bit_utils.bitmap_has(bytes([0b100000]), 5))
print(bit_utils.bitmap_has(bytes([0b0100000]), 5))
print(bit_utils.bitmap_has(bytes([0b00100000]), 5))

#index tests 
print(bit_utils.index(bytes([0b111111]), 0) == 0)

#set_bit tests 
print(bit_utils.set_bit(bytes([0b0]), 0, 1) == bytes([0b00000001]))
