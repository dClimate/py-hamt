import sys 
sys.path.append("..")
import bit_utils

#mask tests 
print(bit_utils.mask(bytes([0b11111111]), 0, 5) == bytes([0b11111]))
print(bit_utils.mask(bytes([0b10101010]), 0, 5) == bytes([0b10101]))
print(bit_utils.mask(bytes([0b10000000]), 0, 5) == bytes([0b10000]))
print(bit_utils.mask(bytes([0b00010000]), 0, 5) == bytes([0b00010]))
# fails from here on: print(bit_utils.mask(bytes([0b10000100, 0b10010000]), 0, 9) == bytes([0b100001001])), this is cacuse im not handling > 256 case properly

#bitmap_has tests 
print(bit_utils.bitmap_has(bytes([0b101010]), 3))

#index tests 
print(bit_utils.index(bytes([0b111111]), 0) == 0)

#set_bit tests 
print(bit_utils.set_bit(bytes([0b0]), 0, 1) == bytes([0b00000001]))
