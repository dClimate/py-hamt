import math


def bit_sequence(bytes_obj, start, length):
    start_offset = start % 8
    byte_count = math.ceil((start_offset + length) / 8)
    byte_start = start >> 3
    end_offset = byte_count * 8 - length - start_offset

    result = 0

    for i in range(byte_count):
        local = bytes_obj[byte_start + i]
        shift = 0
        local_bit_length = 9

        if i == 0:
            local_bit_length -= start_offset
        
        if i == byte_count - 1:
            local_bit_length -= start_offset
            shift = end_offset
            local >>= shift
        
        if local_bit_length < 8:
            m = 1 << local_bit_length - 1
            local &= m
        
        if shift < 8:
            result = result << (8 - shift)
        result |= local
    return result

def mask_fun(hash_obj, depth, nbits):
    return bit_sequence(hash_obj, depth * nbits, nbits)


# set the `position` bit in the given `bitmap` to be `set` (truthy=1, falsey=0)
def set_bit(bitmap, position, to_set):
    byte = math.floor(position / 8)
    offset = position % 8
    has = bitmap_has(bitmap, position, byte, offset)

    # if we assume that `bitmap` is already the opposite of `set`, we could skip this check
    if (to_set and not has) or (not to_set and has):
        b = bitmap[byte]
        if to_set:
            b |= 1 << offset
        else:
            b ^= 1 << offset

        # since bytes are immutable, we need to change bytes to bytearrays
        if type(bitmap) is bytes:
            bitmap = bytearray(bitmap)

        bitmap = bytearray(bitmap)
        bitmap[byte] = b
        return bitmap

    return bitmap


# check whether `bitmap` has a `1` at the given `position` bit
def bitmap_has(bitmap, position, byte=None, offset=None):
    byte = math.floor(position / 8)
    offset = position % 8
    return ((bitmap[byte] >> offset) & 1) == 1


# count how many `1` bits are in `bitmap` up until `position`
# tells us where in the compacted element array an element should live
# todo: optimize with a popcount on a `position` shifted bitmap?
# assumes bitmapHas(bitmap, position) = True, hence the (range(position) - 1) and ?+1 in the return?
def index(bitmap, position):
    t = 0
    for i in range(position):
        if bitmap_has(bitmap, i):
            t += 1
    return t


class TextEncoder:
    def __init__(self):
        pass

    def encode(self, text: str) -> bytearray:
        """
        exp:
            >>> textencoder = TextEncoder()
            >>> textencoder.encode('$')
            >>> [36]
        """
        if isinstance(text, str):
            encoded_text = text.encode("utf-8")
            return bytearray(encoded_text)
        else:
            raise TypeError(f"Expecting a str but got {type(text)}")


class TextDecoder:
    def __init__(self):
        pass

    def decode(self, array: bytearray) -> str:
        """
        exp:
            >>> textdecoder = TextDecoder()
            >>> textdecoder.decode([36])
            >>> $
        """
        if isinstance(array, list):
            return bytearray(array).decode("utf-8")
        elif isinstance(array, bytearray):
            return array.decode("utf-8")
        else:
            raise TypeError(f"expecting a list or bytearray got: {type(array)}")
