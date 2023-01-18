import math
import typing


def bit_sequence(bytes_obj: bytes, start: int, length: int) -> int:
    """Given a bytes-like object (i.e. an array of 8-bit integers),
    extract an arbitrary sequence of the underlying bits and convert
    them into an unsigned integer value
    Adapted from https://github.com/rvagg/bit-sequence
    Args:
        bytes_obj (bytes): bytes-like object from which to extract bits
        start (int):  an integer bit index to start extraction (not a byte index)
        length (int):  is the number of bits to extract from `bytes_obj`

    Returns:
        int: An unsigned integer version of the bit sequence
    """
    start_offset = start % 8
    byte_count = math.ceil((start_offset + length) / 8)
    byte_start = start >> 3
    end_offset = byte_count * 8 - length - start_offset

    result = 0

    for i in range(byte_count):
        local = bytes_obj[byte_start + i]
        shift = 0
        local_bit_length = 8

        if i == 0:
            local_bit_length -= start_offset

        if i == byte_count - 1:
            local_bit_length -= start_offset
            shift = end_offset
            local >>= shift

        if local_bit_length < 8:
            m = (1 << local_bit_length) - 1
            local &= m

        if shift < 8:
            result = result << (8 - shift)
        result |= local

    return result


def mask_fun(hash_obj: bytes, depth: int, nbits: int) -> int:
    """Helper function that calls `bit_sequence` with specific start and length
    that are functions of characteristics of the hamt

    Args:
        hash_obj (bytes): binary hash to extract bit sequence from
        depth (int): depth of the node containing the hash
        nbits (int): bit width of hash

    Returns:
        int: _description_
    """
    return bit_sequence(hash_obj, depth * nbits, nbits)


def set_bit(bitmap: bytes, position: int, to_set: bool) -> bytes:
    """set the `position` bit in the given `bitmap` to be `to_set` (truthy=1, falsey=0)

    Args:
        bitmap (bytes): bitmap to modify
        position (int): location in the bitmap to modify
        to_set (bool): whether to set true or false

    Returns:
        bytes: Modified bitmap
    """
    byte = math.floor(position / 8)
    offset = position % 8
    has = bitmap_has(bitmap, None, byte, offset)
    # if we assume that `bitmap` is already the opposite of `set`, we could skip this check
    if (to_set and not has) or (not to_set and has):
        new_bit_map = bytearray(bitmap)
        b = bitmap[byte]
        if to_set:
            b |= 1 << offset
        else:
            b ^= 1 << offset

        # since bytes are immutable, we need to change bytes to bytearrays
        new_bit_map[byte] = b
        return bytes(new_bit_map)
    return bitmap


def bitmap_has(
    bitmap: bytes,
    position: typing.Optional[int],
    byte: typing.Optional[int] = None,
    offset: typing.Optional[int] = None,
) -> bool:
    """check whether `bitmap` has a `1` at the given `position` bit. If `position`
    is `None`, intead caclulate position using a `byte` position and an `offset`

    Args:
        bitmap (bytes): bytes to check
        position (typing.Optional[int]): Position of bit to modify
        byte (typing.Optional[int], optional): byte postition to modify. Defaults to None.
        offset (typing.Optional[int], optional) Defaults to None.

    Returns:
        bool: _description_
    """
    if position is not None:
        byte = math.floor(position / 8)
        offset = position % 8
    return ((bitmap[byte] >> offset) & 1) == 1


def index(bitmap: bytes, position: int) -> int:
    """count how many `1` bits are in `bitmap` up until `position`
    tells us where in the compacted element array an element should live
    Args:
        bitmap (bytes): bitmap to count truthy bits on
        position (int): where to stop coutning

    Returns:
        int: how many bits are `1` in `bitmap`
    """
    t = 0
    for i in range(position):
        if bitmap_has(bitmap, i):
            t += 1
    return t
