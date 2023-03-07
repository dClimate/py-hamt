import math


def extract_bits(hash_obj: bytes, depth: int, nbits: int) -> int:
    """Extract `nbits` bits from `hash_obj`, beginning at position `depth * nbits`,
    and convert them into an unsigned integer value.

    Args:
        hash_obj (bytes): binary hash to extract bit sequence from
        depth (int): depth of the node containing the hash
        nbits (int): bit width of hash

    Returns:
        int: An unsigned integer version of the bit sequence
    """
    start = depth * nbits
    start_offset = start % 8

    byte_count = math.ceil((start_offset + nbits) / 8)
    byte_start = start >> 3
    end_offset = byte_count * 8 - nbits - start_offset

    result = 0

    for i in range(byte_count):
        local = hash_obj[byte_start + i]
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


def set_bit(bitmap: bytes, position: int, to_set: bool) -> bytes:
    """set the `position` bit in the given `bitmap` to be `to_set` (truthy=1, falsey=0)

    Args:
        bitmap (bytes): bitmap to modify
        position (int): location in the bitmap to modify
        to_set (bool): whether to set true or false

    Returns:
        bytes: Modified bitmap
    """
    has = bitmap_has(bitmap, position)
    byte = math.floor(position / 8)
    offset = position % 8
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
    position: int,
) -> bool:
    """check whether `bitmap` has a `1` at the given `position` bit.

    Args:
        bitmap (bytes): bytes to check
        position (int): Position of bit to read

    Returns:
        bool: whether the `bitmap` has a 1 value at the `position` bit
    """
    byte = math.floor(position / 8)
    offset = position % 8
    return ((bitmap[byte] >> offset) & 1) == 1


def rank(bitmap: bytes, position: int) -> int:
    """count how many `1` bits are in `bitmap` up until `position`
    tells us where in the compacted element array an element should live
    Args:
        bitmap (bytes): bitmap to count truthy bits on
        position (int): where to stop counting

    Returns:
        int: how many bits are `1` in `bitmap`
    """
    t = 0
    for i in range(position):
        if bitmap_has(bitmap, i):
            t += 1
    return t
