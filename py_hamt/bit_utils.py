def extract_bits(hash_bytes: bytes, depth: int, nbits: int) -> int:
    """Extract `nbits` bits from `hash_bytes`, beginning at position `depth * nbits`,
    and convert them into an unsigned integer value.

    Args:
        hash_bytes (bytes): binary hash to extract bit sequence from
        depth (int): depth of the node containing the hash
        nbits (int): bit width of hash

    Returns:
        int: An unsigned integer version of the bit sequence
    """
    # This is needed since int.bit_length on a integer representation of a bytes object ignores leading 0s
    hash_bit_length = len(hash_bytes) * 8

    start_bit_index = depth * nbits

    if hash_bit_length - start_bit_index < nbits:
        raise IndexError("Arguments extract more bits than remain in the hash bits")

    mask = (0b1 << (hash_bit_length - start_bit_index)) - 1
    n_chop_off_at_end = int.bit_length(mask) - nbits

    hash_as_int = int.from_bytes(hash_bytes)
    result = (hash_as_int & mask) >> n_chop_off_at_end

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
    byte = position // 8
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
    byte_index = position // 8
    offset = position % 8
    return ((bitmap[byte_index] >> offset) & 1) == 1


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
