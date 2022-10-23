import math

#there is no bit-sequence lib in python like there is in go/js... converting to bitstring (works like https://www.npmjs.com/package/bit-sequence)
def bitstring_to_bytes(s):
    return int(s, 2).to_bytes((len(s) + 7) // 8, byteorder="big")

def bit_sequence(bytes, start, length):  
    # Converting the bytes array into one long bit string
    binstring = "".join([bin(byte)[2:].zfill(8) for byte in bytes])

    # Converting the required part of the bitstring
    ret_byte = bitstring_to_bytes(binstring[start:start+length])

    return  ret_byte

def mask(hash, depth, nbits):
  return bit_sequence(hash, depth * nbits, nbits)

# set the `position` bit in the given `bitmap` to be `set` (truthy=1, falsey=0)
def set_bit(bitmap, position, set):
    byte = math.floor(position / 8)
    offset = position % 8 
    has = bitmap_has(bitmap, position, byte, offset)

    #if we assume that `bitmap` is already the opposite of `set`, we could skip this check
    if (set and not has) or (not set and has): 
        b = bitmap[byte]
        if set:
            b |= (1 << offset)
        else: 
            b ^= (1 << offset)
        
        #since bytes are immutable, we need to change bytes to bytearrays
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
    for i in range(position - 1):
        if bitmap_has(bitmap, i):
            t += 1
    return t 
