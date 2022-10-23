from bit_utils import mask, set_bit, bitmap_has, index
import asyncio

#set defaults
default_bit_width = 8 # 2^8 = 256 buckets or children per node 
default_bucket_size = 5 # array size for a bucket of values

class KV(): 
    def __init__(self, key, value):
        self.key = key 
        self.value = value

    def to_serializable(self):
        return [self.key, self.value]

    @classmethod
    def from_serializable(self, obj):
        #to do, add checks for obj
        #assert(Array.isArray(obj))
        #assert(obj.length === 2)
        return KV(obj[0], obj[1])

# a element in the data array that each node holds, each element could be either a container of
#  an array (bucket) of KVs or a link to a child node
class Element():
    def __init__(self, bucket=None, link=None):
        assert ((bucket == None) == (link != None)) # assert that either the bucket or link is None
        self.bucket = bucket #should be array of KV's
        self.link = link

    def to_serializable():
        pass

class IAMap(object):
    #Create a new IAMap instance with a backing store
    ## define async create func https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init
    @classmethod
    async def create(self, data, options={'bit_width': default_bit_width, 'bucket_size': default_bucket_size}, map = [], depth=0):
        self = IAMap()
        self.data = data
        self.options = options 
        self.map = map
        self.depth = depth
        return self

    # Asynchronously create a new `IAMap` instance identical to this one but with `key` set to `value`.
    async def set(self, key, value, _cached_hash):
        #const hash = _cachedHash instanceof Uint8Array ? _cachedHash : await hasher(this)(key)
        #const bitpos = mask(hash, this.depth, this.config.bitWidth)
        hash = _cached_hash #this needs to check if the submitted hash is a byte array, and if not await call the hasher function for this key... for now i will assume hash input
        bitpos = mask(hash, self.depth, self.options['bit_width'])
        if bitmap_has(self.map, bitpos):
            pass

        pass 

    # Asynchronously find and return a value for the given `key` if it exists within this `IAMap`.
    async def get(self, key):
        pass 

    async def delete(self, key):
        pass

def find_element(node, bitpos, key): 
    element_at = index(node.map, bitpos)
    element = node.data[element_at]
    if element.bucket: 
        for bucket_index in range(len(element.bucket)):
            #still to do this
            print(bucket_index)
        pass

#lets instantiate an iamap 

#sample input 
async def main(data):
    iamap = await IAMap.create(data)
    return iamap

# this is an example of a few kv's
kv1 = KV(bytes([0x13, 0x00, 0x00, 0x00, 0x08, 0x00]), 3)
kv2 = KV(bytes([0x11, 0x00, 0x00, 0x00, 0x08, 0x00]), 6)
kv3 = KV(bytes([0x13, 0x11, 0x00, 0x00, 0x08, 0x00]), 3)

# this is an example of a couple of bucket elements 
elem1 = Element([kv1, kv2], None)
elem2 = Element([kv3], None)

#this is an example of the data for an iamap (need ot figure out links for this to actually make sense)
data = [elem1, elem2]

# this is an example of an iamap
iamap = asyncio.run(main(data))
