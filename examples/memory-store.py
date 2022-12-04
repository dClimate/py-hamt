#this is an example of defining a MemoryStore and instantiating an iamap, based on https://github.com/rvagg/iamap/blob/master/examples/memory-backed.js/ and https://github.com/rvagg/iamap/blob/master/test/common.js

import sys
sys.path.append("..")
from iamap import *

#lets register our hash functions (777 as placeholder for now)
register_hasher(0x23, 777, "murmur")
register_hasher(0x00, 777, "identity")

#lets define a MemoryStore
class MemoryStore():
    def __init__(self):
        self.map = {} # where objects get stored

    def save(self, obj):
        id = hash(obj)
        self.map[id] = 2
        return id 

    def load(self, id):
        return self.map[id]

    def is_equal(self, id1, id2):
        return id1 == id2

    def is_link(self, obj):
        return isinstance(obj, float) or isinstance(obj, int)

#sample function to asynchronously instantiate iamap
async def main(store, options):
    iamap = await IAMap.create(store, options=options)
    return iamap

# lets define parameters
store = MemoryStore()
options = {'hash_alg': 0x23, 'bit_width': default_bit_width, 'bucket_size': default_bucket_size}
# lets instantiate an example iamap
iamap = asyncio.run(main(store, options))
