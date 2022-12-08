# this is an example of defining a MemoryStore and instantiating an iamap, based on https://github.com/rvagg/iamap/blob/master/examples/memory-backed.js/ and https://github.com/rvagg/iamap/blob/master/test/common.js

import json
import sys

sys.path.append("..")
from iamap import create, TextEncoder, load, save
import asyncio
from multiformats import multihash, CID

def my_hash(obj):
    stringified =  (obj)
    buf = TextEncoder().encode(stringified)
    return multihash.get("murmur3-x64-64").digest(buf)

# lets define a MemoryStore
class MemoryStore:
    def __init__(self):
        self.map = {}  # where objects get stored

    async def save(self, obj):
        id = my_hash(obj)
        self.map[id] =  obj
        return id

    async def load(self, id):
        return self.map.get(id)

    def is_equal(self, id1, id2):
        return id1 == id2

    def is_link(self, obj):
        return isinstance(obj, float) or isinstance(obj, int)


# sample function to asynchronously instantiate iamap
async def main(store):
    iamap = await create(store)
    import ipdb; ipdb.set_trace()
    new_map = await iamap.set(b"a", CID.decode("zb2rhe5P4gXftAwvA4eXQ5HJwsER2owDyS9sKaQRRVQPn93bA"))
    import ipdb; ipdb.set_trace()


# lets define parameters
store = MemoryStore()
# options = {
#     "hash_alg": 0x23,
#     "bit_width": DEFAULT_BIT_WIDTH,
#     "bucket_size": DEFAULT_BUCKET_SIZE,
# }
# lets instantiate an example iamap
iamap = asyncio.run(main(store))
import ipdb; ipdb.set_trace()

