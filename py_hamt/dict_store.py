from multiformats import multihash
from multiformats.multihash import Multihash

from .store import Store


# Inspired by https://github.com/rvagg/iamap/blob/master/examples/memory-backed.js
class DictStore(Store):
    store: dict[bytes, bytes]
    hash_alg: Multihash

    def __init__(self):
        self.store = {}
        self.hash_alg = multihash.get("blake3")

    async def save(self, node: bytes) -> bytes:
        hash = self.hash_alg.digest(node, size=32)
        self.store[hash] = node
        return hash

    async def load(self, id: bytes) -> bytes:
        if id in self.store:
            return self.store[id]
        else:
            raise Exception("ID not found in store")
