from abc import ABC, abstractmethod
from multiformats import multihash
from multiformats.multihash import Multihash


class Store(ABC):
    """This is an Abstract Base Class that represents a storage mechanism the HAMT can use for keeping data."""

    @abstractmethod
    def save(self, node: bytes) -> bytes:
        """Take any set of bytes, save it to the storage mechanism, and return an ID that can be used to retrieve those bytes later."""

    @abstractmethod
    def load(self, id: bytes) -> bytes:
        """Retrieve the bytes based on an ID returned earlier by the save function."""


# Inspired by https://github.com/rvagg/iamap/blob/master/examples/memory-backed.js
class DictStore(Store):
    """A basic implementation of a backing store, mostly for demonstration and testing purposes. It hashes all inputs and uses that as a key to an in-memory python dict. The hash bytes are the ID that `save` returns and `load` takes in."""

    store: dict[bytes, bytes]
    """@private"""
    hash_alg: Multihash
    """@private"""

    def __init__(self):
        self.store = {}
        self.hash_alg = multihash.get("blake3")

    def save(self, node: bytes) -> bytes:
        hash = self.hash_alg.digest(node, size=32)
        self.store[hash] = node
        return hash

    def load(self, id: bytes) -> bytes:
        if id in self.store:
            return self.store[id]
        else:
            raise Exception("ID not found in store")
