from abc import ABC, abstractmethod


class Store(ABC):
    """This is an Abstract Base Class that represents a storage mechanism the HAMT can use for keeping data."""

    @abstractmethod
    async def save(self, node: bytes) -> bytes:
        """Take any set of bytes, save it to the storage mechanism, and return an ID that can be used to retrieve those bytes later."""

    @abstractmethod
    async def load(self, id: bytes) -> bytes:
        """Retrieve the bytes based on an ID returned earlier by the save function."""
