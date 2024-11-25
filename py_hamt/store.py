from abc import ABC, abstractmethod
from dag_cbor.ipld import IPLDKind
import requests
from msgspec import json
from multiformats import multihash
from multiformats import CID
from multiformats.multihash import Multihash


class Store(ABC):
    """This is an Abstract Base Class that represents a storage mechanism the HAMT can use for keeping data.

    This should be some sort of immutable store. So different ids should be returned for different data, as opposed to one id being mutated.

    The return type of save and input to load is really type IPLDKind, but the documentation generates this strange type instead since IPLDKind is a type union.
    """

    @abstractmethod
    def save_raw(self, data: bytes) -> IPLDKind:
        """Take any set of bytes, save it to the storage mechanism, and return an ID in the type of IPLDKind that can be used to retrieve those bytes later."""

    @abstractmethod
    def save_dag_cbor(self, data: bytes) -> IPLDKind:
        """Take a set of bytes and save it just like `save_raw`, except this method has additional context that the data is in a dag-cbor format."""

    @abstractmethod
    def load(self, id: IPLDKind) -> bytes:
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

    def save(self, data: bytes) -> bytes:
        hash = self.hash_alg.digest(data, size=32)
        self.store[hash] = data
        return hash

    def save_raw(self, data: bytes) -> bytes:
        """"""
        return self.save(data)

    def save_dag_cbor(self, data: bytes) -> bytes:
        """"""
        return self.save(data)

    # Ignore the type error since bytes is in the IPLDKind type
    def load(self, id: bytes) -> bytes:  # type: ignore
        """"""
        if id in self.store:
            return self.store[id]
        else:
            raise Exception("ID not found in store")


class IPFSStore(Store):
    """
    Use IPFS as a backing store for a HAMT. The IDs returned from save and used by load are IPFS CIDs.

    save methods use the RPC API but Load uses the HTTP Gateway, so read-only on HAMTs will only access HTTP Gateways.
    If only doing reads, then the RPC API will never be called.
    """

    def __init__(
        self,
        timeout_seconds=30,
        gateway_uri_stem="http://127.0.0.1:8080",
        rpc_uri_stem="http://127.0.0.1:5001",
        hasher="blake3",
    ):
        self.timeout_seconds = timeout_seconds
        """
        You can modify this variable directly if you choose.

        This sets the timeout in seconds for all HTTP requests.
        """
        self.gateway_uri_stem = gateway_uri_stem
        """
        URI stem of the IPFS HTTP gateway that IPFSStore will retrieve blocks from.
        """
        self.rpc_uri_stem = rpc_uri_stem
        """URI Stem of the IPFS RPC API that IPFSStore will send data to."""
        self.hasher = hasher
        """The hash function to send to IPFS when storing bytes."""

    def save(self, data: bytes, cid_codec: str) -> CID:
        """
        This saves the data to an ipfs daemon by calling the RPC API, and then returns the CID. By default, `save` pins content it adds.

        ```python
        from py_hamt import IPFSStore

        ipfs_store = IPFSStore()
        cid = ipfs_store.save("foo".encode(), "raw")
        print(cid.human_readable)
        ```
        """
        response = requests.post(
            f"{self.rpc_uri_stem}/api/v0/add?cid-codec={cid_codec}&hash={self.hasher}&pin=true",
            files={"file": data},
        )

        cid_str: str = json.decode(response.content)["Hash"]  # type: ignore
        cid = CID.decode(cid_str)

        return cid

    def save_raw(self, data: bytes) -> CID:
        return self.save(data, "raw")

    def save_dag_cbor(self, data: bytes) -> CID:
        return self.save(data, "dag-cbor")

    # Ignore the type error since CID is in the IPLDKind type
    def load(self, id: CID) -> bytes:  # type: ignore
        """
        This retrieves the raw bytes by calling the provided HTTP gateway.
        ```python
        from py_hamt import IPFSStore
        from multiformats import CID

        ipfs_store = IPFSStore()
        # This is just an example CID
        cid = CID.decode("bafybeiaysi4s6lnjev27ln5icwm6tueaw2vdykrtjkwiphwekaywqhcjze")
        data = ipfs_store.load(cid)
        print(data)
        ```
        """
        response = requests.get(
            f"{self.gateway_uri_stem}/ipfs/{str(id)}", timeout=self.timeout_seconds
        )

        return response.content
