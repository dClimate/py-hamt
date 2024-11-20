from abc import ABC, abstractmethod
from dag_cbor.ipld import IPLDKind
import requests
from msgspec import json
from multiformats import multihash
from multiformats import CID
from multiformats.multihash import Multihash


class Store(ABC):
    """This is an Abstract Base Class that represents a storage mechanism the HAMT can use for keeping data.

    The return type of save and input to load is really type IPLDKind, but the documentation generates this strange type instead since IPLDKind is a type union.
    """

    @abstractmethod
    def save(self, data: bytes) -> IPLDKind:
        """Take any set of bytes, save it to the storage mechanism, and return an ID in the type of IPLDKind that can be used to retrieve those bytes later."""

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

    # Ignore the type error since bytes is in the IPLDKind type
    def load(self, id: bytes) -> bytes:  # type: ignore
        if id in self.store:
            return self.store[id]
        else:
            raise Exception("ID not found in store")


class IPFSStore(Store):
    """
    Use IPFS as a backing store for a HAMT. The IDs returned from save and used by load are IPFS CIDs.

    Save uses the RPC API but Load uses the HTTP Gateway, so read-only on HAMTs will only access HTTP Gateways.
    If only doing reads, then the RPC API will never be called.
    """

    def __init__(
        self,
        timeout_seconds=30,
        gateway_uri_stem="http://127.0.0.1:8080",
        rpc_uri_stem="http://127.0.0.1:5001",
        cid_codec="dag-cbor",
        mhtype="blake3",
        mhlen="32",
    ):
        self.timeout_seconds = timeout_seconds
        """
        You can modify this variable directly if you choose.

        This sets the timeout in seconds for an HTTP request in both `save` and `load`.
        """
        self.gateway_uri_stem = gateway_uri_stem
        """
        URI stem of the IPFS HTTP gateway that IPFSStore will retrieve blocks from.
        """
        self.rpc_uri_stem = rpc_uri_stem
        self.cid_codec = cid_codec
        """URI Stem of the IPFS RPC API that IPFSStore will send data to save to."""
        self.mhtype = mhtype
        """The Multihash hash function that the RPC requests send. This is used by the IPFS daemon to generate the CID."""
        self.mhlen = mhlen
        """The length of the hash. Necessary for some hash functions like blake3, which has variable hash size."""

    def save(self, data: bytes) -> CID:
        """
        This saves the data to an ipfs daemon by calling the RPC API, and then returns the CID. By default, `save` pins content it adds.

        To get and print the CID, do a decode.
        ```python
        from ipldstore import IPFSStore

        ipfs_store = IPFSStore()
        cid = ipfs_store.save("foo".encode())
        print(cid.human_readable)
        ```
        """
        rpc_response = requests.post(
            f"{self.rpc_uri_stem}/api/v0/block/put?cid-codec={self.cid_codec}&mhtype={self.mhtype}&mhlen={self.mhlen}&pin=true",
            files={"file": data},
        )

        cid_str: str = json.decode(rpc_response.content)["Key"]  # type: ignore
        cid = CID.decode(cid_str)

        return cid

    # Ignore the type error since CID is in the IPLDKind type
    def load(self, id: CID) -> bytes:  # type: ignore
        """
        This retrieves the raw bytes by calling the provided HTTP gateway.
        ```python
        from ipldstore import IPFSStore
        from multiformats import CID

        ipfs_store = IPFSStore()
        cid = CID.decode("bafybeiaysi4s6lnjev27ln5icwm6tueaw2vdykrtjkwiphwekaywqhcjze")
        data = ipfs_store.load(cid)
        print(data)
        ```
        """
        response = requests.get(
            f"{self.gateway_uri_stem}/ipfs/{str(id)}", timeout=self.timeout_seconds
        )

        return response.content
