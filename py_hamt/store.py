import io
from typing import Literal
from abc import ABC, abstractmethod
from dag_cbor.ipld import IPLDKind
from msgspec import json
from multiformats import multihash
from multiformats import CID
from multiformats.multihash import Multihash

import aiohttp

type CodecInput = Literal["raw", "dag-cbor"]


class Store(ABC):
    """Abstract class that represents a storage mechanism the HAMT can use for keeping data.

    The return type of save and input to load is really type IPLDKind, but the documentation generates something a bit strange since IPLDKind is a type union.
    """

    @abstractmethod
    async def save(self, data: bytes, codec: CodecInput) -> IPLDKind:
        """Take any set of bytes, save it to the storage mechanism, and reutrn an ID in the type of IPLDKind which can be used to retrieve those bytes later. This also includes extra information in `codec` on whether to mark this data as special linked data."""
        # TODO add bit about its important that dag-cbor is only called for an internal data structure node, and ONLY that type

    @abstractmethod
    async def load(self, id: IPLDKind) -> bytes:
        """Retrieve the bytes based on an ID returned earlier by the save function."""


# Inspired by https://github.com/rvagg/iamap/blob/master/examples/memory-backed.js
class DictStore(Store):
    """A basic implementation of a backing store, mostly for demonstration and testing purposes. It hashes all inputs and uses that as a key to an in-memory python dict. The hash bytes are the ID that `save` returns and `load` takes in."""

    store: dict[bytes, bytes]
    """@private"""
    hash_alg: Multihash
    """@private"""

    def __init__(self):
        self.store = dict()
        self.hash_alg = multihash.get("blake3")

    async def save(self, data: bytes, codec: CodecInput) -> bytes:
        hash = self.hash_alg.digest(data, size=32)
        self.store[hash] = data
        return hash

    # Ignore the type error since bytes is in the IPLDKind type
    async def load(self, id: bytes) -> bytes:  # type: ignore since bytes is a subset of the IPLDKind type
        if id in self.store:
            return self.store[id]
        else:
            raise Exception("ID not found in store")


class IPFSStore(Store):
    """
    Use IPFS as a backing store for a HAMT. The IDs returned from save and used by load are IPFS CIDs.

    Save methods use the RPC API but `load` uses the HTTP Gateway, so read-only HAMTs will only access the HTTP Gateway. This allows for connection to remote gateways as well.

    You can write to an authenticated IPFS node by providing credentials in the constructor. The following authentication methods are supported:
    - Basic Authentication: Provide a tuple of (username, password) to the `basic_auth` parameter.
    - Bearer Token: Provide a bearer token to the `bearer_token` parameter.
    - API Key: Provide an API key to the `api_key` parameter. You can customize the header name for the API key by setting the `api_key_header` parameter.
    """

    KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL = "http://127.0.0.1:8080"
    KUBO_DEFAULT_LOCAL_RPC_BASE_URL = "http://127.0.0.1:5001"

    # Take in a aiohttp session that can be reused across POSTs and GETs to a specific ipfs daemon, also allow for not doing that and creating our own single request lifetime session instead
    def __init__(
        self,
        gateway_session: aiohttp.ClientSession | None = None,
        rpc_session: aiohttp.ClientSession | None = None,
        hasher: str = "blake3",
    ):
        """
        TODO write about user needing to close the aiohttp session at the end of IPFSStore's use if this thing is being used in a long running program
        """
        # Don't initialize a self-managed ClientSession here, so that they are only created when needed, a lot of clients will only ever read from IPFS
        self.gateway_session = gateway_session
        self.rpc_session = rpc_session

        self.hasher = hasher
        """The hash function to send to IPFS when storing bytes."""

        self.rpc_url = f"/api/v0/add?hash={self.hasher}&pin=false"

        # TODO make this a future feature for IPFSStore after all the HAMT stuff is already done
        # self.debug: bool = debug
        # """If true, this records the total number of bytes sent in and out of IPFSStore to the network. You can access this information in `total_sent` and `total_received`. Bytes are counted in terms of either how much was sent to IPFS to store a CID, or how much data was inside of a retrieved IPFS block. This does not include the overhead of the HTTP requests themselves."""
        # self.total_sent: None | int = 0 if debug else None
        # """Total bytes sent to IPFS. Used for debugging purposes."""
        # self.total_received: None | int = 0 if debug else None
        # """Total bytes in responses from IPFS for blocks. Used for debugging purposes."""

    async def save(self, data: bytes, codec: CodecInput) -> CID:
        if self.rpc_session is None:
            self.rpc_session = aiohttp.ClientSession(
                base_url=IPFSStore.KUBO_DEFAULT_LOCAL_RPC_BASE_URL
            )

        async with self.rpc_session.request(
            method="POST", url=self.rpc_url, data={"file": io.BytesIO(data)}
        ) as response:
            response.raise_for_status()
            json_bytes = await response.read()
            cid_str: str = json.decode(json_bytes)["Hash"]
            cid = CID.decode(cid_str)
            # If it's dag-pb it means we should not reset the cid codec, since this is a UnixFS entry for a large amount of data that thus had to be sharded
            # 0x70 means dag-pb
            if cid.codec.code != 0x70:
                cid = cid.set(codec=codec)

            return cid

    async def load(  # type: ignore CID is definitely in the IPLDKind type
        self, id: CID
    ) -> bytes:
        if self.gateway_session is None:
            self.gateway_session = aiohttp.ClientSession(
                base_url=IPFSStore.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL
            )

        async with self.gateway_session.request(
            method="GET", url=f"/ipfs/{str(id)}"
        ) as response:
            response.raise_for_status()
            return await response.read()
