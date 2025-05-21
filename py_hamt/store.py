import asyncio
from typing import Literal
from abc import ABC, abstractmethod
from dag_cbor.ipld import IPLDKind
from msgspec import json
from multiformats import multihash
from multiformats import CID
from multiformats.multihash import Multihash
import requests


class ContentAddressedStore(ABC):
    """
    Abstract class that represents a content addressed storage that the `HAMT` can use for keeping data.

    Note that the return type of save and input to load is really type `IPLDKind`, but the documentation generator pdoc mangles it unfortunately.

    #### A note on the IPLDKind return types
    Save and load return the type IPLDKind and not just a CID. As long as python regards the underlying type as immutable it can be used, allowing for more flexibility. There are two exceptions:
    1. No lists or dicts, since python does not classify these as immutable.
    2. No `None` values since this is used in HAMT's `__init__` to indicate that an empty HAMT needs to be initialized.
    """

    type CodecInput = Literal["raw", "dag-cbor"]

    @abstractmethod
    async def save(self, data: bytes, codec: CodecInput) -> IPLDKind:
        """Save data to a storage mechanism, and return an ID for the data in the IPLDKind type.

        `codec` will be set to "dag-cbor" if this data should be marked as special linked data a la IPLD data model."""

    @abstractmethod
    async def load(self, id: IPLDKind) -> bytes:
        """Retrieve data."""


class InMemoryCAS(ContentAddressedStore):
    """Used mostly for faster testing, this is why this is not exported. It hashes all inputs and uses that as a key to an in-memory python dict, mimicking a content addressed storage system. The hash bytes are the ID that `save` returns and `load` takes in."""

    store: dict[bytes, bytes]
    hash_alg: Multihash

    def __init__(self):
        self.store = dict()
        self.hash_alg = multihash.get("blake3")

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> bytes:
        hash = self.hash_alg.digest(data, size=32)
        self.store[hash] = data
        return hash

    async def load(self, id: bytes) -> bytes:  # type: ignore since bytes is a subset of the IPLDKind type
        if id in self.store:
            return self.store[id]

        raise KeyError


class KuboCAS(ContentAddressedStore):
    """
    Talks to a kubo daemon. The IDs in save and load are IPLD CIDs.

    `save` uses the RPC API and `load` uses the HTTP Gateway. This means that read-only HAMTs will only access the HTTP Gateway, so no RPC endpoint is required for use.
    """

    KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL = "http://127.0.0.1:8080"
    KUBO_DEFAULT_LOCAL_RPC_BASE_URL = "http://127.0.0.1:5001"

    DAG_PB_MARKER = 0x70
    """@private"""

    # Take in a aiohttp session that can be reused across POSTs and GETs to a specific ipfs daemon, also allow for not doing that and creating our own single request lifetime session instead
    def __init__(
        self,
        hasher: str = "blake3",
        requests_session: requests.Session | None = None,
        rpc_base_url: str | None = KUBO_DEFAULT_LOCAL_RPC_BASE_URL,
        gateway_base_url: str | None = KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL,
    ):
        """
        If None is passed into the rpc or gateway base url, then the default for kubo local daemons will be used. The default local values will also be used if nothing is passed in at all.

        ### `requests.Session` Management
        If `requests_session` is not provided, it will be automatically initialized. It is the responsibility of the user to close this at an appropriate time, as a class instance cannot know when it will no longer be in use.

        ### Authenticated RPC/Gateway Access
        Users can set whatever headers and auth credentials they need if they are connecting to an authenticated kubo instance by setting them in their own `requests.Session` and then passing that in.

        ### RPC and HTTP Gateway Base URLs
        These are the first part of the url, defaults that refer to the default that kubo launches with on a local machine are provided.
        """

        self.hasher = hasher
        """The hash function to send to IPFS when storing bytes. Cannot be changed after initialization. The default blake3 follows the default hashing algorithm used by HAMT."""

        if rpc_base_url is None:
            rpc_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_RPC_BASE_URL
        if gateway_base_url is None:
            gateway_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL

        self.rpc_url = f"{rpc_base_url}/api/v0/add?hash={self.hasher}&pin=false"
        """@private"""
        self.gateway_base_url = gateway_base_url + "/ipfs/"
        """@private"""

        self.requests_session: requests.Session
        """@private"""
        if requests_session is None:
            self.requests_session = requests.Session()
        else:
            self.requests_session = requests_session

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        """@private"""
        response = await asyncio.to_thread(
            self.requests_session.post, self.rpc_url, files={"file": data}
        )
        response.raise_for_status()

        cid_str: str = json.decode(response.content)["Hash"]  # type: ignore
        cid = CID.decode(cid_str)

        # If it's dag-pb it means we should not reset the cid codec, kubo shards large data into a UnixFS structure
        if cid.codec.code != KuboCAS.DAG_PB_MARKER:
            cid = cid.set(codec=codec)

        return cid

    async def load(  # type: ignore CID is definitely in the IPLDKind type
        self, id: CID
    ) -> bytes:
        """@private"""
        url = self.gateway_base_url + str(id)
        response = await asyncio.to_thread(self.requests_session.get, url)
        response.raise_for_status()
        return response.content
