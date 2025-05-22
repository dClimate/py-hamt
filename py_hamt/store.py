from abc import ABC, abstractmethod
from dag_cbor.ipld import IPLDKind
import requests
from msgspec import json
from multiformats import multihash
from multiformats import CID
import aiohttp
import weakref
from multiformats.multihash import Multihash


class Store(ABC):
    """Abstract class that represents a storage mechanism the HAMT can use for keeping data.

    The return type of save and input to load is really type IPLDKind, but the documentation generates something a bit strange since IPLDKind is a type union.
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

    Save methods use the RPC API but `load` uses the HTTP Gateway, so read-only HAMTs will only access the HTTP Gateway. This allows for connection to remote gateways as well.

    You can write to an authenticated IPFS node by providing credentials in the constructor. The following authentication methods are supported:
    - Basic Authentication: Provide a tuple of (username, password) to the `basic_auth` parameter.
    - Bearer Token: Provide a bearer token to the `bearer_token` parameter.
    - API Key: Provide an API key to the `api_key` parameter. You can customize the header name for the API key by setting the `api_key_header` parameter.
    """

    def __init__(
        self,
        timeout_seconds: int = 30,
        gateway_uri_stem: str = "http://127.0.0.1:8080",
        rpc_uri_stem: str = "http://127.0.0.1:5001",
        hasher: str = "blake3",
<<<<<<< Updated upstream
        pin_on_add: bool = False,
        debug: bool = False,
        # Authentication parameters
        basic_auth: tuple[str, str] | None = None,  # (username, password)
        bearer_token: str | None = None,
        api_key: str | None = None,
        api_key_header: str = "X-API-Key",  # Customizable API key header
=======
        session: aiohttp.ClientSession | None = None,
        rpc_base_url=KUBO_DEFAULT_LOCAL_RPC_BASE_URL,
        gateway_base_url=KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL,
>>>>>>> Stashed changes
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
        self.pin_on_add: bool = pin_on_add
        """Whether IPFSStore should tell the daemon to pin the generated CIDs in API calls. This can be changed in between usage, but should be kept the same value for the lifetime of the program."""
        self.debug: bool = debug
        """If true, this records the total number of bytes sent in and out of IPFSStore to the network. You can access this information in `total_sent` and `total_received`. Bytes are counted in terms of either how much was sent to IPFS to store a CID, or how much data was inside of a retrieved IPFS block. This does not include the overhead of the HTTP requests themselves."""
        self.total_sent: None | int = 0 if debug else None
        """Total bytes sent to IPFS. Used for debugging purposes."""
        self.total_received: None | int = 0 if debug else None
        """Total bytes in responses from IPFS for blocks. Used for debugging purposes."""

        # Authentication settings
        self.basic_auth = basic_auth
        """Tuple of (username, password) for Basic Authentication"""
        self.bearer_token = bearer_token
        """Bearer token for token-based authentication"""
        self.api_key = api_key
        """API key for API key-based authentication"""
        self.api_key_header = api_key_header
        """Header name to use for API key authentication"""

<<<<<<< Updated upstream
    def save(self, data: bytes, cid_codec: str) -> CID:
        """
        This saves the data to an ipfs daemon by calling the RPC API, and then returns the CID, with a multicodec set by the input cid_codec. We need to do this since the API always returns either a multicodec of raw or dag-pb if it had to shard the input data.

        By default, `save` pins content it adds.

        ```python
        from py_hamt import IPFSStore

        ipfs_store = IPFSStore()
        cid = ipfs_store.save("foo".encode(), "raw")
        print(cid.human_readable)
        ```
        """
        pin_string: str = "true" if self.pin_on_add else "false"

        # Apply authentication based on provided credentials
        headers = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        elif self.api_key:
            headers[self.api_key_header] = self.api_key

        # Prepare request parameters
        url = f"{self.rpc_uri_stem}/api/v0/add?hash={self.hasher}&pin={pin_string}"

        # Make the request with appropriate authentication
        response = requests.post(
            url,
            files={"file": data},
            headers=headers,
            auth=self.basic_auth,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()

        cid_str: str = json.decode(response.content)["Hash"]  # type: ignore
        cid = CID.decode(cid_str)
        # If it's dag-pb it means we should not reset the cid codec, since this is a UnixFS entry for a large amount of data that thus had to be sharded
        # We don't worry about HAMT nodes being larger than 1 MB
        # with a conservative calculation of 256 map keys * 10 (bucket size of 9 and 1 link per map key)*100 bytes huge size for a cid=0.256 MB, so we can always safely recodec those as dag-cbor, which is what they are
        # 0x70 means dag-pb
        if cid.codec.code != 0x70:
            cid = cid.set(codec=cid_codec)

        # if everything is succesful, record debug information
        if self.debug:
            self.total_sent += len(data)  # type: ignore
=======
        concurrency = 32
        self._session = session or aiohttp.ClientSession()
        self._sem = asyncio.Semaphore(concurrency)
        self._external_session = session
        self._sessions: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, aiohttp.ClientSession]" = (
            weakref.WeakKeyDictionary()
        )

    async def _loop_session(self) -> aiohttp.ClientSession:
        """Return a ClientSession bound to *this* loop, lazily creating one."""
        loop = asyncio.get_running_loop()
        if self._external_session is not None:
            # caller manages lifetime; just return it
            return self._external_session                   # caller manages it
        try:
            return self._sessions[loop]                     # reuse if we made one
        except KeyError:
            sess = aiohttp.ClientSession()
            self._sessions[loop] = sess
            return sess
        
    async def aclose(self):
        if self._external_session is None:
            await asyncio.gather(*[s.close() for s in self._sessions.values()])

    async def __aenter__(self): return self
    async def __aexit__(self, *exc): await self.aclose()

    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        """@private"""
        async with self._sem:  # throttle RPC calls if needed
            session = await self._loop_session()          # your loop-local helper
            form = aiohttp.FormData()
            # field-name MUST be "file", filename can be anything
            form.add_field("file",
                        data,
                        filename="blob",
                        content_type="application/octet-stream")

            async with session.post(self.rpc_url, data=form) as resp:
                resp.raise_for_status()                   # no more 400s
                cid_str = (await resp.json())["Hash"]

            cid = CID.decode(cid_str)
            if cid.codec.code != self.DAG_PB_MARKER:
                cid = cid.set(codec=codec)
>>>>>>> Stashed changes

            return cid

<<<<<<< Updated upstream
    def save_raw(self, data: bytes) -> CID:
        """See `save`"""
        return self.save(data, "raw")

    def save_dag_cbor(self, data: bytes) -> CID:
        """See `save`"""
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
        response.raise_for_status()

        if self.debug:
            self.total_received += len(response.content)  # type: ignore

        return response.content
=======
    async def load(  # type: ignore CID is definitely in the IPLDKind type
        self, id: CID
    ) -> bytes:
        """@private"""
        url = self.gateway_base_url + str(id)
        # If this coroutine is running inside an event loop,         #
        # off-load the *blocking* requests call to a worker thread.  #
        async with self._sem:                         # throttle gateway
            async with (await self._loop_session()).get(self.gateway_base_url + str(id)) as r:
                r.raise_for_status()
                return await r.read()
>>>>>>> Stashed changes
