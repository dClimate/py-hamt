import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Tuple, cast

import httpx
from dag_cbor.ipld import IPLDKind
from multiformats import CID, multihash
from multiformats.multihash import Multihash


class ContentAddressedStore(ABC):
    """
    Abstract class that represents a content addressed storage that the `HAMT` can use for keeping data.

    Note that the return type of save and input to load is really type `IPLDKind`, but the documentation generator pdoc mangles it unfortunately.

    #### A note on the IPLDKind return types
    Save and load return the type IPLDKind and not just a CID. As long as python regards the underlying type as immutable it can be used, allowing for more flexibility. There are two exceptions:
    1. No lists or dicts, since python does not classify these as immutable.
    2. No `None` values since this is used in HAMT's `__init__` to indicate that an empty HAMT needs to be initialized.
    """

    CodecInput = Literal["raw", "dag-cbor"]

    @abstractmethod
    async def save(self, data: bytes, codec: CodecInput) -> IPLDKind:
        """Save data to a storage mechanism, and return an ID for the data in the IPLDKind type.

        `codec` will be set to "dag-cbor" if this data should be marked as special linked data a la IPLD data model.
        """

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
        hash: bytes = self.hash_alg.digest(data, size=32)
        self.store[hash] = data
        return hash

    async def load(self, id: IPLDKind) -> bytes:
        """
        `ContentAddressedStore` allows any IPLD scalar key.  For the in-memory
        backend we *require* a `bytes` hash; anything else is rejected at run
        time. In OO type-checking, a subclass may widen (make more general) argument types,
        but it must never narrow them; otherwise callers that expect the base-class contract can break.
        Mypy enforces this contra-variance rule and emits the "violates Liskov substitution principle" error.
        This is why we use `cast` here, to tell mypy that we know what we are doing.
        h/t https://stackoverflow.com/questions/75209249/overriding-a-method-mypy-throws-an-incompatible-with-super-type-error-when-ch
        """
        key = cast(bytes, id)
        if not isinstance(key, (bytes, bytearray)):  # defensive guard
            raise TypeError(
                f"InMemoryCAS only supports byte‐hash keys; got {type(id).__name__}"
            )

        try:
            return self.store[key]
        except KeyError as exc:
            raise KeyError("Object not found in in-memory store") from exc


class KuboCAS(ContentAddressedStore):
    """
    Connects to an **IPFS Kubo** daemon.

    The IDs in save and load are IPLD CIDs.

    * **save()**  → RPC  (`/api/v0/add`)
    * **load()**  → HTTP gateway  (`/ipfs/{cid}`)

    `save` uses the RPC API and `load` uses the HTTP Gateway. This means that read-only HAMTs will only access the HTTP Gateway, so no RPC endpoint is required for use.

    ### Authentication / custom headers
    You have two options:

    1. **Bring your own `httpx.AsyncClient`**
       Pass it via `client=...` — any default headers or auth
       configured on that client are reused for **every** request.
    2. **Let `KuboCAS` build the client** but pass
       `headers=` *and*/or `auth=` kwargs; they are forwarded to the
       internally–created `AsyncClient`.

    ```python
    import httpx
    from py_hamt import KuboCAS

    # Option 1: user-supplied client
    client = httpx.AsyncClient(
        headers={"Authorization": "Bearer <token>"},
        auth=("user", "pass"),
    )
    cas = KuboCAS(client=client)

    # Option 2: let KuboCAS create the client
    cas = KuboCAS(
        headers={"X-My-Header": "yes"},
        auth=("user", "pass"),
    )
    ```

    ### Parameters
    - **hasher** (str): multihash name (defaults to *blake3*).
    - **client** (`httpx.AsyncClient | None`): reuse an existing
      client; if *None* KuboCAS will create one lazily.
    - **headers** (dict[str, str] | None): default headers for the
      internally-created client.
    - **auth** (`tuple[str, str] | None`): authentication tuple (username, password)
      for the internally-created client.
    - **rpc_base_url / gateway_base_url** (str | None): override daemon
      endpoints (defaults match the local daemon ports).

    ...
    """

    KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL: str = "http://127.0.0.1:8080"
    KUBO_DEFAULT_LOCAL_RPC_BASE_URL: str = "http://127.0.0.1:5001"

    DAG_PB_MARKER: int = 0x70
    """@private"""

    # Take in a httpx client that can be reused across POSTs and GETs to a specific IPFS daemon
    def __init__(
        self,
        hasher: str = "blake3",
        client: httpx.AsyncClient | None = None,
        rpc_base_url: str | None = None,
        gateway_base_url: str | None = None,
        concurrency: int = 32,
        *,
        headers: dict[str, str] | None = None,
        auth: Tuple[str, str] | None = None,
    ):
        """
        If None is passed into the rpc or gateway base url, then the default for kubo local daemons will be used. The default local values will also be used if nothing is passed in at all.

        ### `httpx.AsyncClient` Management
        If `client` is not provided, it will be automatically initialized. It is the responsibility of the user to close this at an appropriate time, using `await cas.aclose()`
        as a class instance cannot know when it will no longer be in use, unless explicitly told to do so.

        If you are using the `KuboCAS` instance in an `async with` block, it will automatically close the client when the block is exited which is what we suggest below:
        ```python
        async with httpx.AsyncClient() as client, KuboCAS(
            rpc_base_url=rpc_base_url,
            gateway_base_url=gateway_base_url,
            client=client,
        ) as kubo_cas:
            hamt = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
            zhs = ZarrHAMTStore(hamt)
            # Use the KuboCAS instance as needed
            # ...
        ```
        As mentioned, if you do not use the `async with` syntax, you should call `await cas.aclose()` when you are done using the instance to ensure that all resources are cleaned up.
        ``` python
        cas = KuboCAS(rpc_base_url=rpc_base_url, gateway_base_url=gateway_base_url)
        # Use the KuboCAS instance as needed
        # ...
        await cas.aclose()  # Ensure resources are cleaned up
        ```

        ### Authenticated RPC/Gateway Access
        Users can set whatever headers and auth credentials they need if they are connecting to an authenticated kubo instance by setting them in their own `httpx.AsyncClient` and then passing that in.
        Alternatively, they can pass in `headers` and `auth` parameters to the constructor, which will be used to create a new `httpx.AsyncClient` if one is not provided.
        If you do not need authentication, you can leave these parameters as `None`.

        ### RPC and HTTP Gateway Base URLs
        These are the first part of the url, defaults that refer to the default that kubo launches with on a local machine are provided.
        """

        self.hasher: str = hasher
        """The hash function to send to IPFS when storing bytes. Cannot be changed after initialization. The default blake3 follows the default hashing algorithm used by HAMT."""

        if rpc_base_url is None:
            rpc_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_RPC_BASE_URL  # pragma
        if gateway_base_url is None:
            gateway_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL

        if "/ipfs/" in gateway_base_url:
            gateway_base_url = gateway_base_url.split("/ipfs/")[0]

        # Standard gateway URL construction with proper path handling
        if gateway_base_url.endswith("/"):
            gateway_base_url = f"{gateway_base_url}ipfs/"
        else:
            gateway_base_url = f"{gateway_base_url}/ipfs/"

        self.rpc_url: str = f"{rpc_base_url}/api/v0/add?hash={self.hasher}&pin=false"
        """@private"""
        self.gateway_base_url: str = gateway_base_url
        """@private"""

        self._client_per_loop: Dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}

        if client is not None:
            # user supplied → bind it to *their* current loop
            self._client_per_loop[asyncio.get_running_loop()] = client
            self._owns_client: bool = False
        else:
            self._owns_client = True  # we'll create clients lazily

        # store for later use by _loop_client()
        self._default_headers = headers
        self._default_auth = auth

        self._sem: asyncio.Semaphore = asyncio.Semaphore(concurrency)
        self._closed: bool = False

    # --------------------------------------------------------------------- #
    # helper: get or create the client bound to the current running loop    #
    # --------------------------------------------------------------------- #
    def _loop_client(self) -> httpx.AsyncClient:
        """Get or create a client for the current event loop."""
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            return self._client_per_loop[loop]
        except KeyError:
            # Create a new client
            client = httpx.AsyncClient(
                timeout=60.0,
                headers=self._default_headers,
                auth=self._default_auth,
                limits=httpx.Limits(max_connections=64, max_keepalive_connections=32),
                # Uncomment when they finally support Robost HTTP/2 GOAWAY responses
                # http2=True,
            )
            self._client_per_loop[loop] = client
            return client

    # --------------------------------------------------------------------- #
    # graceful shutdown: close **all** clients we own                       #
    # --------------------------------------------------------------------- #
    async def aclose(self) -> None:
        """Close all internally-created clients."""
        if not self._owns_client:
            # User supplied the client; they are responsible for closing it.
            return

        for client in list(self._client_per_loop.values()):
            if not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    # Best-effort cleanup; ignore errors during shutdown
                    pass

        self._client_per_loop.clear()
        self._closed = True

    # At this point, _client_per_loop should be empty or only contain
    # clients from loops we haven't seen (which shouldn't happen in practice)
    async def __aenter__(self) -> "KuboCAS":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """Best-effort close for internally-created clients."""
        if not self._owns_client or self._closed:
            return

        # Attempt proper cleanup if possible
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        try:
            if loop is None or not loop.is_running():
                asyncio.run(self.aclose())
            else:
                loop.create_task(self.aclose())
        except Exception:
            # Suppress all errors during interpreter shutdown or loop teardown
            pass

    # --------------------------------------------------------------------- #
    # save() – now uses the per-loop client                                 #
    # --------------------------------------------------------------------- #
    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        async with self._sem:  # throttle RPC
            # Create multipart form data
            files = {"file": data}

            # Send the POST request
            client = self._loop_client()
            response = await client.post(self.rpc_url, files=files)
            response.raise_for_status()
            cid_str: str = response.json()["Hash"]

        cid: CID = CID.decode(cid_str)
        if cid.codec.code != self.DAG_PB_MARKER:
            cid = cid.set(codec=codec)
        return cid

    async def load(self, id: IPLDKind) -> bytes:
        """@private"""
        cid = cast(CID, id)  # CID is definitely in the IPLDKind type
        url: str = f"{self.gateway_base_url + str(cid)}"

        async with self._sem:  # throttle gateway
            client = self._loop_client()
            response = await client.get(url)
            response.raise_for_status()
            return response.content
