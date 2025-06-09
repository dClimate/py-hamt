import asyncio
from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import aiohttp
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
        Mypy enforces this contra-variance rule and emits the “violates Liskov substitution principle” error.
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

    1. **Bring your own `aiohttp.ClientSession`**
       Pass it via `session=...` — any default headers or `BasicAuth`
       configured on that session are reused for **every** request.
    2. **Let `KuboCAS` build the session** but pass
       `headers=` *and*/or `auth=` kwargs; they are forwarded to the
       internally–created `ClientSession`.

    ```python
    import aiohttp
    from py_hamt import KuboCAS

    # Option 1: user-supplied session
    sess = aiohttp.ClientSession(
        headers={"Authorization": "Bearer <token>"},
        auth=aiohttp.BasicAuth("user", "pass"),
    )
    cas  = KuboCAS(session=sess)

    # Option 2: let KuboCAS create the session
    cas = KuboCAS(
        headers={"X-My-Header": "yes"},
        auth=aiohttp.BasicAuth("user", "pass"),
    )
    ```

    ### Parameters
    - **hasher** (str): multihash name (defaults to *blake3*).
    - **session** (`aiohttp.ClientSession | None`): reuse an existing
      session; if *None* KuboCAS will create one lazily.
    - **headers** (dict[str, str] | None): default headers for the
      internally-created session.
    - **auth** (`aiohttp.BasicAuth | None`): authentication object for
      the internally-created session.
    - **rpc_base_url / gateway_base_url** (str | None): override daemon
      endpoints (defaults match the local daemon ports).

    ...
    """

    KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL: str = "http://127.0.0.1:8080"
    KUBO_DEFAULT_LOCAL_RPC_BASE_URL: str = "http://127.0.0.1:5001"

    DAG_PB_MARKER: int = 0x70
    """@private"""

    # Take in a aiohttp session that can be reused across POSTs and GETs to a specific ipfs daemon, also allow for not doing that and creating our own single request lifetime session instead
    def __init__(
        self,
        hasher: str = "blake3",
        session: aiohttp.ClientSession | None = None,
        rpc_base_url: str | None = KUBO_DEFAULT_LOCAL_RPC_BASE_URL,
        gateway_base_url: str | None = KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL,
        concurrency: int = 32,
        *,
        headers: dict[str, str] | None = None,
        auth: aiohttp.BasicAuth | None = None,
    ):
        """
        If None is passed into the rpc or gateway base url, then the default for kubo local daemons will be used. The default local values will also be used if nothing is passed in at all.

        ### `aiohttp.ClientSession` Management
        If `session` is not provided, it will be automatically initialized. It is the responsibility of the user to close this at an appropriate time, using `await cas.aclose()`
        as a class instance cannot know when it will no longer be in use, unless explicitly told to do so.

        If you are using the `KuboCAS` instance in an `async with` block, it will automatically close the session when the block is exited which is what we suggest below:
        ```python
        async with aiohttp.ClientSession() as session, KuboCAS(
            rpc_base_url=rpc_base_url,
            gateway_base_url=gateway_base_url,
            session=session,
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
        Users can set whatever headers and auth credentials they need if they are connecting to an authenticated kubo instance by setting them in their own `aiohttp.ClientSession` and then passing that in.
        Alternatively, they can pass in `headers` and `auth` parameters to the constructor, which will be used to create a new `aiohttp.ClientSession` if one is not provided.
        If you do not need authentication, you can leave these parameters as `None`.

        ### RPC and HTTP Gateway Base URLs
        These are the first part of the url, defaults that refer to the default that kubo launches with on a local machine are provided.
        """

        self.hasher: str = hasher
        """The hash function to send to IPFS when storing bytes. Cannot be changed after initialization. The default blake3 follows the default hashing algorithm used by HAMT."""

        if rpc_base_url is None:
            rpc_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_RPC_BASE_URL
        if gateway_base_url is None:
            gateway_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL

        self.rpc_url: str = f"{rpc_base_url}/api/v0/add?hash={self.hasher}&pin=false"
        """@private"""
        self.gateway_base_url: str = f"{gateway_base_url}/ipfs/"
        """@private"""

        self._session_per_loop: dict[
            asyncio.AbstractEventLoop, aiohttp.ClientSession
        ] = {}

        if session is not None:
            # user supplied → bind it to *their* current loop
            self._session_per_loop[asyncio.get_running_loop()] = session
            self._owns_session: bool = False
        else:
            self._owns_session = True  # we’ll create sessions lazily

        # store for later use by _loop_session()
        self._default_headers = headers
        self._default_auth = auth

        self._sem: asyncio.Semaphore = asyncio.Semaphore(concurrency)

    # --------------------------------------------------------------------- #
    # helper: get or create the session bound to the current running loop   #
    # --------------------------------------------------------------------- #
    def _loop_session(self) -> aiohttp.ClientSession:
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            return self._session_per_loop[loop]
        except KeyError:
            sess = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=60),
                connector=aiohttp.TCPConnector(limit=64, limit_per_host=32),
                headers=self._default_headers,
                auth=self._default_auth,
            )
            self._session_per_loop[loop] = sess
            return sess

    # --------------------------------------------------------------------- #
    # graceful shutdown: close **all** sessions we own                      #
    # --------------------------------------------------------------------- #
    async def aclose(self) -> None:
        if self._owns_session:
            for sess in list(self._session_per_loop.values()):
                if not sess.closed:
                    await sess.close()

    async def __aenter__(self) -> "KuboCAS":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    # --------------------------------------------------------------------- #
    # save() – now uses the per-loop session                                #
    # --------------------------------------------------------------------- #
    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        async with self._sem:  # throttle RPC
            form = aiohttp.FormData()
            form.add_field(
                "file", data, filename="blob", content_type="application/octet-stream"
            )

            async with self._loop_session().post(self.rpc_url, data=form) as resp:
                resp.raise_for_status()
                cid_str: str = (await resp.json())["Hash"]

        cid: CID = CID.decode(cid_str)
        if cid.codec.code != self.DAG_PB_MARKER:
            cid = cid.set(codec=codec)
        return cid

    async def load(self, id: IPLDKind) -> bytes:
        """@private"""
        cid = cast(CID, id)  # CID is definitely in the IPLDKind type
        url: str = self.gateway_base_url + str(cid)
        async with self._sem:  # throttle gateway
            async with self._loop_session().get(url) as resp:
                resp.raise_for_status()
                return await resp.read()
