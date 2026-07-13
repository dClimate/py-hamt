import asyncio
import logging
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, Literal, Optional, Tuple, cast

import httpx
from dag_cbor.ipld import IPLDKind
from multiformats import CID, multihash
from multiformats.multihash import Multihash

from . import instrumentation

logger = logging.getLogger(__name__)


def _slice_requested_range(
    data: bytes,
    offset: Optional[int],
    length: Optional[int],
    suffix: Optional[int],
) -> bytes:
    """Apply content-store range arguments to a complete object body."""
    if offset is not None:
        if length is not None:
            return data[offset : offset + length]
        return data[offset:]
    if suffix is not None:
        if suffix == 0:
            return b""
        return data[-suffix:]
    return data


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
    async def load(
        self,
        id: IPLDKind,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        suffix: Optional[int] = None,
    ) -> bytes:
        """Retrieve data."""

    async def pin_cid(self, id: IPLDKind, target_rpc: str) -> None:
        """Pin a CID in the storage."""
        pass  # pragma: no cover

    async def unpin_cid(self, id: IPLDKind, target_rpc: str) -> None:
        """Unpin a CID in the storage."""
        pass  # pragma: no cover

    async def pin_update(
        self, old_id: IPLDKind, new_id: IPLDKind, target_rpc: str
    ) -> None:
        """Update the pinned CID in the storage."""
        pass  # pragma: no cover

    async def pin_ls(self, target_rpc: str) -> list[Dict[str, Any]]:
        """List all pinned CIDs in the storage."""
        return []  # pragma: no cover


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

    async def load(
        self,
        id: IPLDKind,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        suffix: Optional[int] = None,
    ) -> bytes:
        """
        Retrieve all or part of an object using Python slice semantics.

        A zero ``length`` or ``suffix`` returns an empty byte string.

        `ContentAddressedStore` allows any IPLD scalar key.  For the in-memory
        backend we *require* a `bytes` hash; anything else is rejected at run
        time. In OO type-checking, a subclass may widen (make more general) argument types,
        but it must never narrow them; otherwise callers that expect the base-class contract can break.
        Mypy enforces this contra-variance rule and emits the "violates Liskov substitution principle" error.
        This is why we use `cast` here, to tell mypy that we know what we are doing.
        h/t https://stackoverflow.com/questions/75209249/overriding-a-method-mypy-throws-an-incompatible-with-super-type-error-when-ch
        """
        if (offset is not None and length == 0) or (offset is None and suffix == 0):
            return b""

        key = cast(bytes, id)
        if not isinstance(key, (bytes, bytearray)):  # defensive guard
            raise TypeError(
                f"InMemoryCAS only supports byte‐hash keys; got {type(id).__name__}"
            )
        data: bytes
        try:
            data = self.store[key]
        except KeyError as exc:
            raise KeyError("Object not found in in-memory store") from exc

        return _slice_requested_range(data, offset, length, suffix)


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
        follow_redirects=True,
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
      client and its configured timeout and redirect policy. User-supplied
      clients should set ``follow_redirects=True`` when gateways may redirect.
      If *None*, KuboCAS will create one lazily with a 60-second timeout and
      redirect following enabled.
    - **headers** (dict[str, str] | None): default headers for the
      internally-created client.
    - **auth** (`tuple[str, str] | None`): authentication tuple (username, password)
      for the internally-created client.
    - **rpc_base_url / gateway_base_url** (str | None): override daemon
      endpoints (defaults match the local daemon ports). Gateway URLs may end
      with `/ipfs` and may include a trailing slash.
    - **chunker** (str): chunking algorithm specification for Kubo's `add`
      RPC. Accepted formats are `"size-<positive int>"`, `"rabin"`, or
      `"rabin-<min>-<avg>-<max>"`.

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
        pin_on_add: bool = False,
        chunker: str = "size-1048576",
        max_retries: int = 3,
        initial_delay: float = 1.0,
        backoff_factor: float = 2.0,
    ):
        """
        If None is passed into the rpc or gateway base url, then the default for kubo local daemons will be used. The default local values will also be used if nothing is passed in at all.

        ### `httpx.AsyncClient` Management
        If `client` is not provided, it will be automatically initialized. It is the responsibility of the user to close this at an appropriate time, using `await cas.aclose()`
        as a class instance cannot know when it will no longer be in use, unless explicitly told to do so.

        A supplied client is associated with the running event loop lazily on
        first use, so constructing ``KuboCAS`` does not require an async context.
        Its configured timeout is respected. Clients created internally by
        ``KuboCAS`` use a 60-second timeout and follow redirects. Supplied
        clients retain their own redirect policy and should be configured with
        ``follow_redirects=True`` when gateways may redirect.

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

        self._owns_client: bool = False
        self._closed: bool = True
        self._client_per_loop: Dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self._internally_created_clients: set[httpx.AsyncClient] = set()
        self._semaphore_per_loop: Dict[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = {}

        # Now, perform validation that might raise an exception
        chunker_pattern = r"(?:size-[1-9]\d*|rabin(?:-[1-9]\d*-[1-9]\d*-[1-9]\d*)?)"
        if re.fullmatch(chunker_pattern, chunker) is None:
            raise ValueError("Invalid chunker specification")
        self.chunker: str = chunker

        self.hasher: str = hasher
        """The hash function to send to IPFS when storing bytes. Cannot be changed after initialization. The default blake3 follows the default hashing algorithm used by HAMT."""

        if rpc_base_url is None:
            rpc_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_RPC_BASE_URL  # pragma
        if gateway_base_url is None:
            gateway_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL

        gateway_base_url = gateway_base_url.rstrip("/")
        if not gateway_base_url.endswith("/ipfs"):
            gateway_base_url = f"{gateway_base_url}/ipfs"
        gateway_base_url = f"{gateway_base_url}/"

        pin_string: str = "true" if pin_on_add else "false"
        self.rpc_url: str = f"{rpc_base_url}/api/v0/add?hash={self.hasher}&chunker={self.chunker}&pin={pin_string}"
        """@private"""
        self.gateway_base_url: str = gateway_base_url
        """@private"""

        if client is not None:
            # Bind the user-supplied client lazily on first async use.
            self._owns_client = False
            self._supplied_client: httpx.AsyncClient | None = client
            self._user_client: httpx.AsyncClient | None = client
            self._default_headers: httpx.Headers | dict[str, str] | None = (
                httpx.Headers(client.headers)
            )
            self._default_auth: httpx.Auth | Tuple[str, str] | None = client.auth
            self._default_timeout: httpx.Timeout | float = client.timeout
            self._default_limits = self._copy_client_limits(client)
        else:
            # No client supplied. We will own any clients we create.
            self._owns_client = True
            self._supplied_client = None
            self._user_client = None
            self._default_headers = headers
            self._default_auth = auth
            self._default_timeout = 60.0
            self._default_limits = httpx.Limits(
                max_connections=64, max_keepalive_connections=32
            )

        if concurrency < 0:
            raise ValueError("Semaphore initial value must be >= 0")
        self._concurrency: int = concurrency
        self._closed = False

        # Validate retry parameters
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if initial_delay <= 0:
            raise ValueError("initial_delay must be positive")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0 for exponential backoff")

        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    @staticmethod
    def _copy_client_limits(client: httpx.AsyncClient) -> httpx.Limits:
        """Copy connection limits from a standard HTTPX async transport.

        HTTPX does not expose client limits publicly, so custom transports fall
        back to the limits KuboCAS uses for its own clients.
        """
        transport: Any = client._transport
        pool: Any = getattr(transport, "_pool", None)
        return httpx.Limits(
            max_connections=getattr(pool, "_max_connections", 64),
            max_keepalive_connections=getattr(pool, "_max_keepalive_connections", 32),
            keepalive_expiry=getattr(pool, "_keepalive_expiry", 5.0),
        )

    # --------------------------------------------------------------------- #
    # helper: get or create the client bound to the current running loop    #
    # --------------------------------------------------------------------- #
    def _loop_semaphore(self) -> asyncio.Semaphore:
        """Get or create the concurrency semaphore for the running event loop.

        Semaphores cannot be shared safely across event loops once contended,
        so their lifecycle mirrors the per-loop HTTP clients.
        """
        if self._closed:
            if not self._owns_client:
                raise RuntimeError("KuboCAS is closed; create a new instance")
            self._closed = False
            self._client_per_loop = {}
            self._internally_created_clients = set()
            self._semaphore_per_loop = {}

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            return self._semaphore_per_loop[loop]
        except KeyError:
            semaphore = asyncio.Semaphore(self._concurrency)
            self._semaphore_per_loop[loop] = semaphore
            return semaphore

    def _loop_client(self) -> httpx.AsyncClient:
        """Get or create a client for the current event loop.

        A user-supplied client is bound to the first loop that requests it.
        If the instance was previously closed but owns its clients, a fresh
        client mapping is lazily created on demand. Users that supplied their
        own ``httpx.AsyncClient`` still receive an error when the instance has
        been closed, as we cannot safely recreate their client.
        """
        if self._closed:
            if not self._owns_client:
                raise RuntimeError("KuboCAS is closed; create a new instance")
            # We previously closed all internally-owned clients. Reset the
            # state so that new clients can be created lazily.
            self._closed = False
            self._client_per_loop = {}
            self._semaphore_per_loop = {}

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            return self._client_per_loop[loop]
        except KeyError:
            if self._supplied_client is not None:
                client = self._supplied_client
                self._supplied_client = None
            else:
                client = httpx.AsyncClient(
                    timeout=self._default_timeout,
                    headers=self._default_headers,
                    auth=self._default_auth,
                    limits=self._default_limits,
                    follow_redirects=True,
                    # Uncomment when they finally support Robust HTTP/2 GOAWAY responses
                    # http2=True,
                )
                self._internally_created_clients.add(client)
            self._client_per_loop[loop] = client
            return client

    # --------------------------------------------------------------------- #
    # graceful shutdown: close **all** clients we own                       #
    # --------------------------------------------------------------------- #
    async def aclose(self) -> None:
        """
        Close every internally-created client, leaving a supplied client open.

        Must be called from an async context.
        """
        # This method is async, so we can reliably await the async close method.
        # The complex sync/async logic is handled by __del__.
        for client in list(self._internally_created_clients):
            if not client.is_closed:
                try:
                    await client.aclose()
                except Exception:
                    pass  # best-effort cleanup

        self._client_per_loop.clear()
        self._internally_created_clients.clear()
        self._semaphore_per_loop.clear()
        self._closed = True

    # At this point, _client_per_loop should be empty or only contain
    # clients from loops we haven't seen (which shouldn't happen in practice)
    async def __aenter__(self) -> "KuboCAS":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.aclose()

    def __del__(self) -> None:
        """Best-effort close for internally-created clients."""
        if not hasattr(self, "_owns_client") or not hasattr(self, "_closed"):
            return

        if (
            not self._owns_client
            and not getattr(self, "_internally_created_clients", set())
        ) or self._closed:
            return

        # Attempt proper cleanup if possible
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop - can't do async cleanup
            # Just clear the client references synchronously
            if hasattr(self, "_client_per_loop"):
                # We can't await client.aclose() without a loop,
                # so just clear the references
                self._client_per_loop.clear()
                self._semaphore_per_loop.clear()
                self._closed = True
            return

        # If we get here, we have a running loop
        try:
            if loop.is_running():
                # Schedule cleanup in the existing loop
                loop.create_task(self.aclose())
            else:
                # Loop exists but not running - try asyncio.run
                coro = self.aclose()  # Create the coroutine
                try:
                    asyncio.run(coro)
                except Exception:
                    # If asyncio.run fails, we need to close the coroutine properly
                    coro.close()  # This prevents the RuntimeWarning
                    raise  # Re-raise to hit the outer except block
        except Exception:
            # If all else fails, just clear references
            if hasattr(self, "_client_per_loop"):
                self._client_per_loop.clear()
                self._semaphore_per_loop.clear()
                self._closed = True

    # --------------------------------------------------------------------- #
    # save() – now uses the per-loop client                                 #
    # --------------------------------------------------------------------- #
    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        async with self._loop_semaphore():
            files = {"file": data}
            client = self._loop_client()
            retry_count = 0

            while retry_count <= self.max_retries:
                try:
                    response = await client.post(self.rpc_url, files=files)
                    response.raise_for_status()
                    cid_str: str = response.json()["Hash"]
                    cid: CID = CID.decode(cid_str)
                    if cid.codec.code != self.DAG_PB_MARKER:
                        cid = cid.set(codec=codec)
                    return cid

                except (httpx.TimeoutException, httpx.RequestError) as e:
                    retry_count += 1
                    if retry_count > self.max_retries:
                        raise httpx.TimeoutException(
                            f"Failed to save data after {self.max_retries} retries: {str(e)}",
                            request=e.request
                            if isinstance(e, httpx.RequestError)
                            else None,
                        )

                    # Calculate backoff delay
                    delay = self.initial_delay * (
                        self.backoff_factor ** (retry_count - 1)
                    )
                    # Add some jitter to prevent thundering herd
                    jitter = delay * 0.1 * (random.random() - 0.5)
                    await asyncio.sleep(delay + jitter)

                except httpx.HTTPStatusError:
                    # Re-raise non-timeout HTTP errors immediately
                    raise
        raise RuntimeError("Exited the retry loop unexpectedly.")  # pragma: no cover

    async def load(
        self,
        id: IPLDKind,
        offset: Optional[int] = None,
        length: Optional[int] = None,
        suffix: Optional[int] = None,
    ) -> bytes:
        """Load all or part of a CID using the IPFS gateway.

        Gateways that ignore a Range header and return a complete ``200`` body
        are handled by applying the requested byte window locally. Zero-length
        and zero-suffix reads return immediately without a gateway request.
        """
        if (offset is not None and length == 0) or (offset is None and suffix == 0):
            return b""

        cid = cast(CID, id)
        url: str = f"{self.gateway_base_url + str(cid)}"
        headers: Dict[str, str] = {}

        # Construct the Range header if required
        if offset is not None:
            start = offset
            if length is not None:
                # Standard HTTP Range: bytes=start-end (inclusive)
                end = start + length - 1
                headers["Range"] = f"bytes={start}-{end}"
            else:
                # Standard HTTP Range: bytes=start- (from start to end)
                headers["Range"] = f"bytes={start}-"
        elif suffix is not None:
            # Standard HTTP Range: bytes=-N (last N bytes)
            headers["Range"] = f"bytes=-{suffix}"

        trace_started_at = instrumentation.begin_cas_load(cid, bool(headers))
        response_bytes = 0
        final_status = "ok"
        final_retry_count = 0
        try:
            async with self._loop_semaphore():  # Throttle gateway
                client = self._loop_client()
                retry_count = 0

                while retry_count <= self.max_retries:
                    try:
                        response = await client.get(url, headers=headers or None)
                        response.raise_for_status()
                        content = response.content
                        response_bytes = len(content)
                        final_retry_count = retry_count
                        if headers and response.status_code == httpx.codes.OK:
                            logger.debug(
                                "Gateway ignored Range request for CID %s; "
                                "slicing the complete response locally",
                                cid,
                            )
                            return _slice_requested_range(
                                content, offset, length, suffix
                            )
                        return content

                    except (httpx.TimeoutException, httpx.RequestError) as e:
                        retry_count += 1
                        if retry_count > self.max_retries:
                            final_status = "timeout"
                            final_retry_count = retry_count
                            raise httpx.TimeoutException(
                                f"Failed to load data after {self.max_retries} retries: {str(e)}",
                                request=e.request
                                if isinstance(e, httpx.RequestError)
                                else None,
                            )

                        # Calculate backoff delay with jitter
                        delay = self.initial_delay * (
                            self.backoff_factor ** (retry_count - 1)
                        )
                        jitter = delay * 0.1 * (random.random() - 0.5)
                        await asyncio.sleep(delay + jitter)

                    except httpx.HTTPStatusError:
                        # Re-raise non-timeout HTTP errors immediately
                        final_status = "http_error"
                        final_retry_count = retry_count
                        raise
        finally:
            instrumentation.end_cas_load(
                trace_started_at,
                byte_count=response_bytes,
                retries=final_retry_count,
                status=final_status,
            )
        raise RuntimeError("Exited the retry loop unexpectedly.")  # pragma: no cover

    # --------------------------------------------------------------------- #
    # pin_cid() – method to pin a CID                                       #
    # --------------------------------------------------------------------- #
    async def pin_cid(
        self,
        cid: CID,
        target_rpc: str = "http://127.0.0.1:5001",
    ) -> None:
        """
        Pins a CID to the local Kubo node via the RPC API.

        This call is recursive by default, pinning all linked objects.

        Args:
            cid (CID): The Content ID to pin.
            target_rpc (str): The RPC URL of the Kubo node.
        """
        params = {"arg": str(cid), "recursive": "true"}
        pin_add_url_base: str = f"{target_rpc}/api/v0/pin/add"

        async with self._loop_semaphore():  # throttle RPC
            client = self._loop_client()
            response = await client.post(pin_add_url_base, params=params)
            response.raise_for_status()

    async def unpin_cid(
        self, cid: CID, target_rpc: str = "http://127.0.0.1:5001"
    ) -> None:
        """
        Unpins a CID from the local Kubo node via the RPC API.

        Args:
            cid (CID): The Content ID to unpin.
        """
        params = {"arg": str(cid), "recursive": "true"}
        unpin_url_base: str = f"{target_rpc}/api/v0/pin/rm"
        async with self._loop_semaphore():  # throttle RPC
            client = self._loop_client()
            response = await client.post(unpin_url_base, params=params)
            response.raise_for_status()

    async def pin_update(
        self,
        old_id: IPLDKind,
        new_id: IPLDKind,
        target_rpc: str = "http://127.0.0.1:5001",
    ) -> None:
        """
        Updates the pinned CID in the storage.

        Args:
            old_id (IPLDKind): The old Content ID to replace.
            new_id (IPLDKind): The new Content ID to pin.
        """
        params = {"arg": [str(old_id), str(new_id)]}
        pin_update_url_base: str = f"{target_rpc}/api/v0/pin/update"
        async with self._loop_semaphore():  # throttle RPC
            client = self._loop_client()
            response = await client.post(pin_update_url_base, params=params)
            response.raise_for_status()

    async def pin_ls(
        self, target_rpc: str = "http://127.0.0.1:5001"
    ) -> list[Dict[str, Any]]:
        """
        Lists all pinned CIDs on the local Kubo node via the RPC API.

        Args:
            target_rpc (str): The RPC URL of the Kubo node.

        Returns:
            List[CID]: A list of pinned CIDs.
        """
        pin_ls_url_base: str = f"{target_rpc}/api/v0/pin/ls"
        async with self._loop_semaphore():  # throttle RPC
            client = self._loop_client()
            response = await client.post(pin_ls_url_base)
            response.raise_for_status()
            pins = response.json().get("Keys", [])
            return pins
