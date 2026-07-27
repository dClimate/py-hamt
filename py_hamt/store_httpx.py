import asyncio
import logging
import random
import re
import threading
import time
import warnings
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Literal, Optional, Tuple, cast
from urllib.parse import urlsplit

import httpx
from dag_cbor.ipld import IPLDKind
from multiformats import CID, multihash
from multiformats.multihash import Multihash

from . import instrumentation

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Ceiling on server-directed waits so a broken or hostile gateway cannot make
# a request sleep unbounded (e.g. ``Retry-After: inf`` or a far-future date).
_MAX_RETRY_AFTER_SECONDS = 300.0

# Consecutive failures before a gateway is considered unhealthy and moved to the
# back of the rotation, and how long it stays there before being retried. A
# tripped gateway is never permanently removed: after the cooldown it is probed
# again by ordinary traffic, so a gateway that recovers rejoins on its own.
_GATEWAY_FAILURE_THRESHOLD = 3
_GATEWAY_COOLDOWN_SECONDS = 30.0

# Headers forwarded to a gateway outside the credentialed origin. This is an
# allowlist, not a denylist: KuboCAS documents arbitrary headers as a supported
# way to authenticate ("set whatever headers ... they need"), so any header the
# caller configured may be a credential -- ``X-API-Key`` and ``X-Auth-Token``
# name themselves, but a bearer token can live under any name at all. Only
# httpx's own content-negotiation defaults are safe to send to a foreign
# gateway; anything else is dropped rather than guessed about.
#
# ``Host`` is deliberately absent: it is derived from the request URL, and
# forwarding the primary's value would misroute the fallback request.
_FORWARDABLE_HEADERS = frozenset({
    "accept",
    "accept-encoding",
    "accept-language",
    "user-agent",
})


def _origin_of(url: str) -> tuple[str, str, int | None]:
    """Scheme/host/port triple used to decide if two URLs share an origin."""
    parsed = urlsplit(url)
    return (parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port)


def _normalize_gateway_base_url(gateway_base_url: str) -> str:
    """Normalize a gateway base URL to a ``.../ipfs/`` prefix.

    Accepts a bare host (``https://example.com``), an explicit path
    (``https://example.com/ipfs``), and either with a trailing slash.
    """
    gateway_base_url = gateway_base_url.rstrip("/")
    if not gateway_base_url.endswith("/ipfs"):
        gateway_base_url = f"{gateway_base_url}/ipfs"
    return f"{gateway_base_url}/"


class _GatewayHealth:
    """Consecutive-failure circuit breaker for a single gateway.

    Gateways are never removed from the rotation permanently. Once
    ``_GATEWAY_FAILURE_THRESHOLD`` consecutive failures trip the breaker, the
    gateway is deprioritized (tried only after every healthy gateway) until
    ``_GATEWAY_COOLDOWN_SECONDS`` elapse, after which ordinary traffic probes it
    again. Any success resets the counter.

    Times are ``time.monotonic()`` readings. That clock is process-wide, so a
    trip recorded while one event loop is running stays comparable from another
    -- unlike ``loop.time()``, whose epoch is only meaningful within a single
    loop and would make the cooldown expire instantly or never once this shared
    state is touched from a second loop.
    """

    __slots__ = ("consecutive_failures", "tripped_at")

    def __init__(self) -> None:
        self.consecutive_failures: int = 0
        self.tripped_at: float | None = None

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.tripped_at = None

    def record_failure(self, now: float) -> None:
        self.consecutive_failures += 1
        if self.consecutive_failures >= _GATEWAY_FAILURE_THRESHOLD:
            self.tripped_at = now

    def is_healthy(self, now: float) -> bool:
        if self.tripped_at is None:
            return True
        if now - self.tripped_at >= _GATEWAY_COOLDOWN_SECONDS:
            # Cooldown elapsed. Clear the trip so a single probe failure does
            # not immediately re-trip on a stale counter, but keep the gateway
            # on probation by leaving the failure count one short of the
            # threshold: one more failure re-trips it right away.
            self.tripped_at = None
            self.consecutive_failures = _GATEWAY_FAILURE_THRESHOLD - 1
            return True
        return False


def _retry_delay(
    initial_delay: float,
    backoff_factor: float,
    retry_number: int,
    response: Optional[httpx.Response] = None,
) -> float:
    """Return a valid ``Retry-After`` value, otherwise a jittered backoff."""
    backoff_delay = initial_delay * (backoff_factor ** (retry_number - 1))
    retry_after = response.headers.get("Retry-After") if response is not None else None
    if retry_after is not None:
        try:
            retry_after_seconds = float(retry_after)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                retry_after_seconds = (
                    retry_at - datetime.now(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError, OverflowError):
                retry_after_seconds = -1

        if retry_after_seconds >= 0:
            return min(retry_after_seconds, _MAX_RETRY_AFTER_SECONDS)

    jitter = backoff_delay * 0.1 * (random.random() - 0.5)
    return backoff_delay + jitter


class _LoadStats:
    """Mutable trace counters shared across the gateways one load attempts."""

    __slots__ = ("response_bytes", "retries", "status")

    def __init__(self) -> None:
        self.response_bytes: int = 0
        self.retries: int = 0
        self.status: str = "ok"


class GatewayContentMismatch(Exception):
    """A gateway returned bytes that do not hash to the requested CID.

    Raised only when content verification is enabled. Treated as a per-gateway
    failure, so a multi-gateway ``KuboCAS`` fails over to the next gateway
    rather than returning corrupt data to the caller.
    """


def _cid_is_verifiable(cid: CID, offset: Optional[int], suffix: Optional[int]) -> bool:
    """Whether a gateway response for ``cid`` can be checked against its digest.

    Verification requires hashing the *complete* block, which holds only for:

    * **Full-body reads.** A Range request yields a slice, which does not hash
      to the CID.
    * **Non-``dag-pb`` CIDs.** A gateway serving a ``dag-pb`` CID returns the
      reassembled UnixFS file, not the encoded block the CID commits to, so the
      digest legitimately differs. ``raw`` and ``dag-cbor`` blocks -- what the
      HAMT itself stores -- are returned verbatim and do hash correctly.
    """
    if offset is not None or suffix is not None:
        return False
    # An identity multihash inlines the block in the CID, but load() still
    # fetches and returns whatever the gateway sends, so the response is
    # verifiable -- by direct comparison rather than rehashing.
    return cid.codec.code != KuboCAS.DAG_PB_MARKER


def _verify_cid_content(cid: CID, data: bytes) -> None:
    """Raise ``GatewayContentMismatch`` if ``data`` does not hash to ``cid``.

    An ``identity`` multihash inlines the block in the CID itself, so the bytes
    are compared directly rather than rehashed.

    The digest is computed at the CID's own digest length. That is required for
    correctness in both directions: variable-output functions (blake3, the
    default hasher here) *reject* a call that omits the size, and a legitimately
    truncated digest would never match one computed at full length.

    Verification is skipped only when the local ``multiformats`` build cannot
    compute the function at all -- we cannot prove the content wrong, so the
    read is allowed through with a warning. That path must stay narrow: silently
    skipping is indistinguishable from passing, which would make
    ``verify_content`` worthless exactly when it matters.
    """
    raw_digest = bytes(cid.raw_digest)
    if cid.hashfun.name == "identity":
        # Nothing was hashed: the CID carries the content verbatim.
        if data != raw_digest:
            raise GatewayContentMismatch(
                f"gateway returned {len(data)} bytes that do not match the "
                f"content inlined in identity CID {cid}"
            )
        return

    try:
        computed = multihash.digest(data, cid.hashfun.name, size=len(raw_digest))
    except Exception:  # pragma: no cover - depends on multiformats build
        logger.warning(
            "Cannot verify CID %s: local multiformats cannot compute %s at "
            "%d bytes; returning unverified gateway content",
            cid,
            cid.hashfun.name,
            len(raw_digest),
        )
        return
    if bytes(multihash.unwrap(computed)) != raw_digest:
        raise GatewayContentMismatch(
            f"gateway returned {len(data)} bytes that do not hash to {cid}"
        )


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


def _range_not_satisfiable_is_empty(
    response: httpx.Response, offset: Optional[int], suffix: Optional[int]
) -> bool:
    """Whether a ``416`` response describes a read Python slicing treats as empty.

    A spec-compliant gateway rejects an unsatisfiable range with ``416 Range Not
    Satisfiable`` and a ``Content-Range: bytes */N`` header giving the object
    size ``N``. Python slice semantics (and ``InMemoryCAS``) yield ``b""`` for
    those same reads, so ``KuboCAS`` matches by treating them as empty instead of
    surfacing the error:

    * an ``offset`` read is empty when it starts at or past EOF (``offset >= N``);
    * a ``suffix`` read is unsatisfiable only against a zero-length object
      (``N == 0``); ``data[-suffix:]`` on an empty object is likewise ``b""``.
      (For a non-empty object a suffix range is always satisfiable, so a ``416``
      there is a genuine error and is surfaced.)
    """
    content_range = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes \*/(\d+)", content_range.strip())
    if match is None:
        return False
    total = int(match.group(1))
    if suffix is not None:
        # A suffix range only fails against a zero-length object.
        return total == 0
    return offset is not None and offset >= total


def _validate_partial_content(
    response: httpx.Response,
    offset: Optional[int],
    length: Optional[int],
    suffix: Optional[int],
    body_len: int,
) -> None:
    """Reject a ``206`` whose byte window is missing or inconsistent.

    ``raise_for_status`` accepts any 2xx, so a gateway can answer a Range request
    with a malformed ``206`` -- an absent, unparseable, or ``*``-total
    ``Content-Range``, a declared window that disagrees with the body length, or
    a window that does not match what was requested -- and silently hand back the
    wrong bytes. We recompute the exact window the request maps to (the object
    size ``N`` is always known for content-addressed reads) and raise an
    ``httpx.HTTPStatusError`` on any mismatch so a corrupt partial read fails
    loudly rather than corrupting the caller's data.
    """
    content_range = response.headers.get("Content-Range", "")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range.strip())
    if match is None:
        raise httpx.HTTPStatusError(
            f"malformed 206 Content-Range {content_range!r}",
            request=response.request,
            response=response,
        )
    start, end, total = (int(group) for group in match.groups())
    # The declared inclusive window must match the number of bytes delivered.
    if end - start + 1 != body_len:
        raise httpx.HTTPStatusError(
            f"206 Content-Range {content_range!r} declares "
            f"{end - start + 1} bytes but body is {body_len}",
            request=response.request,
            response=response,
        )
    # Recompute the window the request maps to and demand an exact match, so a
    # gateway cannot return a shifted or wrong-sized slice of the object.
    if offset is not None:
        expected_start = offset
        expected_len = total - offset if length is None else min(length, total - offset)
    else:
        expected_start = max(total - cast(int, suffix), 0)
        expected_len = min(cast(int, suffix), total)
    if start != expected_start or body_len != expected_len:
        raise httpx.HTTPStatusError(
            f"206 Content-Range {content_range!r} (start={start}, {body_len} bytes) "
            f"does not match the requested window "
            f"(start={expected_start}, {expected_len} bytes)",
            request=response.request,
            response=response,
        )


# Upper bound on how long aclose() waits for a cross-loop client close that was
# scheduled onto a *running* owner loop. Bounds the window where that loop stops
# between the is_running() check and the coroutine executing, which would
# otherwise leave the wrapped future pending forever. Module-level so tests can
# shrink it; a healthy running loop closes near-instantly, well under this.
_CROSS_LOOP_ACLOSE_TIMEOUT_S = 30.0


def _close_client_on_stopped_loop(
    owner_loop: asyncio.AbstractEventLoop, client: httpx.AsyncClient
) -> None:
    """Run client cleanup on its stopped but still usable owner loop."""
    owner_loop.run_until_complete(client.aclose())


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

    1. **Bring your own `httpx.AsyncClient` or client factory**
       Pass a client via `client=...` for use on one event loop, or pass
       `client_factory=...` to build a fully configured client for each event
       loop. Reusing a supplied client from a later loop emits a warning and
       falls back to an internal client that preserves only headers, auth,
       timeout, redirect policy, and event hooks.
    2. **Let `KuboCAS` build the client** but pass
       `headers=` *and*/or `auth=` kwargs; they are forwarded to the
       internally-created `AsyncClient`.

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
      redirect following and HTTP/2 enabled. Plaintext endpoints continue to
      use HTTP/1.1 because HTTP/2 negotiation requires TLS/ALPN.
    - **client_factory** (`Callable[[], httpx.AsyncClient] | None`): create a
      separate, fully configured client for each event loop. KuboCAS owns and
      closes clients returned by the factory. Mutually exclusive with
      **client**.
    - **headers** (dict[str, str] | None): default headers for the
      internally-created client.
    - **auth** (`tuple[str, str] | None`): authentication tuple (username, password)
      for the internally-created client.
    - **rpc_base_url / gateway_base_url** (str | None): override daemon
      endpoints (defaults match the local daemon ports). Gateway URLs may end
      with `/ipfs` and may include a trailing slash.
    - **gateway_base_urls** (list[str] | None): read from several gateways with
      automatic failover. Mutually exclusive with `gateway_base_url`. Each read
      tries one gateway at a time, healthy gateways first, until one succeeds;
      requests are not raced in parallel. A gateway that fails three times in a
      row is moved to the back of the rotation for 30 seconds and then probed
      again. `concurrency` applies per gateway. If every gateway fails, an
      `ExceptionGroup` of the underlying errors is raised.
    - **verify_content** (bool): check that returned bytes hash to the
      requested CID, raising `GatewayContentMismatch` (and failing over to the
      next gateway) when they do not. Worth enabling when reading from public
      gateways you do not control. Only full-body reads of non-`dag-pb` CIDs
      can be verified; Range reads and `dag-pb` reads are passed through
      unchecked because neither returns the exact bytes the CID commits to.
      Requests to a gateway outside the origin of
      `gateway_base_url`/`rpc_base_url` forward only content-negotiation
      headers (`Accept`, `Accept-Encoding`, `Accept-Language`, `User-Agent`)
      and drop client-level `auth`. Because any header name may carry a
      credential, everything else is withheld -- so a private primary can
      safely be paired with public fallbacks, but a foreign gateway that needs
      its own custom header will not receive one.
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
        gateway_base_urls: list[str] | None = None,
        verify_content: bool = False,
        client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
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
        first use, so constructing ``KuboCAS`` does not require an async
        context. On a later event loop, KuboCAS warns and uses an internally
        created fallback that preserves only the supplied client's headers,
        auth, timeout, redirect policy, and event hooks. Pass
        ``client_factory`` instead when every event loop needs the client's
        full configuration. Factory clients are owned and closed by KuboCAS.
        Clients created internally by ``KuboCAS`` use a 60-second timeout,
        follow redirects, and negotiate HTTP/2 for HTTPS endpoints that
        support it.

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

        if client is not None and client_factory is not None:
            raise ValueError("client and client_factory are mutually exclusive")
        if client_factory is not None and (headers is not None or auth is not None):
            raise ValueError(
                "client_factory is mutually exclusive with headers/auth; "
                "configure them on the clients the factory builds"
            )

        self._owns_client: bool = False
        self._closed: bool = True
        self._client_per_loop: Dict[asyncio.AbstractEventLoop, httpx.AsyncClient] = {}
        self._internally_created_clients: set[httpx.AsyncClient] = set()
        # Serializes first-use client binding so concurrent event loops on
        # different threads cannot both consume ``_supplied_client`` and bind
        # one httpx.AsyncClient to two loops.
        self._first_use_lock: threading.Lock = threading.Lock()
        self._semaphore_per_loop: Dict[
            asyncio.AbstractEventLoop, asyncio.Semaphore
        ] = {}
        # Gateway reads get a semaphore per (loop, gateway) so ``concurrency``
        # means "in-flight requests per gateway". Sharing one budget across
        # gateways would divide effective parallelism by the gateway count and
        # let a slow gateway starve the healthy ones of slots.
        self._gateway_semaphore_per_loop: Dict[
            Tuple[asyncio.AbstractEventLoop, str], asyncio.Semaphore
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

        if gateway_base_urls is not None:
            if gateway_base_url is not None:
                raise ValueError(
                    "gateway_base_url and gateway_base_urls are mutually "
                    "exclusive; pass every gateway in gateway_base_urls"
                )
            if not gateway_base_urls:
                raise ValueError("gateway_base_urls must not be empty")
            normalized = [_normalize_gateway_base_url(url) for url in gateway_base_urls]
            # Preserve caller order while dropping duplicates: a repeated
            # gateway would otherwise get several rotation slots and several
            # independent concurrency budgets pointed at one host.
            self.gateway_base_urls: list[str] = list(dict.fromkeys(normalized))
        else:
            if gateway_base_url is None:
                gateway_base_url = KuboCAS.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL
            self.gateway_base_urls = [_normalize_gateway_base_url(gateway_base_url)]

        pin_string: str = "true" if pin_on_add else "false"
        self.rpc_url: str = f"{rpc_base_url}/api/v0/add?hash={self.hasher}&chunker={self.chunker}&pin={pin_string}"
        """@private"""
        self.gateway_base_url: str = self.gateway_base_urls[0]
        """@private"""

        # Origins the caller's credentials were configured for: the primary
        # gateway and the RPC endpoint. Reads to any other gateway drop
        # credentialed headers and client auth (see _load_from_gateway).
        self._credentialed_origins: set[tuple[str, str, int | None]] = {
            _origin_of(self.gateway_base_url),
            _origin_of(rpc_base_url),
        }

        self.verify_content: bool = verify_content
        """@private"""
        # Health is per gateway but shared across event loops: a gateway that is
        # rate-limiting or down is doing so regardless of which loop observed it.
        self._gateway_health: Dict[str, _GatewayHealth] = {
            url: _GatewayHealth() for url in self.gateway_base_urls
        }

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
            self._default_follow_redirects: bool = client.follow_redirects
            # Snapshot the hooks like the headers above: later mutations of the
            # supplied client must not leak into fallback clients.
            self._default_event_hooks: dict[str, list[Callable[..., Any]]] | None = {
                event: list(hooks) for event, hooks in client.event_hooks.items()
            }
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
            self._default_follow_redirects = True
            self._default_event_hooks = None
        self._client_factory: Optional[Callable[[], httpx.AsyncClient]] = client_factory

        if concurrency <= 0:
            raise ValueError("concurrency must be a positive integer")
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
            self._gateway_semaphore_per_loop = {}

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            return self._semaphore_per_loop[loop]
        except KeyError:
            semaphore = asyncio.Semaphore(self._concurrency)
            self._semaphore_per_loop[loop] = semaphore
            return semaphore

    def _gateway_semaphore(self, gateway_base_url: str) -> asyncio.Semaphore:
        """Get or create the concurrency semaphore for one gateway on this loop.

        With a single gateway this is equivalent to ``_loop_semaphore``; with
        several it keeps each gateway's ``concurrency`` budget independent.
        """
        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        key = (loop, gateway_base_url)
        try:
            return self._gateway_semaphore_per_loop[key]
        except KeyError:
            semaphore = asyncio.Semaphore(self._concurrency)
            self._gateway_semaphore_per_loop[key] = semaphore
            return semaphore

    def _ordered_gateways(self) -> list[str]:
        """Gateways to try, best first.

        Ordered by health, then by whether a concurrency slot is free right now.
        The second key avoids head-of-line blocking: attempts wait on their
        gateway's semaphore, so without it a read queued behind a saturated
        gateway would stall even when an idle gateway could serve it
        immediately -- precisely the case multiple gateways exist to handle.

        Deprioritizing rather than dropping unhealthy gateways means a run where
        every gateway has tripped still attempts them all instead of failing
        with nothing tried.
        """
        if len(self.gateway_base_urls) == 1:
            return self.gateway_base_urls

        now = time.monotonic()
        # is_healthy() clears an expired trip, so evaluate it exactly once per
        # gateway rather than inside a sort key, which may call it repeatedly.
        ranks: Dict[str, tuple[int, int]] = {}
        for url in self.gateway_base_urls:
            unhealthy = 0 if self._gateway_health[url].is_healthy(now) else 1
            busy = 1 if self._gateway_semaphore(url).locked() else 0
            ranks[url] = (unhealthy, busy)

        # Stable sort, so configured order breaks ties within a rank.
        return sorted(self.gateway_base_urls, key=lambda url: ranks[url])

    def _loop_client(self) -> httpx.AsyncClient:
        """Get or create a client for the current event loop.

        A user-supplied client is bound to the first loop that requests it.
        If the instance was previously closed but owns its clients, a fresh
        client mapping is lazily created on demand. Users that supplied their
        own ``httpx.AsyncClient`` still receive an error when the instance has
        been closed, as we cannot safely recreate their client. Internally
        created clients enable HTTP/2 negotiation for HTTPS endpoints.
        """
        if self._closed:
            if not self._owns_client:
                raise RuntimeError("KuboCAS is closed; create a new instance")
            # We previously closed all internally-owned clients. Reset the
            # state so that new clients can be created lazily.
            self._closed = False
            self._client_per_loop = {}
            self._semaphore_per_loop = {}
            self._gateway_semaphore_per_loop = {}

        loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        try:
            return self._client_per_loop[loop]
        except KeyError:
            # First use on this loop. Hold the lock across supplied-client
            # consumption, client creation, and the per-loop assignment so two
            # loops racing on different threads cannot bind the same client.
            with self._first_use_lock:
                if self._supplied_client is not None:
                    client = self._supplied_client
                    self._supplied_client = None
                elif self._client_factory is not None:
                    client = self._client_factory()
                    self._internally_created_clients.add(client)
                else:
                    if self._user_client is not None:
                        warnings.warn(
                            "A user-supplied httpx.AsyncClient cannot be reused "
                            "across event loops; falling back to an internally "
                            "created client that preserves only headers, auth, "
                            "timeout, limits, redirect policy, and event hooks. "
                            "Pass client_factory to preserve full configuration.",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                    client = httpx.AsyncClient(
                        timeout=self._default_timeout,
                        headers=self._default_headers,
                        auth=self._default_auth,
                        limits=self._default_limits,
                        follow_redirects=self._default_follow_redirects,
                        event_hooks=self._default_event_hooks,
                        http2=True,
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

        For clients owned by closed loops with stock async-only transports,
        cleanup degenerates to a warning. The OS-level socket is shut down
        with a FIN, but its local file descriptor is released at garbage
        collection. Callers that require deterministic release should call
        ``aclose()`` on the owning loop before it exits.
        """
        try:
            current_loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
        except RuntimeError:
            current_loop = None

        for owner_loop, client in list(self._client_per_loop.items()):
            if client not in self._internally_created_clients:
                continue

            try:
                if owner_loop is current_loop:
                    await client.aclose()
                    continue

                if not owner_loop.is_closed():
                    if owner_loop.is_running():
                        close_future = asyncio.run_coroutine_threadsafe(
                            client.aclose(), owner_loop
                        )
                        try:
                            await asyncio.wait_for(
                                asyncio.wrap_future(close_future),
                                timeout=_CROSS_LOOP_ACLOSE_TIMEOUT_S,
                            )
                        except TimeoutError:
                            # The owner loop stopped (or stalled) after
                            # is_running() succeeded, so the scheduled close can
                            # never complete. Cancel it and fall through to the
                            # synchronous transport shutdown below.
                            close_future.cancel()
                        else:
                            continue
                    else:
                        await asyncio.to_thread(
                            _close_client_on_stopped_loop, owner_loop, client
                        )
                        continue

                # AsyncClient marks itself closed before awaiting its transport.
                # A dead owner loop therefore needs the transport's sync fallback.
                transport: Any = client._transport
                close_transport = getattr(transport, "close", None)
                if close_transport is None:
                    await client.aclose()
                    continue

                close_transport()
                try:
                    await client.aclose()
                except Exception:
                    pass  # The transport was already closed synchronously.
            except Exception as exc:
                warnings.warn(
                    f"Failed to close an internally created HTTP client: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

        self._client_per_loop.clear()
        self._internally_created_clients.clear()
        self._semaphore_per_loop.clear()
        self._gateway_semaphore_per_loop.clear()
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
                self._gateway_semaphore_per_loop.clear()
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
                self._gateway_semaphore_per_loop.clear()
                self._closed = True

    # --------------------------------------------------------------------- #
    # save() - now uses the per-loop client                                 #
    # --------------------------------------------------------------------- #
    async def save(self, data: bytes, codec: ContentAddressedStore.CodecInput) -> CID:
        """Add data to Kubo and return its CID.

        Transient request failures and gateway statuses are retried. Retrying
        the ``/api/v0/add`` POST is safe because the uploaded content is
        content-addressed, making repeated additions idempotent. Concurrency
        slots are held per HTTP attempt and released during retry backoff.
        """
        files = {"file": data}
        client = self._loop_client()
        semaphore = self._loop_semaphore()
        retry_count = 0

        while retry_count <= self.max_retries:
            try:
                async with semaphore:
                    response = await client.post(self.rpc_url, files=files)
                response.raise_for_status()
                cid_str: str = response.json()["Hash"]
                cid: CID = CID.decode(cid_str)
                if cid.codec.code != self.DAG_PB_MARKER:
                    cid = cid.set(codec=codec)
                return cid

            except httpx.RequestError:
                if retry_count >= self.max_retries:
                    raise
                retry_count += 1
                await asyncio.sleep(
                    _retry_delay(self.initial_delay, self.backoff_factor, retry_count)
                )

            except httpx.HTTPStatusError as error:
                if error.response.status_code not in _RETRYABLE_STATUS_CODES:
                    raise
                if retry_count >= self.max_retries:
                    raise
                retry_count += 1
                await asyncio.sleep(
                    _retry_delay(
                        self.initial_delay,
                        self.backoff_factor,
                        retry_count,
                        error.response,
                    )
                )
        raise RuntimeError("Exited the retry loop unexpectedly.")  # pragma: no cover

    async def _load_from_gateway(
        self,
        gateway_base_url: str,
        cid: CID,
        headers: Dict[str, str],
        offset: Optional[int],
        length: Optional[int],
        suffix: Optional[int],
        stats: "_LoadStats",
    ) -> bytes:
        """Fetch ``cid`` from one gateway, retrying that gateway's transients.

        Raises on failure so the caller can fail over. ``stats`` accumulates the
        byte count and retry total across every gateway attempted, so the trace
        emitted by ``load`` reflects the whole operation rather than the last leg.

        Credentials configured for the primary gateway or the RPC endpoint are
        stripped when this gateway is on a different origin, so failing over to
        a public fallback cannot disclose a private gateway's token.
        """
        url = f"{gateway_base_url}{cid}"
        client = self._loop_client()
        semaphore = self._gateway_semaphore(gateway_base_url)
        retry_count = 0

        # httpx merges client-level headers into every request and offers no way
        # to drop one per-request (Client._merge_headers starts from
        # self.headers and only update()s, so an omitted or blanked entry is
        # reinstated). Building the Request explicitly and calling send()
        # bypasses that merge, which is the only reliable way to withhold a
        # credential from a foreign origin.
        strip_credentials = (
            _origin_of(gateway_base_url) not in self._credentialed_origins
        )
        request: httpx.Request | None = None
        if strip_credentials:
            safe_headers = {
                name: value
                for name, value in client.headers.items()
                if name.lower() in _FORWARDABLE_HEADERS
            }
            # Range headers are computed by load() for this request, never
            # caller-supplied credentials, so they are always safe to send.
            safe_headers.update(headers)
            request = httpx.Request("GET", url, headers=safe_headers)

        while retry_count <= self.max_retries:
            try:
                async with semaphore:  # Throttle each gateway attempt
                    if request is not None:
                        # auth=None also suppresses client-level httpx.Auth,
                        # which would otherwise re-add an Authorization header.
                        response = await client.send(
                            request, auth=None, follow_redirects=client.follow_redirects
                        )
                    else:
                        response = await client.get(url, headers=headers or None)
                # An unsatisfiable range is answered with 416 by a compliant
                # gateway; return b"" to match Python-slice semantics (and
                # InMemoryCAS) instead of raising.
                if (
                    response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE
                    and _range_not_satisfiable_is_empty(response, offset, suffix)
                ):
                    return b""
                response.raise_for_status()
                content = response.content
                stats.response_bytes = len(content)
                if headers:
                    if response.status_code == httpx.codes.OK:
                        logger.debug(
                            "Gateway ignored Range request for CID %s; "
                            "slicing the complete response locally",
                            cid,
                        )
                        return _slice_requested_range(content, offset, length, suffix)
                    if response.status_code == httpx.codes.PARTIAL_CONTENT:
                        # Trust the partial body only after proving its
                        # Content-Range matches the requested byte window.
                        _validate_partial_content(
                            response, offset, length, suffix, stats.response_bytes
                        )
                        return content
                    # Any other 2xx to a Range request is unexpected: we
                    # cannot know which bytes it carries, so fail rather than
                    # return a possibly-wrong window.
                    raise httpx.HTTPStatusError(
                        f"unexpected {response.status_code} response to a "
                        "Range request",
                        request=response.request,
                        response=response,
                    )
                if self.verify_content and _cid_is_verifiable(cid, offset, suffix):
                    # A mismatch means this gateway served wrong bytes. Raising
                    # here routes it through the caller's failover path like any
                    # other per-gateway failure.
                    try:
                        _verify_cid_content(cid, content)
                    except GatewayContentMismatch:
                        stats.status = "content_mismatch"
                        raise
                return content

            except httpx.RequestError:
                if retry_count >= self.max_retries:
                    stats.status = "request_error"
                    raise
                retry_count += 1
                stats.retries += 1
                await asyncio.sleep(
                    _retry_delay(self.initial_delay, self.backoff_factor, retry_count)
                )

            except httpx.HTTPStatusError as error:
                if (
                    error.response.status_code not in _RETRYABLE_STATUS_CODES
                    or retry_count >= self.max_retries
                ):
                    stats.status = "http_error"
                    raise
                retry_count += 1
                stats.retries += 1
                await asyncio.sleep(
                    _retry_delay(
                        self.initial_delay,
                        self.backoff_factor,
                        retry_count,
                        error.response,
                    )
                )
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
        are handled by applying the requested byte window locally. Transient
        request failures, rate limits, and gateway server errors are retried;
        other HTTP errors fail immediately. Zero-length and zero-suffix reads
        return immediately without a gateway request. Concurrency slots are
        held per HTTP attempt and released during retry backoff.

        When several gateways are configured, each is tried in turn -- healthy
        ones first -- until one succeeds. Requests are *not* raced in parallel:
        fanning every read out to every gateway would multiply egress and burn
        each gateway's rate-limit budget N times over, which is the opposite of
        what helps when rate limiting is the problem being solved. A gateway
        that fails ``_GATEWAY_FAILURE_THRESHOLD`` times consecutively is moved
        to the back of the rotation for a cooldown. If every gateway fails, the
        collected errors are raised together as an ``ExceptionGroup``.
        """
        if (offset is not None and length == 0) or (offset is None and suffix == 0):
            return b""

        cid = cast(CID, id)
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
        stats = _LoadStats()
        gateways = self._ordered_gateways()
        failures: list[Exception] = []
        try:
            for gateway_base_url in gateways:
                health = self._gateway_health[gateway_base_url]
                try:
                    content = await self._load_from_gateway(
                        gateway_base_url, cid, headers, offset, length, suffix, stats
                    )
                except (httpx.HTTPError, GatewayContentMismatch) as error:
                    health.record_failure(time.monotonic())
                    failures.append(error)
                    if len(gateways) > 1:
                        logger.debug(
                            "Gateway %s failed for CID %s (%s); trying the next one",
                            gateway_base_url,
                            cid,
                            error,
                        )
                    continue
                else:
                    health.record_success()
                    # A gateway leg may have set a failure status before a later
                    # gateway succeeded; the operation as a whole is a success.
                    stats.status = "ok"
                    return content

            # Every gateway failed. With one configured, re-raise its error
            # unchanged so existing single-gateway callers keep seeing the exact
            # httpx exception type they handle today.
            if len(failures) == 1:
                raise failures[0]
            raise ExceptionGroup(
                f"all {len(gateways)} gateways failed for CID {cid}", failures
            )
        finally:
            instrumentation.end_cas_load(
                trace_started_at,
                byte_count=stats.response_bytes,
                retries=stats.retries,
                status=stats.status,
            )

    # --------------------------------------------------------------------- #
    # pin_cid() - method to pin a CID                                       #
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
