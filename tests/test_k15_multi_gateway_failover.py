"""Multi-gateway failover, per-gateway concurrency, and content verification."""

import asyncio
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable

import httpx
import pytest
from multiformats import CID, multihash

from py_hamt import GatewayContentMismatch, KuboCAS
from py_hamt import store_httpx as store_module

BODY = b"multi gateway body"
# raw + sha2-256 CID over BODY, so verify_content can check it for real.
GOOD_CID = CID("base32", 1, "raw", multihash.digest(BODY, "sha2-256"))


@dataclass
class FakeGateway:
    """A stub IPFS gateway recording every request it serves."""

    url: str
    hits: list[str] = field(default_factory=list)
    # Set by tests to control responses; returns (status, body).
    responder: Callable[[str], tuple[int, bytes]] = lambda _cid: (200, BODY)
    max_concurrent: int = 0
    _inflight: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)


def _serve(gateway_holder: list[FakeGateway]) -> Iterator[FakeGateway]:
    """Run a threaded HTTP server backed by ``gateway_holder[0]``."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            gw = gateway_holder[0]
            cid = self.path.rsplit("/", 1)[-1].split("?", 1)[0]
            with gw._lock:
                gw.hits.append(cid)
                gw._inflight += 1
                gw.max_concurrent = max(gw.max_concurrent, gw._inflight)
            try:
                status, body = gw.responder(cid)
            finally:
                with gw._lock:
                    gw._inflight -= 1
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if body:
                self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    gateway_holder[0].url = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        yield gateway_holder[0]
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def gateway_a() -> Iterator[FakeGateway]:
    yield from _serve([FakeGateway(url="")])


@pytest.fixture
def gateway_b() -> Iterator[FakeGateway]:
    yield from _serve([FakeGateway(url="")])


@pytest.fixture
def gateway_c() -> Iterator[FakeGateway]:
    yield from _serve([FakeGateway(url="")])


def make_cas(*gateways: FakeGateway, **kwargs: object) -> KuboCAS:
    return KuboCAS(
        gateway_base_urls=[gw.url for gw in gateways],
        rpc_base_url=gateways[0].url,
        initial_delay=0.01,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# construction and normalization                                              #
# --------------------------------------------------------------------------- #


def test_gateway_base_urls_normalize_and_expose_first() -> None:
    cas = KuboCAS(
        gateway_base_urls=[
            "https://a.example.com",
            "https://b.example.com/ipfs",
            "https://c.example.com/ipfs/",
        ],
        rpc_base_url="https://a.example.com",
    )
    assert cas.gateway_base_urls == [
        "https://a.example.com/ipfs/",
        "https://b.example.com/ipfs/",
        "https://c.example.com/ipfs/",
    ]
    # gateway_base_url stays meaningful for single-gateway callers and docs.
    assert cas.gateway_base_url == "https://a.example.com/ipfs/"


def test_gateway_base_urls_deduplicates_after_normalization() -> None:
    cas = KuboCAS(
        gateway_base_urls=[
            "https://a.example.com",
            "https://a.example.com/ipfs/",
            "https://b.example.com",
        ],
        rpc_base_url="https://a.example.com",
    )
    assert cas.gateway_base_urls == [
        "https://a.example.com/ipfs/",
        "https://b.example.com/ipfs/",
    ]


def test_gateway_base_url_and_urls_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        KuboCAS(
            gateway_base_url="https://a.example.com",
            gateway_base_urls=["https://b.example.com"],
        )


def test_empty_gateway_base_urls_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        KuboCAS(gateway_base_urls=[])


def test_single_gateway_default_is_unchanged() -> None:
    cas = KuboCAS(rpc_base_url="https://a.example.com")
    assert cas.gateway_base_urls == [
        f"{KuboCAS.KUBO_DEFAULT_LOCAL_GATEWAY_BASE_URL}/ipfs/"
    ]


# --------------------------------------------------------------------------- #
# failover                                                                     #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_failover_to_second_gateway_on_server_error(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    gateway_a.responder = lambda _cid: (500, b"")

    async with make_cas(gateway_a, gateway_b, max_retries=0) as cas:
        assert await cas.load(GOOD_CID) == BODY

    assert gateway_a.hits, "the first gateway should have been tried"
    assert gateway_b.hits, "the second gateway should have served the read"


@pytest.mark.asyncio
async def test_first_healthy_gateway_wins_and_others_are_untouched(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """Reads are sequential, not raced: a success must not fan out."""
    async with make_cas(gateway_a, gateway_b) as cas:
        assert await cas.load(GOOD_CID) == BODY

    assert len(gateway_a.hits) == 1
    assert gateway_b.hits == []


@pytest.mark.asyncio
async def test_404_fails_over_rather_than_failing_the_read(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """A gateway that lacks the block is a reason to ask another one.

    404 is non-retryable *on one gateway*, but across gateways it is exactly
    the case failover exists for.
    """
    gateway_a.responder = lambda _cid: (404, b"")

    async with make_cas(gateway_a, gateway_b, max_retries=0) as cas:
        assert await cas.load(GOOD_CID) == BODY

    assert len(gateway_a.hits) == 1
    assert len(gateway_b.hits) == 1


@pytest.mark.asyncio
async def test_all_gateways_failing_raises_exception_group(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    gateway_a.responder = lambda _cid: (500, b"")
    gateway_b.responder = lambda _cid: (503, b"")

    async with make_cas(gateway_a, gateway_b, max_retries=0) as cas:
        with pytest.raises(ExceptionGroup) as excinfo:
            await cas.load(GOOD_CID)

    assert "all 2 gateways failed" in str(excinfo.value)
    assert len(excinfo.value.exceptions) == 2
    # The underlying errors survive, rather than being flattened to a message.
    assert all(
        isinstance(exc, httpx.HTTPStatusError) for exc in excinfo.value.exceptions
    )


@pytest.mark.asyncio
async def test_single_gateway_still_raises_bare_httpx_error(
    gateway_a: FakeGateway,
) -> None:
    """One gateway must keep its pre-existing exception contract."""
    gateway_a.responder = lambda _cid: (500, b"")

    async with make_cas(gateway_a, max_retries=0) as cas:
        with pytest.raises(httpx.HTTPStatusError):
            await cas.load(GOOD_CID)


@pytest.mark.asyncio
async def test_retries_are_exhausted_per_gateway_before_failover(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    gateway_a.responder = lambda _cid: (503, b"")

    async with make_cas(gateway_a, gateway_b, max_retries=2) as cas:
        assert await cas.load(GOOD_CID) == BODY

    # Initial attempt plus two retries against the failing gateway.
    assert len(gateway_a.hits) == 3
    assert len(gateway_b.hits) == 1


# --------------------------------------------------------------------------- #
# circuit breaker                                                              #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_unhealthy_gateway_is_deprioritized_then_probed_again(
    gateway_a: FakeGateway, gateway_b: FakeGateway, monkeypatch: pytest.MonkeyPatch
) -> None:
    gateway_a.responder = lambda _cid: (500, b"")

    async with make_cas(gateway_a, gateway_b, max_retries=0) as cas:
        for _ in range(store_module._GATEWAY_FAILURE_THRESHOLD):
            assert await cas.load(GOOD_CID) == BODY

        tripped_hits = len(gateway_a.hits)
        assert tripped_hits == store_module._GATEWAY_FAILURE_THRESHOLD

        # Now tripped: subsequent reads should skip straight to gateway B.
        for _ in range(3):
            assert await cas.load(GOOD_CID) == BODY
        assert len(gateway_a.hits) == tripped_hits, (
            "tripped gateway still receiving traffic"
        )

        # After the cooldown it is probed again by ordinary traffic.
        monkeypatch.setattr(store_module, "_GATEWAY_COOLDOWN_SECONDS", 0.0)
        assert await cas.load(GOOD_CID) == BODY
        assert len(gateway_a.hits) == tripped_hits + 1


@pytest.mark.asyncio
async def test_recovered_gateway_resumes_priority(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """A success resets the failure counter, so a flap does not trip later."""
    responses = [(500, b""), (500, b"")]

    def flaky(_cid: str) -> tuple[int, bytes]:
        return responses.pop(0) if responses else (200, BODY)

    gateway_a.responder = flaky

    async with make_cas(gateway_a, gateway_b, max_retries=0) as cas:
        # Two failures: one short of the threshold, so A stays in front.
        assert await cas.load(GOOD_CID) == BODY
        assert await cas.load(GOOD_CID) == BODY
        # A recovers and serves this one itself.
        assert await cas.load(GOOD_CID) == BODY
        health = cas._gateway_health[cas.gateway_base_urls[0]]
        assert health.consecutive_failures == 0
        # A further failure does not immediately trip a stale counter.
        assert await cas.load(GOOD_CID) == BODY
        assert len(gateway_b.hits) == 2


@pytest.mark.asyncio
async def test_all_gateways_unhealthy_still_attempts_every_one(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """Deprioritized never means dropped, or a total outage would try nothing."""
    gateway_a.responder = lambda _cid: (500, b"")
    gateway_b.responder = lambda _cid: (500, b"")

    async with make_cas(gateway_a, gateway_b, max_retries=0) as cas:
        for _ in range(store_module._GATEWAY_FAILURE_THRESHOLD):
            with pytest.raises(ExceptionGroup):
                await cas.load(GOOD_CID)

        hits_before = (len(gateway_a.hits), len(gateway_b.hits))
        with pytest.raises(ExceptionGroup):
            await cas.load(GOOD_CID)

    assert len(gateway_a.hits) == hits_before[0] + 1
    assert len(gateway_b.hits) == hits_before[1] + 1


def test_health_cooldown_leaves_gateway_on_probation() -> None:
    health = store_module._GatewayHealth()
    for _ in range(store_module._GATEWAY_FAILURE_THRESHOLD):
        health.record_failure(now=0.0)

    assert not health.is_healthy(now=1.0)
    # Cooldown elapsed: eligible again, but one failure away from re-tripping.
    assert health.is_healthy(now=store_module._GATEWAY_COOLDOWN_SECONDS + 1.0)
    health.record_failure(now=store_module._GATEWAY_COOLDOWN_SECONDS + 1.0)
    assert not health.is_healthy(now=store_module._GATEWAY_COOLDOWN_SECONDS + 1.0)


# --------------------------------------------------------------------------- #
# per-gateway concurrency                                                      #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrency_budget_is_per_gateway(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """Each gateway gets its own ``concurrency`` slots, not a shared pool.

    Asserted structurally rather than by racing wall-clock: each gateway must
    get a distinct semaphore carrying the full ``concurrency`` budget, so
    saturating one cannot starve reads routed to another. Timing-based variants
    of this check are dominated by connection-pool and handler-thread
    scheduling rather than by the semaphores under test.
    """
    async with make_cas(gateway_a, gateway_b, concurrency=2, max_retries=0) as cas:
        url_a, url_b = cas.gateway_base_urls

        # Drive a real read through each gateway so both semaphores are the
        # ones load() actually used, not ones conjured by the assertion.
        gateway_a.responder = lambda _cid: (500, b"")
        assert await cas.load(GOOD_CID) == BODY
        assert gateway_a.hits and gateway_b.hits

        sem_a = cas._gateway_semaphore(url_a)
        sem_b = cas._gateway_semaphore(url_b)
        assert sem_a is not sem_b, "gateways must not share a concurrency budget"

        # Each carries the full budget, and exhausting A leaves B untouched.
        await sem_a.acquire()
        await sem_a.acquire()
        assert sem_a.locked()
        assert not sem_b.locked()
        sem_a.release()
        sem_a.release()


@pytest.mark.asyncio
async def test_gateway_semaphore_bounds_in_flight_requests(
    gateway_a: FakeGateway,
) -> None:
    import time

    def slow(_cid: str) -> tuple[int, bytes]:
        time.sleep(0.05)
        return 200, BODY

    gateway_a.responder = slow

    async with make_cas(gateway_a, concurrency=2) as cas:
        await asyncio.gather(*(cas.load(GOOD_CID) for _ in range(6)))

    assert gateway_a.max_concurrent <= 2


@pytest.mark.asyncio
async def test_saturated_gateway_does_not_block_an_idle_one(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """A busy gateway must not stall reads an idle gateway could serve.

    Attempts wait on their gateway's semaphore, so ordering strictly by
    configured position would queue this read behind saturated gateway A even
    though B is free -- the head-of-line stall that undermines the whole point
    of configuring several gateways.
    """
    async with make_cas(gateway_a, gateway_b, concurrency=1, max_retries=0) as cas:
        sem_a = cas._gateway_semaphore(cas.gateway_base_urls[0])
        await sem_a.acquire()  # gateway A has no free slot
        try:
            result = await asyncio.wait_for(cas.load(GOOD_CID), timeout=5.0)
        finally:
            sem_a.release()

    assert result == BODY
    assert gateway_a.hits == [], "saturated gateway should have been skipped"
    assert len(gateway_b.hits) == 1


@pytest.mark.asyncio
async def test_gateway_semaphores_cleared_on_aclose(gateway_a: FakeGateway) -> None:
    cas = make_cas(gateway_a)
    await cas.load(GOOD_CID)
    assert cas._gateway_semaphore_per_loop
    await cas.aclose()
    assert cas._gateway_semaphore_per_loop == {}


# --------------------------------------------------------------------------- #
# content verification                                                         #
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_verification_rejects_wrong_bytes_and_fails_over(
    gateway_a: FakeGateway, gateway_b: FakeGateway
) -> None:
    """A gateway serving valid-looking but wrong content must not win."""
    gateway_a.responder = lambda _cid: (200, b"corrupted payload")

    async with make_cas(
        gateway_a, gateway_b, verify_content=True, max_retries=0
    ) as cas:
        assert await cas.load(GOOD_CID) == BODY

    assert len(gateway_a.hits) == 1
    assert len(gateway_b.hits) == 1


@pytest.mark.asyncio
async def test_verification_failure_on_sole_gateway_raises(
    gateway_a: FakeGateway,
) -> None:
    gateway_a.responder = lambda _cid: (200, b"corrupted payload")

    async with make_cas(gateway_a, verify_content=True, max_retries=0) as cas:
        with pytest.raises(GatewayContentMismatch, match="do not hash to"):
            await cas.load(GOOD_CID)


@pytest.mark.asyncio
async def test_verification_accepts_matching_bytes(gateway_a: FakeGateway) -> None:
    async with make_cas(gateway_a, verify_content=True) as cas:
        assert await cas.load(GOOD_CID) == BODY


@pytest.mark.asyncio
async def test_verification_off_by_default_passes_bad_bytes_through(
    gateway_a: FakeGateway,
) -> None:
    """Opt-in: the default path must not change behaviour or cost a hash."""
    gateway_a.responder = lambda _cid: (200, b"corrupted payload")

    async with make_cas(gateway_a) as cas:
        assert await cas.load(GOOD_CID) == b"corrupted payload"


@pytest.mark.asyncio
async def test_range_reads_are_not_verified(gateway_a: FakeGateway) -> None:
    """A Range response is a slice and cannot hash to the CID."""
    gateway_a.responder = lambda _cid: (200, BODY)

    async with make_cas(gateway_a, verify_content=True) as cas:
        # The stub ignores Range and returns 200, so load slices locally.
        assert await cas.load(GOOD_CID, offset=0, length=5) == BODY[:5]


def test_dag_pb_cids_are_not_verifiable() -> None:
    """Gateways return reassembled UnixFS files for dag-pb, not the block."""
    dag_pb_cid = CID("base32", 1, "dag-pb", multihash.digest(BODY, "sha2-256"))
    assert not store_module._cid_is_verifiable(dag_pb_cid, None, None)
    assert store_module._cid_is_verifiable(GOOD_CID, None, None)
    assert not store_module._cid_is_verifiable(GOOD_CID, 0, None)
    assert not store_module._cid_is_verifiable(GOOD_CID, None, 4)


def test_unsupported_hash_function_is_not_a_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """We cannot prove content wrong with a hash we cannot compute."""

    def unsupported(*_args: object, **_kwargs: object) -> bytes:
        raise KeyError("unsupported multihash")

    monkeypatch.setattr(store_module.multihash, "digest", unsupported)
    store_module._verify_cid_content(GOOD_CID, b"anything at all")
