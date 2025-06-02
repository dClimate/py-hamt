from urllib.parse import urlparse
import os
import socket
import time

import http.client

import pytest
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy
from multiformats import CID


def cid_strategy() -> SearchStrategy:
    """Generate random CIDs for testing."""

    # Strategy for generating random hash digests
    digests = st.binary(min_size=32, max_size=32).map(
        lambda d: bytes.fromhex("1220") + d  # 0x12 = sha2-256, 0x20 = 32 bytes
    )

    # Generate CIDv1 (more flexible)
    cidv1 = st.tuples(st.just("base32"), st.just(1), st.just("dag-cbor"), digests)

    # Combine the strategies and create CIDs
    return cidv1.map(lambda args: CID(*args))


def ipld_strategy() -> SearchStrategy:
    return st.one_of(
        [
            st.none(),
            st.booleans(),
            st.integers(min_value=-9223372036854775808, max_value=9223372036854775807),
            st.floats(allow_infinity=False, allow_nan=False),
            st.text(),
            st.binary(),
            cid_strategy(),
        ]
    )


key_value_list = st.lists(
    st.tuples(st.text(), ipld_strategy()),
    min_size=0,
    max_size=10000,
    unique_by=lambda x: x[
        0
    ],  # ensure unique keys, otherwise we can't do the length and size checks when using these KVs for the HAMT
)

# ---------- helpers ---------------------------------------------------------


def _rpc_is_up(url: str) -> bool:
    """POST /api/v0/version on the *RPC* port (5001 by default)."""
    p = urlparse(url)
    try:
        conn = http.client.HTTPConnection(p.hostname, p.port, timeout=1)
        conn.request("POST", "/api/v0/version")
        return conn.getresponse().status == 200
    except Exception:
        return False


def _gw_is_up(url: str) -> bool:
    """HEAD / on the *gateway* port (8080 by default)."""
    p = urlparse(url)
    try:
        conn = http.client.HTTPConnection(p.hostname, p.port, timeout=1)
        conn.request("HEAD", "/")
        return conn.getresponse().status in (200, 404)  # 404 = empty root
    except Exception:
        return False


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


try:
    import docker
except ImportError:
    docker = None


def _docker_client_or_none():
    # 1. first try env / default socket
    try:
        c = docker.from_env()
        c.ping()
        return c
    except docker.errors.DockerException:
        pass

    # 2. fall back to the *current* Docker context (Desktop, Colima, etc.)
    try:
        ctx_name = subprocess.check_output(
            ["docker", "context", "show"], text=True
        ).strip()
        host = subprocess.check_output(
            [
                "docker",
                "context",
                "inspect",
                "--format",
                '{{ index .Endpoints "docker" "Host" }}',
                ctx_name,
            ],
            text=True,
        ).strip()
        if host:
            os.environ["DOCKER_HOST"] = host  # make visible to children
            c = docker.DockerClient(base_url=host)
            c.ping()
            return c
    except Exception:
        pass

    return None  # fixture will pytest.skip()


# ---------- fixture ---------------------------------------------------------


@pytest.fixture(scope="session")
def create_ipfs():
    """Yield `(rpc_url, gateway_url)`.

    Order of preference:
    1.  reuse an already-running daemon (checked via RPC probe)
    2.  launch Docker container (if docker is installed & running)
    3.  skip the IPFS-marked tests
    """
    rpc = os.getenv("IPFS_RPC_URL") or "http://127.0.0.1:5001"
    gw = os.getenv("IPFS_GATEWAY_URL") or "http://127.0.0.1:8080"

    # 1. reuse existing node -------------------------------------------------
    if _rpc_is_up(rpc) and _gw_is_up(gw):
        yield rpc, gw
        return

    # 2. fall back to Docker -------------------------------------------------
    # if docker is None:
    #     pytest.skip("Docker not available and no IPFS node listening")
    client = _docker_client_or_none()
    if client is None:
        pytest.skip("Neither IPFS daemon nor Docker available – skipping IPFS tests")

    client = docker.from_env()
    image = "ipfs/kubo:v0.35.0"
    rpc_p = _free_port()
    gw_p = _free_port()

    container = client.containers.run(
        image,
        "daemon --init --offline",
        ports={"5001/tcp": rpc_p, "8080/tcp": gw_p},
        detach=True,
        auto_remove=True,
    )

    try:
        time.sleep(6)  # crude readiness wait
        yield f"http://127.0.0.1:{rpc_p}", f"http://127.0.0.1:{gw_p}"
    finally:
        container.stop(timeout=3)
