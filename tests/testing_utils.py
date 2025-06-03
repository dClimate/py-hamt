import http.client
import os
import socket
import time
from urllib.parse import urlparse

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
    if not p.hostname:
        return False
    try:
        conn = http.client.HTTPConnection(p.hostname, p.port, timeout=1)
        conn.request("POST", "/api/v0/version")
        return conn.getresponse().status == 200
    except Exception:
        return False


def _gw_is_up(url: str) -> bool:
    """HEAD / on the *gateway* port (8080 by default)."""
    p = urlparse(url)
    if not p.hostname:
        return False
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
    """Try to get a working Docker client, with macOS Docker Desktop support."""

    if docker is None:
        return None

    # Common Docker socket locations (in order of preference)
    socket_locations = [
        # Docker Desktop on macOS
        f"unix://{os.path.expanduser('~')}/.docker/run/docker.sock",
        # Docker Desktop alternative location
        f"unix://{os.path.expanduser('~')}/.docker/desktop/docker.sock",
        # Standard Linux locations
        "unix:///var/run/docker.sock",
        "unix:///run/docker.sock",
    ]

    # First, try with DOCKER_HOST if it's already set
    existing_host = os.environ.get("DOCKER_HOST")
    if existing_host:
        try:
            c = docker.DockerClient(base_url=existing_host)
            c.ping()
            return c
        except Exception:
            pass

    # Try each known socket location
    for socket_path in socket_locations:
        try:
            # Check if the socket file exists (for unix sockets)
            if socket_path.startswith("unix://"):
                socket_file = socket_path.replace("unix://", "")
                if not os.path.exists(socket_file):
                    continue

            c = docker.DockerClient(base_url=socket_path)
            c.ping()
            # If successful, set DOCKER_HOST for any child processes
            os.environ["DOCKER_HOST"] = socket_path
            return c
        except Exception:
            continue

    # Last resort: try docker.from_env() which might work with some setups
    try:
        c = docker.from_env()
        c.ping()
        return c
    except Exception:
        pass

    return None


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
    client = _docker_client_or_none()
    if client is None:
        pytest.skip("Neither IPFS daemon nor Docker available – skipping IPFS tests")

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
        # Wait for container to be ready
        for _ in range(30):  # 30 attempts, 0.5s each = 15s max
            if _rpc_is_up(f"http://127.0.0.1:{rpc_p}") and _gw_is_up(
                f"http://127.0.0.1:{gw_p}"
            ):
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("IPFS container failed to start within timeout")

        yield f"http://127.0.0.1:{rpc_p}", f"http://127.0.0.1:{gw_p}"
    finally:
        container.stop(timeout=3)
