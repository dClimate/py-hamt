import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path

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


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))  # Bind to a free port provided by the host.
        return int(s.getsockname()[1])  # Return the port number assigned.


@pytest.fixture(scope="module")
def create_ipfs():
    # Create temporary directory, set it as the IPFS Path
    temp_dir = Path(tempfile.mkdtemp())
    custom_env = os.environ.copy()
    custom_env["IPFS_PATH"] = str(temp_dir)

    # IPFS init
    subprocess.run(
        ["ipfs", "init", "--profile", "pebbleds"], check=True, env=custom_env
    )

    # Edit the config file so that it serves on randomly selected and available ports to not conflict with any currently running ipfs daemons
    swarm_port = find_free_port()
    rpc_port = find_free_port()
    gateway_port = find_free_port()

    config_path = temp_dir / "config"
    config: dict
    with open(config_path, "r") as f:
        config = json.load(f)

    swarm_addrs: list[str] = config["Addresses"]["Swarm"]
    new_port_swarm_addrs = [s.replace("4001", str(swarm_port)) for s in swarm_addrs]
    config["Addresses"]["Swarm"] = new_port_swarm_addrs

    rpc_multiaddr = config["Addresses"]["API"]
    gateway_multiaddr = config["Addresses"]["Gateway"]

    config["Addresses"]["API"] = rpc_multiaddr.replace("5001", str(rpc_port))
    config["Addresses"]["Gateway"] = gateway_multiaddr.replace(
        "8080", str(gateway_port)
    )

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    # Start the daemon
    rpc_uri_stem = f"http://127.0.0.1:{rpc_port}"
    gateway_uri_stem = f"http://127.0.0.1:{gateway_port}"

    ipfs_process = subprocess.Popen(["ipfs", "daemon"], env=custom_env)
    # Should be enough time for the ipfs daemon process to start up
    time.sleep(5)

    yield rpc_uri_stem, gateway_uri_stem

    # Close the daemon
    ipfs_process.kill()

    # Delete the temporary directory
    shutil.rmtree(temp_dir)
