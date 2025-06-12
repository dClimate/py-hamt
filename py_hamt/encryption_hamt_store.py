from typing import cast

import zarr.abc.store
import zarr.core.buffer
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

from py_hamt.hamt import HAMT
from py_hamt.zarr_hamt_store import ZarrHAMTStore


class SimpleEncryptedZarrHAMTStore(ZarrHAMTStore):
    """
    Write and read Zarr v3s with a HAMT, encrypting *everything* for maximum privacy.

    This store uses ChaCha20-Poly1305 to encrypt every single key-value pair
    stored in the Zarr, including all metadata (`zarr.json`, `.zarray`, etc.)
    and data chunks. This provides strong privacy but means the Zarr store is
    completely opaque and unusable without the correct encryption key and header.

    Note: For standard zarr encryption and decryption where metadata is available use the method in https://github.com/dClimate/jupyter-notebooks/blob/main/notebooks/202b%20-%20Encryption%20Example%20(Encryption%20with%20Zarr%20Codecs).ipynb

    #### Encryption Details
    - Uses XChaCha20_Poly1305 (via pycryptodome's ChaCha20_Poly1305 with a 24-byte nonce).
    - Requires a 32-byte encryption key and a header.
    - Each encrypted value includes a 24-byte nonce and a 16-byte tag.

    #### Important Considerations
    - Since metadata is encrypted, standard Zarr tools cannot inspect the
      dataset without prior decryption using this store class.
    - There is no support for partial encryption or excluding variables.
    - There is no metadata caching.

    #### Sample Code
    ```python
    import xarray
    from py_hamt import HAMT, KuboCAS # Assuming an KuboCAS or similar
    from Crypto.Random import get_random_bytes
    import numpy as np

    # Setup
    ds = xarray.Dataset(
        {"data": (("y", "x"), np.arange(12).reshape(3, 4))},
        coords={"y": [1, 2, 3], "x": [10, 20, 30, 40]}
    )
    cas = KuboCAS() # Example ContentAddressedStore
    encryption_key = get_random_bytes(32)
    header = b"fully-encrypted-zarr"

    # --- Write ---
    hamt_write = await HAMT.build(cas=cas, values_are_bytes=True)
    ezhs_write = SimpleEncryptedZarrHAMTStore(
        hamt_write, False, encryption_key, header
    )
    print("Writing fully encrypted Zarr...")
    ds.to_zarr(store=ezhs_write, mode="w")
    await hamt_write.make_read_only()
    root_node_id = hamt_write.root_node_id
    print(f"Wrote Zarr with root: {root_node_id}")

    # --- Read ---
    hamt_read = await HAMT.build(
            cas=cas, root_node_id=root_node_id, values_are_bytes=True, read_only=True
        )
    ezhs_read = SimpleEncryptedZarrHAMTStore(
        hamt_read, True, encryption_key, header
    )
    print("\nReading fully encrypted Zarr...")
    ds_read = xarray.open_zarr(store=ezhs_read)
    print("Read back dataset:")
    print(ds_read)
    xarray.testing.assert_identical(ds, ds_read)
    print("Read successful and data verified.")

    # --- Read with wrong key (demonstrates failure) ---
    wrong_key = get_random_bytes(32)
    hamt_bad = await HAMT.build(
        cas=cas, root_node_id=root_node_id, read_only=True, values_are_bytes=True
    )
    ezhs_bad = SimpleEncryptedZarrHAMTStore(
        hamt_bad, True, wrong_key, header
    )
    print("\nAttempting to read with wrong key...")
    try:
        ds_bad = xarray.open_zarr(store=ezhs_bad)
        print(ds_bad)
    except Exception as e:
        print(f"Failed to read as expected: {type(e).__name__} - {e}")
    ```
    """

    def __init__(
        self, hamt: HAMT, read_only: bool, encryption_key: bytes, header: bytes
    ) -> None:
        """
        Initializes the SimpleEncryptedZarrHAMTStore.

        Args:
            hamt: The HAMT instance for storage. Must have `values_are_bytes=True`.
                  Its `read_only` status must match the `read_only` argument.
            read_only: If True, the store is in read-only mode.
            encryption_key: A 32-byte key for ChaCha20-Poly1305.
            header: A header (bytes) used as associated data in encryption.
        """
        super().__init__(hamt, read_only=read_only)

        if len(encryption_key) != 32:
            raise ValueError("Encryption key must be exactly 32 bytes long.")
        self.encryption_key = encryption_key
        self.header = header
        self.metadata_read_cache: dict[str, bytes] = {}

    def _encrypt(self, val: bytes) -> bytes:
        """Encrypts data using ChaCha20-Poly1305."""
        nonce = get_random_bytes(24)  # XChaCha20 uses a 24-byte nonce
        cipher = ChaCha20_Poly1305.new(key=self.encryption_key, nonce=nonce)
        cipher.update(self.header)
        ciphertext, tag = cipher.encrypt_and_digest(val)
        return nonce + tag + ciphertext

    def _decrypt(self, val: bytes) -> bytes:
        """Decrypts data using ChaCha20-Poly1305."""
        try:
            # Extract nonce (24), tag (16), and ciphertext
            nonce, tag, ciphertext = val[:24], val[24:40], val[40:]
            cipher = ChaCha20_Poly1305.new(key=self.encryption_key, nonce=nonce)
            cipher.update(self.header)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext
        except Exception as e:
            # Catching a broad exception as various issues (key, tag, length) can occur.
            raise ValueError(
                "Decryption failed. Check key, header, or data integrity."
            ) from e

    def __eq__(self, other: object) -> bool:
        """@private"""
        if not isinstance(other, SimpleEncryptedZarrHAMTStore):
            return False
        return (
            self.hamt.root_node_id == other.hamt.root_node_id
            and self.encryption_key == other.encryption_key
            and self.header == other.header
        )

    async def get(
        self,
        key: str,
        prototype: zarr.core.buffer.BufferPrototype,
        byte_range: zarr.abc.store.ByteRequest | None = None,
    ) -> zarr.core.buffer.Buffer | None:
        """@private"""
        try:
            decrypted_val: bytes
            is_metadata: bool = (
                len(key) >= 9 and key[-9:] == "zarr.json"
            )  # if path ends with zarr.json

            if is_metadata and key in self.metadata_read_cache:
                decrypted_val = self.metadata_read_cache[key]
            else:
                raw_val = cast(
                    bytes, await self.hamt.get(key)
                )  # We know values received will always be bytes since we only store bytes in the HAMT
                decrypted_val = self._decrypt(raw_val)
                if is_metadata:
                    self.metadata_read_cache[key] = decrypted_val
            return prototype.buffer.from_bytes(decrypted_val)
        except KeyError:
            return None

    async def set(self, key: str, value: zarr.core.buffer.Buffer) -> None:
        """@private"""
        if self.read_only:
            raise Exception("Cannot write to a read-only store.")

        raw_bytes = value.to_bytes()
        if key in self.metadata_read_cache:
            self.metadata_read_cache[key] = raw_bytes
        # Encrypt it
        encrypted_bytes = self._encrypt(raw_bytes)
        await self.hamt.set(key, encrypted_bytes)
