from typing import Callable
from pathlib import Path

from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

# Metadata files used in zarr v2
_metadata_files = [
    # top level metadata
    ".zattrs",  # Also found within folders for variables
    ".zgroup",
    ".zmetadata",
    # Found within folders for variables
    ".zarray",
    # important for coordinate variables, so that we can read bounds
    "0",
]

type TransformerFN = Callable[[str, bytes], bytes]


def create_zarr_encryption_transformers(
    encryption_key: bytes,
    header: bytes,
    exclude_vars: list[str] = [],
) -> tuple[TransformerFN, TransformerFN]:
    """
    Uses XChaCha20_Poly1305 from the pycryptodome library to perform encryption, while ignoring zarr metadata files.

    https://pycryptodome.readthedocs.io/en/latest/src/cipher/chacha20_poly1305.html

    Note that the encryption key must always be 32 bytes long. A header is required by the underlying encryption algorithm. Every time a zarr chunk is encrypted, a random 24-byte nonce is generated. This is saved with the chunk for use when reading back.

    Metadata within a zarr, such as ".zattrs" or ".zgroup" are always ignored, to allow for calculating an encrypted zarr's structure without necessarily having the encryption key. You may also set some variables to be entirely unencrypted with the exclude_vars argument. This allows for partially encrypted zarrs which can be loaded into xarray but the values of encrypted variables cannot be accessed (errors will be thrown).
    """

    if len(encryption_key) != 32:
        raise ValueError("Encryption key is not 32 bytes")

    def _should_transform(key: str) -> bool:
        p = Path(key)
        if p.parent.name in exclude_vars:
            return False
        if p.name in _metadata_files:
            return False
        return True

    def encrypt(key: str, val: bytes) -> bytes:
        if not _should_transform(key):
            return val

        nonce = get_random_bytes(24)
        cipher = ChaCha20_Poly1305.new(key=encryption_key, nonce=nonce)
        cipher.update(header)
        ciphertext, tag = cipher.encrypt_and_digest(val)
        # + concatenates two byte variables x,y so that it looks like xy
        return nonce + tag + ciphertext

    def decrypt(key: str, val: bytes) -> bytes:
        if not _should_transform(key):
            return val

        nonce, tag, ciphertext = val[:24], val[24:40], val[40:]
        cipher = ChaCha20_Poly1305.new(key=encryption_key, nonce=nonce)
        cipher.update(header)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext

    return (encrypt, decrypt)
