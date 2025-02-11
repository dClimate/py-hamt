from typing import Callable
from pathlib import Path


import io

from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes


# Metadata files to avoid encrypting (and decrypting)
encryption_exclude_files = [
    # top level meta data
    ".zattrs",
    ".zgroup",
    ".zmetadata",
    # found within variables, this includes .zattrs
    ".zarray",
    # important for coordinate variables, so that we can read bounds
    "0",
]

TransformerFN = Callable[[str, bytes], bytes]


def create_zarr_encryption_transformers(
    encryption_key: bytes,
    encrypted_vars: [str],
) -> tuple[TransformerFN, TransformerFN]:
    """An encryption filter for ZARR data.
    This class is serialized and stored along with the Zarr it is used with, so instead
    of storing the encryption key, we store the hash of the encryption key, so it can be
    looked up in the key registry at run time as needed.
    Parameters
    ----------
    key_hash: str
        The hex digest of an encryption key. A key may be generated using
        :func:`generate_encryption_key`. The hex digest is obtained by registering the
        key using :func:`register_encryption_key`.
    """

    # codec_id = "xchacha20poly1305"
    header = b"dClimate-Zarr"

    def _should_transform_key(key: str) -> bool:
        if Path(key).name in encryption_exclude_files:
            return False
        return key.split("/")[0] in encrypted_vars

    def encode(key: str, val: bytes) -> bytes:
        if not _should_transform_key(key):
            return val
        raw = io.BytesIO()
        raw.write(val)
        nonce = get_random_bytes(24)  # XChaCha uses 24 byte (192 bit) nonce
        cipher = ChaCha20_Poly1305.new(key=encryption_key, nonce=nonce)
        cipher.update(header)
        ciphertext, tag = cipher.encrypt_and_digest(raw.getbuffer())

        return nonce + tag + ciphertext

    def decode(key: str, val: bytes):
        if not _should_transform_key(key):
            return val

        nonce, tag, ciphertext = val[:24], val[24:40], val[40:]
        cipher = ChaCha20_Poly1305.new(key=encryption_key, nonce=nonce)
        cipher.update(header)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)

        # if out is not None:
        #     outbuf = io.BytesIO(plaintext)
        #     outbuf.readinto(out)
        #     return out

        return plaintext

    return encode, decode
