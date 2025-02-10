from typing import Callable
from pathlib import Path


import io

from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

import functools
import hashlib
import io

from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

from numcodecs import register_codec
from numcodecs.abc import Codec
from typing import Any
import xarray as xr


class EncryptionCodec(Codec):
    codec_id = "xchacha20poly1305"
    # header = b"dClimate-Zarr"
    _encryption_key = None 
    _encoded_header = None

    def __init__(self, header: str):
        print(header, "BEING USED")
        self.header = header
        self._encoded_header = header.encode()
        if self._encryption_key is None:
            raise ValueError("Encryption key must be set before using EncryptionCodec.")

    @classmethod
    def set_encryption_key(cls, encryption_key: str):
        """Set the encryption key dynamically (once per runtime)."""
        cls._encryption_key = bytes.fromhex(encryption_key)

    def encode(self, buf):
        raw = io.BytesIO()
        raw.write(buf)
        nonce = get_random_bytes(24)  # XChaCha uses 24-byte (192-bit) nonce
        cipher = ChaCha20_Poly1305.new(key=self._encryption_key, nonce=nonce)
        cipher.update(self._encoded_header)
        ciphertext, tag = cipher.encrypt_and_digest(raw.getbuffer())

        return nonce + tag + ciphertext

    def decode(self, buf, out=None):
        nonce, tag, ciphertext = buf[:24], buf[24:40], buf[40:]
        cipher = ChaCha20_Poly1305.new(key=self._encryption_key, nonce=nonce)
        cipher.update(self._encoded_header)
        plaintext = cipher.decrypt_and_verify(ciphertext, tag)
        return plaintext

    @classmethod
    def from_config(cls, config):
        """Prevent requiring encryption key from metadata."""
        header = config.get("header", b"dClimate-Zarr")
        return cls(header=header)  # Just return an instance, without needing a key

class MissingKeyError(Exception):
    def __init__(self, key):
        super().__init__(f"Cannot find encryption key with hash: {key}")
