import json
from typing import Callable, Literal

import dag_cbor
import xarray as xr
from Crypto.Cipher import ChaCha20_Poly1305
from Crypto.Random import get_random_bytes

type TransformerFN = Callable[[str, bytes], bytes]


def create_zarr_encryption_transformers(
    encryption_key: bytes,
    header: bytes,
    exclude_vars: list[str] = [],
    detect_exclude: xr.Dataset
    | Literal["auto-from-read"]
    | Literal[False] = "auto-from-read",
) -> tuple[TransformerFN, TransformerFN]:
    """
    Uses XChaCha20_Poly1305 from the pycryptodome library to perform encryption, while ignoring zarr metadata files.

    https://pycryptodome.readthedocs.io/en/latest/src/cipher/chacha20_poly1305.html

    Note that the encryption key must be exactly 32 bytes long. A header is required by the underlying encryption algorithm. Every time a zarr chunk is encrypted, a random 24-byte nonce is generated. This is saved with the chunk for use when reading back.

    zarr.json metadata files in a zarr v3 are always ignored and passed through unencrypted.

    With `exclude_vars` you may also set some variables to be unencrypted. This allows for partially encrypted zarrs. This should generally include your coordinate variables, along with any data variables you want to keep open.

    `detect_exclude` allows you to put in a xarray Dataset. This will be used to automatically add coordinate variables to the exclusion list. When you reading back a dataset and you do not know the unencrypted variables ahead of time, you can set this to the default "auto-from-read", which will attempt to use any metadata or any decryption errors to detect unencrypted variables.

    To do no automatic detection, set `detect_exclude` to False.

    # Example code
    ```python
    from py_hamt import HAMT, IPFSStore, IPFSZarr3, create_zarr_encryption_transformers

    ds = ... # example xarray Dataset with precip and temp data variables
    encryption_key = bytes(32) # change before using, only for demonstration purposes!
    header = "sample-header".encode()
    encrypt, decrypt = create_zarr_encryption_transformers(
        encryption_key, header, exclude_vars=["temp"], detect_exclude=ds
    )
    hamt = HAMT(
        store=IPFSStore(), transformer_encode=encrypt, transformer_decode=decrypt
    )
    ipfszarr3 = IPFSZarr3(hamt)
    ds.to_zarr(store=ipfszarr3, mode="w")

    print("Attempting to read and print metadata of partially encrypted zarr")
    wrong_key = bytes([0xAA]) * 32
    wrong_header = "".encode()
    bad_encrypt, auto_detecting_decrypt = create_zarr_encryption_transformers(
        wrong_key,
        wrong_header,
    )
    hamt = HAMT(store=IPFSStore(), transformer_encode=bad_encrypt, transformer_decode=auto_detecting_decrypt, root_node_id=ipfszarr3.hamt.root_node_id)
    ipfszarr3 = IPFSZarr3(hamt, read_only=True)
    enc_ds = xr.open_zarr(store=ipfszarr3)
    print(enc_ds)
    assert enc_ds.temp.sum() == ds.temp.sum()
    try:
        enc_ds.precip.sum()
    except:
        print("Couldn't read encrypted variable")
    ```
    """
    if len(encryption_key) != 32:
        raise ValueError("Encryption key is not 32 bytes")

    exclude_var_set = set(exclude_vars)

    if isinstance(detect_exclude, xr.Dataset):
        ds = detect_exclude
        for coord in list(ds.coords):
            exclude_var_set.add(coord)  # type: ignore

    def _should_transform(key: str) -> bool:
        # Find the first directory name in the path since zarr v3 chunks are stored in a nested directory structure
        # e.g. for "precip/c/0/0/1" this would find "precip"
        first_slash = key.find("/")
        if first_slash != -1:
            var_name = key[0:first_slash]
            if var_name in exclude_var_set:
                return False

        # Don't transform metadata files, even for encrypted variables
        if key[-9:] == "zarr.json":
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

    seen_metadata: set[str] = set()

    def decrypt(key: str, val: bytes) -> bytes:
        # Look through files, this relies on the fact that xarray itself will attempt to read the root zarr.json and other metadata files first before any data will ever be accessed
        # Important that this goes before _should_transform since that will return before we get a chance to look at metadata, and it needs information that we can glean here
        if (
            detect_exclude == "auto-from-read"
            and key[-9:] == "zarr.json"
            and key not in seen_metadata
        ):
            seen_metadata.add(key)

            # Assume the zarr.json is unencrypted, which it should be if made with zarr encryption transformers
            metadata = json.loads(dag_cbor.decode(val))  # type: ignore

            # If the global zarr.json, check if it has the list of coordinates in the consolidated metadata
            if (
                "consolidated_metadata" in metadata
                and metadata["consolidated_metadata"] is not None
            ):
                variables = metadata["consolidated_metadata"]["metadata"]
                for var in variables:
                    for dimension in variables[var]["dimension_names"]:
                        exclude_var_set.add(dimension)
            # Otherwise just scan a variable's individual metadata, but first make sure it's not the root zarr.json
            elif "dimension_names" in metadata:
                for dimension in metadata["dimension_names"]:
                    exclude_var_set.add(dimension)

        if not _should_transform(key):
            return val

        try:
            nonce, tag, ciphertext = val[:24], val[24:40], val[40:]
            cipher = ChaCha20_Poly1305.new(key=encryption_key, nonce=nonce)
            cipher.update(header)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            return plaintext
        except Exception as e:
            # If if we are auto detecting coordinates, and there's an error with decrypting, then assume the issue is that this is a partially encrypted zarr, so we need to mark this variable as being one of the unencrypted ones and return like normal
            if detect_exclude == "auto-from-read":
                first_slash = key.find("/")
                if first_slash != -1:
                    var_name = key[0:first_slash]
                    exclude_var_set.add(var_name)
                return val
            else:
                raise e

    return (encrypt, decrypt)
