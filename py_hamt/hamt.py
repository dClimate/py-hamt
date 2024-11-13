from typing import Callable
from threading import Lock

import dag_cbor
from dag_cbor.ipld import IPLDKind
from multiformats import multihash

from .store import Store


def extract_bits(hash_bytes: bytes, depth: int, nbits: int) -> int:
    """Extract `nbits` bits from `hash_bytes`, beginning at position `depth * nbits`,
    and convert them into an unsigned integer value.

    Args:
        hash_bytes (bytes): binary hash to extract bit sequence from
        depth (int): depth of the node containing the hash
        nbits (int): bit width of hash

    Returns:
        int: An unsigned integer version of the bit sequence
    """
    # This is needed since int.bit_length on a integer representation of a bytes object ignores leading 0s
    hash_bit_length = len(hash_bytes) * 8

    start_bit_index = depth * nbits

    if hash_bit_length - start_bit_index < nbits:
        raise IndexError("Arguments extract more bits than remain in the hash bits")

    mask = (0b1 << (hash_bit_length - start_bit_index)) - 1
    n_chop_off_at_end = int.bit_length(mask) - nbits

    hash_as_int = int.from_bytes(hash_bytes)
    result = (hash_as_int & mask) >> n_chop_off_at_end

    return result


class Node:
    # Use a dict for easy serializability with dag_cbor
    data: dict[str, IPLDKind]

    def __init__(self):
        data = {}

        # Each "string" key for both buckets and CIDs is the bits as a string, e.g. str(0b1101) = '13'
        buckets: dict[str, list[dict[str, bytes]]] = {}
        data["b"] = buckets
        links: dict[str, bytes] = {}
        data["c"] = links
        self.data = data

    def _replace_link(self, old_link: bytes, new_link: bytes):
        links: dict[str, bytes] = self.data["c"]  # type: ignore
        for str_key in list(links.keys()):
            link = links[str_key]
            if link == old_link:
                links[str_key] = new_link

    def _remove_link(self, old_link: bytes):
        links: dict[str, bytes] = self.data["c"]  # type: ignore
        for str_key in list(links.keys()):
            link = links[str_key]
            if link == old_link:
                del links[str_key]

    def serialize(self) -> bytes:
        return dag_cbor.encode(self.data)

    @classmethod
    def deserialize(cls, data: bytes) -> "Node":
        decoded = dag_cbor.decode(data)
        if (
            isinstance(decoded, dict)
            and "b" in decoded
            and isinstance(decoded["b"], dict)
            and "c" in decoded
            and isinstance(decoded["c"], dict)
        ):
            node = cls()
            node.data = decoded
            return node
        else:
            raise ValueError("Data not a valid Node serialization")


class Hamt:
    store: Store
    # Every call of this will look like self.hash_fn() and so the first argument will always take a self argument
    hash_fn: Callable[["Hamt", bytes], bytes]
    # Only important for writing, when reading this will use buckets even if they are overly big
    max_bucket_size: int

    root_node_id: bytes
    key_count: int

    # Modifying a HAMT is not multithreading safe
    lock: Lock
    # Allows for multithreaded access since you no longer have to worry about setting operations, acquiring a lock and changing the root node and its links
    read_only: bool

    # We have to use create instead of __init__ since __init__ cannot be async
    @classmethod
    async def create(
        cls,
        store: Store,
        hash_fn: Callable[["Hamt", bytes], bytes],
        max_bucket_size: int = 4,
        read_only: bool = False,
    ) -> "Hamt":
        instance = cls()
        cls.store = store
        cls.hash_fn = hash_fn

        root_node = Node()
        cls.root_node_id = await store.save(root_node.serialize())

        cls.key_count = 0

        cls.max_bucket_size = max_bucket_size
        cls.read_only = read_only
        cls.lock = Lock()

        return instance

    def make_read_only(self):
        self.lock.acquire(blocking=True)
        self.read_only = True
        self.lock.release()

    def enable_write(self):
        self.lock.acquire(blocking=True)
        self.read_only = False
        self.lock.release()

    """
    For internal use
    This function starts from the node at the end of the list and reserializes so that each node holds valid new IDs after insertion into the store
    Takes a stack of nodes, we represent a stack with a list where the first element is the root element and the last element is the top of the stack
    Each element in the list is a tuple where the first element is the ID from the store and the second element is the Node in python
    If a node ends up being empty, then it is deleted entirely, unless it is the root node
    Modifies in place
    """

    async def _reserialize_and_link(self, node_stack: list[tuple[bytes, Node]]):
        # Iterate in the reverse direction
        for stack_index in range(len(node_stack) - 1, -1, -1):
            old_store_id, node = node_stack[stack_index]

            # If this node is empty, and its not the root node, then we can delete it entirely from the list
            buckets: dict[str, list[dict[str, bytes]]] = node.data["b"]  # type: ignore
            links: dict[str, bytes] = node.data["c"]  # type: ignore
            is_empty = len(buckets) == 0 and len(links) == 0

            if is_empty and stack_index > 0:
                # Unlink from the rest of the tree
                _, prev_node = node_stack[stack_index - 1]
                prev_node._remove_link(old_store_id)

                # Remove from our stack
                node_stack.pop(stack_index)
            else:
                # Reserialize
                new_store_id = await self.store.save(node.serialize())
                node_stack[stack_index] = (new_store_id, node)

                # If this is not the last i.e. root node, we need to change the linking of the node prior in the list
                if stack_index > 0:
                    _, prev_node = node_stack[stack_index - 1]
                    prev_node._replace_link(old_store_id, new_store_id)

    async def set(self, key_to_insert: str, val_to_insert: bytes):
        if self.read_only:
            raise Exception("Cannot call set on a read only HAMT")

        if not self.read_only:
            self.lock.acquire(blocking=True)

        node_stack: list[tuple[bytes, Node]] = []
        root_node = Node.deserialize(await self.store.load(self.root_node_id))
        node_stack.append((self.root_node_id, root_node))

        # FIFO queue to keep track of all the KVs we need to insert
        # This is important for if any buckets overflow and thus we need to reinsert all those KVs
        kvs_queue: list[tuple[str, bytes]] = []
        kvs_queue.append((key_to_insert, val_to_insert))

        created_change = False
        # Keep iterating until we have no more KVs to insert
        while len(kvs_queue) != 0:
            _, top_node = node_stack[-1]
            curr_key, curr_val = kvs_queue[0]

            raw_hash = self.hash_fn(bytes(curr_key, "utf-8"))  # type: ignore
            map_key = str(extract_bits(raw_hash, len(node_stack), 8))

            buckets: dict[str, list[dict[str, bytes]]] = top_node.data["b"]  # type: ignore
            links: dict[str, bytes] = top_node.data["c"]  # type: ignore

            if map_key in links and map_key in buckets:
                self.lock.release()
                raise Exception(
                    "Key in both buckets and links of the node, invariant violated"
                )
            # We do not need to check this since new nodes that are created when buckets are too full are pushed on top of the stuck, as if we had traversed down the links already
            # elif map_key in links:
            elif map_key in buckets:
                created_change = True
                bucket = buckets[map_key]

                bucket_has_space = len(bucket) < self.max_bucket_size

                # If this bucket already has this same key, just rewrite the value
                # Since we can't use continues to go back to the top of the while loop, use this boolean flag instead
                should_continue_at_while = False
                for kv in bucket:
                    if curr_key in kv:
                        kv[curr_key] = curr_val
                        kvs_queue.pop(0)
                        should_continue_at_while = True
                        break
                if should_continue_at_while:
                    continue

                if bucket_has_space:
                    bucket.append({curr_key: curr_val})
                    kvs_queue.pop(0)
                    self.key_count += 1
                # If bucket is full and we need to add, then all these KVs need to be taken out of this bucket and reinserted throughout the tree
                else:
                    # Empty the bucket of KVs into the queue
                    for kv in bucket:
                        for k, v in kv.items():
                            kvs_queue.append((k, v))
                            self.key_count -= 1

                    # Delete empty bucket, there should only be a link now
                    del buckets[map_key]

                    # Create a new link to a new node so that we can reflow these KVs into new buckets
                    new_node = Node()
                    new_node_id = await self.store.save(new_node.serialize())

                    links[map_key] = new_node_id

                    # We need to rerun from the top with the new queue, but this time this node will have a link to put KVs deeper down in the tree
                    node_stack.append((new_node_id, new_node))
            else:
                # If there is no link and no bucket, then we can create a new bucket, insert, and be done with this key
                bucket: list[dict[str, bytes]] = []
                bucket.append({curr_key: curr_val})
                kvs_queue.pop(0)
                buckets[map_key] = bucket
                self.key_count += 1
                created_change = True

        # Finally, reserialize and fix all links, deleting empty nodes as needed
        if created_change:
            await self._reserialize_and_link(node_stack)
            self.root_node_id = node_stack[0][0]

        if not self.read_only:
            self.lock.release()

    async def delete(self, key: str):
        if self.read_only:
            raise Exception("Cannot call delete on a read only HAMT")

        if not self.read_only:
            self.lock.acquire(blocking=True)

        raw_hash = self.hash_fn(bytes(key, "utf-8"))  # type: ignore

        node_stack: list[tuple[bytes, Node]] = []
        root_node = Node.deserialize(await self.store.load(self.root_node_id))
        node_stack.append((self.root_node_id, root_node))

        created_change = False
        while True:
            top_id, top_node = node_stack[-1]
            map_key = str(extract_bits(raw_hash, len(node_stack), 8))

            buckets: dict[str, list[dict[str, bytes]]] = top_node.data["b"]  # type: ignore
            links: dict[str, bytes] = top_node.data["c"]  # type: ignore

            if map_key in buckets and map_key in links:
                self.lock.release()
                raise Exception(
                    "Key in both buckets and links of the node, invariant violated"
                )
            elif map_key in buckets:
                bucket = buckets[map_key]

                # Delete from within this bucket
                for bucket_index in range(len(bucket)):
                    kv = bucket[bucket_index]
                    if key in kv:
                        created_change = True
                        bucket.pop(bucket_index)

                        # If this bucket becomes empty then delete this dict entry for the bucket
                        if len(bucket) == 0:
                            del buckets[map_key]

                        # This must be done to avoid IndexErrors after continuing to iterate since the length of the bucket has now changed
                        break

                break
            elif map_key in links:
                link = links[map_key]
                next_node = Node.deserialize(await self.store.load(link))
                node_stack.append((link, next_node))
                continue

            else:
                # This key is not even in the HAMT so just exit
                break

        # Finally, reserialize and fix all links, deleting empty nodes as needed
        if created_change:
            await self._reserialize_and_link(node_stack)
            self.key_count -= 1
            self.root_node_id = node_stack[0][0]

        if not self.read_only:
            self.lock.release()

    async def get(self, key: str) -> bytes | None:
        if not self.read_only:
            self.lock.acquire(blocking=True)

        raw_hash = self.hash_fn(bytes(key, "utf-8"))  # type: ignore

        node_id_stack: list[bytes] = []
        node_id_stack.append(self.root_node_id)

        result: bytes | None = None
        while True:
            top_id = node_id_stack[-1]
            top_node = Node.deserialize(await self.store.load(top_id))
            map_key = str(extract_bits(raw_hash, len(node_id_stack), 8))

            # Check if this node is in one of the buckets
            buckets: dict[str, list[dict[str, bytes]]] = top_node.data["b"]  # type: ignore
            if map_key in buckets:
                bucket = buckets[map_key]
                for kv in bucket:
                    if key in kv:
                        result = kv[key]
                        break

            # If it isn't in one of the buckets, check if there's a link to another serialized node id
            links: dict[str, bytes] = top_node.data["c"]  # type: ignore
            if map_key in links:
                link_id = links[map_key]
                node_id_stack.append(link_id)
                continue
            # Nowhere left to go, stop walking down the tree
            else:
                break

        if not self.read_only:
            self.lock.release()

        return result

    async def has(self, key: str) -> bool:
        result = await self.get(key)

        return result is not None

    def size(self) -> int:
        """Total number of keys"""
        if not self.read_only:
            self.lock.acquire(blocking=True)

        key_count = self.key_count

        if not self.read_only:
            self.lock.release()

        return key_count

    async def keys(self):
        """Async generator of all keys"""
        if not self.read_only:
            self.lock.acquire(blocking=True)

        node_id_stack = []
        node_id_stack.append(self.root_node_id)

        while True:
            if len(node_id_stack) == 0:
                break

            id_top = node_id_stack.pop()
            top = await self.store.load(id_top)
            node = Node.deserialize(top)

            # Collect all keys from all buckets
            buckets: dict[str, list[dict[str, bytes]]] = node.data["b"]  # type: ignore
            for bucket in buckets.values():
                for kv in bucket:
                    for k in kv:
                        yield k

            # Traverse down list of CIDs
            cids: dict[str, bytes] = node.data["c"]  # type: ignore
            for cid in cids.values():
                node_id_stack.append(cid)

        if not self.read_only:
            self.lock.release()

    async def ids(self):
        """Async generator of all IDs as the store uses"""
        if not self.read_only:
            self.lock.acquire(blocking=True)

        node_id_stack = []
        node_id_stack.append(self.root_node_id)

        while len(node_id_stack) > 0:
            top_id = node_id_stack.pop()
            yield top_id
            top_node = Node.deserialize(await self.store.load(top_id))

            # Traverse down list of ids that are the store's links'
            links: dict[str, bytes] = top_node.data["c"]  # type: ignore
            for link in links.values():
                node_id_stack.append(link)

        if not self.read_only:
            self.lock.release()


b3 = multihash.get("blake3")


def blake3_hashfn(self: Hamt, input_bytes: bytes) -> bytes:
    # 32 bytes is the recommended byte size for blake3 and the default, but multihash forces us to explicitly specify
    digest = b3.digest(input_bytes, size=32)
    raw_bytes = b3.unwrap(digest)
    return raw_bytes
