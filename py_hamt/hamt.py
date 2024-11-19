from threading import Lock
from typing import Callable
from collections.abc import MutableMapping

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


# Used for readability, since a HAMT also stores IPLDKind objects
# Store IDs are also IPLDKind, which makes the code harder to read if only using IPLDKind rather than this type alias
type Link = IPLDKind


class Node:
    # Use a dict for easy serializability with dag_cbor
    data: dict[str, IPLDKind]

    def __init__(self):
        data = {}

        # Each "string" key for both buckets and CIDs is the bits as a string, e.g. str(0b1101) = '13'
        buckets: dict[str, list[dict[str, IPLDKind]]] = {}
        data["B"] = buckets
        links: dict[str, Link] = {}
        data["L"] = links
        self.data = data

    # By having these two methods, the HAMT class has to know less about Node's internal structure and we get better type checking since we don't need to put #type:ignore everywhere
    def get_buckets(self) -> dict[str, list[dict[str, IPLDKind]]]:
        buckets: dict[str, list[dict[str, IPLDKind]]] = self.data["B"]  # type: ignore
        return buckets

    def get_links(self) -> dict[str, Link]:
        links: dict[str, Link] = self.data["L"]  # type: ignore
        return links

    def _replace_link(self, old_link: Link, new_link: Link):
        links = self.get_links()
        for str_key in list(links.keys()):
            link = links[str_key]
            if link == old_link:
                links[str_key] = new_link

    def _remove_link(self, old_link: Link):
        links = self.get_links()
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
            and "B" in decoded
            and isinstance(decoded["B"], dict)
            and "L" in decoded
            and isinstance(decoded["L"], dict)
        ):
            node = cls()
            node.data = decoded
            return node
        else:
            raise ValueError("Data not a valid Node serialization")


b3 = multihash.get("blake3")


def blake3_hashfn(input_bytes: bytes) -> bytes:
    """
    This is provided as a default recommended hash function by py-hamt. It uses the blake3 hash function and uses 32 bytes as the hash size.

    To bring your own hash function, just create a function that takes in bytes and returns the hash bytes, and use that in the HAMT init method.

    It's important to note that the resulting hash must must always be a multiple of 8 bits since python bytes object can only represent in segments of bytes, and thus 8 bits.
    """
    # 32 bytes is the recommended byte size for blake3 and the default, but multihash forces us to explicitly specify
    digest = b3.digest(input_bytes, size=32)
    raw_bytes = b3.unwrap(digest)
    return raw_bytes


class HAMT(MutableMapping):
    """
    This HAMT presents key value interface like a normal python dictionary, except keys can only be strings and values can only be bytes. See some sample usage code here:
    ```python
    from py_hamt import HAMT, DictStore

    in_memory_store = DictStore()
    hamt = HAMT(store=in_memory_store)
    hamt["foo"] = "bar"
    assert b"bar" == hamt["foo"]
    assert len(hamt) == 1
    hamt["foo"] = b"bar1"
    hamt["foo2"] = b"bar2"
    assert len(hamt) == 2
    for key in hamt:
        print(key)
    print (list(hamt)) # [foo, foo2], order depending on hash function
    del hamt["foo"]
    assert len(hamt) == 1
    ```
    py-hamt uses blake3 with a 32 byte wide hash by default, but to bring your own, read more in the documentation of `blake3_hashfn`

    A HAMT is mutable, so you can just keep calling operations on that class instance.

    Some notes about thread safety. Since modifying a HAMT changes all parts of the tree, due to reserializing and saving to the backing store, modificiations are not thread safe. Thus, we offer a read-only mode, or write enabled mode. You can see what type it is from the `read_only` variable. HAMTs default to write enabled mode on creation. Calling mutating operations in read only mode will raise Exceptions.

    In write enabled mode, all operations block and wait, so multiple threads can write to the HAMT but the operations will in reality happen one after the other. Read-only mode allows for no modifications but it has the advantages that reads can now be parallelized, as reads don't modify the HAMT. Normally, reads are done only after a write is finished so that it is not reading an invalid tree.
    """

    store: Store
    """@private"""
    # Every call of this will look like self.hash_fn() and so the first argument will always take a self argument
    hash_fn: Callable[[bytes], bytes]
    """@private"""
    # Only important for writing, when reading this will use buckets even if they are overly big
    max_bucket_size: int
    """@private"""

    # Don't use the type alias here since this is exposed in the documentation
    root_node_id: IPLDKind
    """DO NOT modify this directly.

    This is the ID that the store returns for the root node of the HAMT. This is exposed since it is sometimes useful to read this for storing to construct other HAMTs later if using a persistent store.
    """
    key_count: int
    """@private"""

    lock: Lock
    """
    @private
    For use in multithreading
    """

    read_only: bool
    """
    DO NOT modify this directly. This is here for you to read and check.

    You can modify the read status of a HAMT through the `make_read_only` or `enable_write` functions, so that the HAMT will block on making a change until all mutating operations are done.
    """

    def __init__(
        self,
        store: Store,
        hash_fn: Callable[[bytes], bytes] = blake3_hashfn,
        max_bucket_size: int = 4,
        read_only: bool = False,
        root_node_id: IPLDKind = None,
    ):
        self.store = store
        self.hash_fn = hash_fn

        if root_node_id is None:
            root_node = Node()
            self.root_node_id = store.save(root_node.serialize())
        else:
            self.root_node_id = root_node_id

        self.key_count = 0

        self.max_bucket_size = max_bucket_size
        self.read_only = read_only
        self.lock = Lock()

    # dunder for the python deepcopy module
    def __deepcopy__(self, memo) -> "HAMT":
        if not self.read_only:
            self.lock.acquire(blocking=True)

        copy_hamt = HAMT(
            store=self.store,
            hash_fn=self.hash_fn,
            max_bucket_size=self.max_bucket_size,
            read_only=self.read_only,
            root_node_id=self.root_node_id,
        )
        copy_hamt.key_count = self.key_count

        if not self.read_only:
            self.lock.release()

        return copy_hamt

    def make_read_only(self):
        """Disables all mutation of this HAMT. When enabled, the HAMT will throw errors if set or delete are called. When a HAMT is only in read only mode, it allows for safe multithreaded reads, increasing performance."""
        self.lock.acquire(blocking=True)
        self.read_only = True
        self.lock.release()

    def enable_write(self):
        self.lock.acquire(blocking=True)
        self.read_only = False
        self.lock.release()

    def _reserialize_and_link(self, node_stack: list[tuple[Link, Node]]):
        """
        For internal use
        This function starts from the node at the end of the list and reserializes so that each node holds valid new IDs after insertion into the store
        Takes a stack of nodes, we represent a stack with a list where the first element is the root element and the last element is the top of the stack
        Each element in the list is a tuple where the first element is the ID from the store and the second element is the Node in python
        If a node ends up being empty, then it is deleted entirely, unless it is the root node
        Modifies in place
        """
        # Iterate in the reverse direction, imitating going deeper into a stack
        for stack_index in range(len(node_stack) - 1, -1, -1):
            old_store_id, node = node_stack[stack_index]

            # If this node is empty, and its not the root node, then we can delete it entirely from the list
            buckets = node.get_buckets()
            links = node.get_links()
            is_empty = len(buckets) == 0 and len(links) == 0
            is_not_root = stack_index > 0

            if is_empty and is_not_root:
                # Unlink from the rest of the tree
                _, prev_node = node_stack[stack_index - 1]
                prev_node._remove_link(old_store_id)

                # Remove from our stack
                node_stack.pop(stack_index)
            else:
                # Reserialize
                new_store_id = self.store.save(node.serialize())
                node_stack[stack_index] = (new_store_id, node)

                # If this is not the last i.e. root node, we need to change the linking of the node prior in the list
                if stack_index > 0:
                    _, prev_node = node_stack[stack_index - 1]
                    prev_node._replace_link(old_store_id, new_store_id)

    def __setitem__(self, key_to_insert: str, val_to_insert: IPLDKind):
        if self.read_only:
            raise Exception("Cannot call set on a read only HAMT")

        if not self.read_only:
            self.lock.acquire(blocking=True)

        node_stack: list[tuple[Link, Node]] = []
        root_node = Node.deserialize(self.store.load(self.root_node_id))
        node_stack.append((self.root_node_id, root_node))

        # FIFO queue to keep track of all the KVs we need to insert
        # This is important for if any buckets overflow and thus we need to reinsert all those KVs
        kvs_queue: list[tuple[str, IPLDKind]] = []
        kvs_queue.append((key_to_insert, val_to_insert))

        created_change = False
        # Keep iterating until we have no more KVs to insert
        while len(kvs_queue) != 0:
            _, top_node = node_stack[-1]
            curr_key, curr_val = kvs_queue[0]

            raw_hash = self.hash_fn(curr_key.encode())
            map_key = str(extract_bits(raw_hash, len(node_stack), 8))

            buckets = top_node.get_buckets()
            links = top_node.get_links()

            if map_key in links and map_key in buckets:
                self.lock.release()
                raise Exception(
                    "Key in both buckets and links of the node, invariant violated"
                )
            elif map_key in links:
                next_node_id = links[map_key]
                next_node = Node.deserialize(self.store.load(next_node_id))
                node_stack.append((next_node_id, next_node))
            elif map_key in buckets:
                bucket = buckets[map_key]
                created_change = True

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

                bucket_has_space = len(bucket) < self.max_bucket_size
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
                    new_node_id = self.store.save(new_node.serialize())

                    links[map_key] = new_node_id

                    # We need to rerun from the top with the new queue, but this time this node will have a link to put KVs deeper down in the tree
                    node_stack.append((new_node_id, new_node))
            else:
                # If there is no link and no bucket, then we can create a new bucket, insert, and be done with this key
                bucket: list[dict[str, IPLDKind]] = []
                bucket.append({curr_key: curr_val})
                kvs_queue.pop(0)
                buckets[map_key] = bucket
                self.key_count += 1
                created_change = True

        # Finally, reserialize and fix all links, deleting empty nodes as needed
        if created_change:
            self._reserialize_and_link(node_stack)
            node_stack_top_id = node_stack[0][0]
            self.root_node_id = node_stack_top_id

        if not self.read_only:
            self.lock.release()

    def __delitem__(self, key: str):
        if self.read_only:
            raise Exception("Cannot call delete on a read only HAMT")

        if not self.read_only:
            self.lock.acquire(blocking=True)

        raw_hash = self.hash_fn(key.encode())

        node_stack: list[tuple[Link, Node]] = []
        root_node = Node.deserialize(self.store.load(self.root_node_id))
        node_stack.append((self.root_node_id, root_node))

        created_change = False
        while True:
            top_id, top_node = node_stack[-1]
            map_key = str(extract_bits(raw_hash, len(node_stack), 8))

            buckets = top_node.get_buckets()
            links = top_node.get_links()

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
                next_node = Node.deserialize(self.store.load(link))
                node_stack.append((link, next_node))
                continue

            else:
                # This key is not even in the HAMT so just exit
                break

        # Finally, reserialize and fix all links, deleting empty nodes as needed
        if created_change:
            self._reserialize_and_link(node_stack)
            self.key_count -= 1
            node_stack_top_id = node_stack[0][0]
            self.root_node_id = node_stack_top_id

        if not self.read_only:
            self.lock.release()

        if not created_change:
            raise KeyError

    def __getitem__(self, key: str) -> IPLDKind:
        if not self.read_only:
            self.lock.acquire(blocking=True)

        raw_hash = self.hash_fn(key.encode())

        node_id_stack: list[Link] = []
        node_id_stack.append(self.root_node_id)

        # Don't check if result is none but use a boolean to indicate finding something, this is because None is a possible value the HAMT can store
        result: IPLDKind = None
        found_a_result: bool = False
        while True:
            top_id = node_id_stack[-1]
            top_node = Node.deserialize(self.store.load(top_id))
            map_key = str(extract_bits(raw_hash, len(node_id_stack), 8))

            # Check if this node is in one of the buckets
            buckets = top_node.get_buckets()
            if map_key in buckets:
                bucket = buckets[map_key]
                for kv in bucket:
                    if key in kv:
                        result = kv[key]
                        found_a_result = True
                        break

            # If it isn't in one of the buckets, check if there's a link to another serialized node id
            links = top_node.get_links()
            if map_key in links:
                link_id = links[map_key]
                node_id_stack.append(link_id)
                continue
            # Nowhere left to go, stop walking down the tree
            else:
                break

        if not self.read_only:
            self.lock.release()

        if not found_a_result:
            raise KeyError

        return result

    def __len__(self) -> int:
        """Total number of keys"""
        if not self.read_only:
            self.lock.acquire(blocking=True)

        key_count = self.key_count

        if not self.read_only:
            self.lock.release()

        return key_count

    def __iter__(self):
        """Generator of all string keys. When initially called, this will freeze the root node and then start iteration, so subsequent sets and deletes that mutate will not be reflected in the keys returned in this iteration."""
        if not self.read_only:
            self.lock.acquire(blocking=True)

        node_id_stack = []
        node_id_stack.append(self.root_node_id)

        if not self.read_only:
            self.lock.release()

        while True:
            if len(node_id_stack) == 0:
                break

            id_top = node_id_stack.pop()
            top = self.store.load(id_top)
            node = Node.deserialize(top)

            # Collect all keys from all buckets
            buckets = node.get_buckets()
            for bucket in buckets.values():
                for kv in bucket:
                    for k in kv:
                        yield k

            # Traverse down list of links
            links = node.get_links()
            for link in links.values():
                node_id_stack.append(link)

    def ids(self):
        """Generator of all IDs the backing store uses"""
        if not self.read_only:
            self.lock.acquire(blocking=True)

        node_id_stack: list[Link] = []
        node_id_stack.append(self.root_node_id)

        if not self.read_only:
            self.lock.release()

        while len(node_id_stack) > 0:
            top_id = node_id_stack.pop()
            yield top_id
            top_node = Node.deserialize(self.store.load(top_id))

            # Traverse down list of ids that are the store's links'
            links = top_node.get_links()
            for link in links.values():
                node_id_stack.append(link)
