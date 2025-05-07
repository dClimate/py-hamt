from abc import ABC, abstractmethod
from sys import getsizeof
from typing import Callable, Iterator, AsyncIterator, Literal
import uuid
from random import shuffle
import asyncio

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


# Used for readability, since a HAMT also stores IPLDKind objects
# Store IDs are also IPLDKind, which makes the code harder to read if only using IPLDKind rather than this type alias
type Link = IPLDKind
type StoreID = IPLDKind


class Node:
    def __init__(self):
        self.data: list[IPLDKind] = [dict() for _ in range(0, 256)]
        # empty dicts represent empty buckets, lists with one element are links, with the internal element being a link

    def iter_bucket_indices(self) -> Iterator[int]:
        """Return the list indices where there are buckets, empty or not."""
        for i in range(len(self.data)):
            if isinstance(self.data[i], dict):
                yield i

    def iter_link_indices(self) -> Iterator[int]:
        """Return the list indices where there are links."""
        for i in range(len(self.data)):
            if isinstance(self.data[i], list):
                yield i

    def get_link(self, index: int) -> IPLDKind:
        link_wrapper: list[IPLDKind] = self.data[index]  # type: ignore
        return link_wrapper[0]

    def iter_links(self) -> Iterator[IPLDKind]:
        for i in self.iter_link_indices():
            yield self.get_link(i)

    def set_link(self, index: int, link: IPLDKind):
        wrapped = [link]
        self.data[index] = wrapped

    def replace_link(self, old_link: IPLDKind, new_link: IPLDKind):
        for link_index in self.iter_link_indices():
            link = self.get_link(link_index)
            if link == old_link:
                self.set_link(link_index, new_link)
                return

    def serialize(self) -> bytes:
        return dag_cbor.encode(self.data)

    @classmethod
    def deserialize(cls, data: bytes) -> "Node":
        try:
            decoded_data = dag_cbor.decode(data)
        except:  # noqa: E722
            raise Exception(
                "Invalid dag-cbor encoded data from the store was attempted to be decoded"
            )
        if isinstance(decoded_data, list) and len(decoded_data) == 256:
            node = cls()
            node.data = decoded_data
            return node
        else:
            raise ValueError("Data does not contain a valid Node")


# For HAMT internal use for increaing performance with nodes
class NodeStore(ABC):
    @abstractmethod
    async def save(self, original_id: IPLDKind, node: Node) -> IPLDKind:
        pass

    @abstractmethod
    async def load(self, id: IPLDKind) -> Node:
        pass


class ReadCacheStore(NodeStore):
    def __init__(self, hamt: "HAMT"):
        self.hamt = hamt
        self.cache: dict[IPLDKind, Node] = dict()

    def cache_eviction_lru(self):
        while getsizeof(self.cache) > self.hamt.read_cache_limit:
            if len(self.cache) == 0:
                return
            stalest_node_id = next(iter(self.cache.keys()))
            del self.cache[stalest_node_id]

    async def save(self, original_id: IPLDKind, node: Node) -> IPLDKind:
        raise Exception(
            "Save called in virtual store ReadCacheStore, only supposed to be used for reading"
        )

    async def load(self, id: IPLDKind) -> Node:
        # Cache Hit
        if id in self.cache:
            # Reinsert to put this key as the most recently used and last in the dict insertion order
            node = self.cache[id]
            del self.cache[id]
            self.cache[id] = node

            return node

        # Cache Miss
        node = Node.deserialize(await self.hamt.store.load(id))
        self.cache[id] = node
        self.cache_eviction_lru()
        return node


class InMemoryTreeStore(NodeStore):
    # Add note about how this initiates by adding the HAMT's root node to the in memory tree buffer
    # TODO add notice about how flushing needs to be handled by the HAMT internally and called after every operation is finished
    def __init__(self, hamt: "HAMT"):
        self.hamt = hamt
        # This buffer goes from the node's virtual store uuid to a tuple of (original store id, Node), we can return the original node
        self.buffer: dict[int, Node] = dict()

    def is_buffer_id(self, id: IPLDKind) -> bool:
        return isinstance(id, int) and id in self.buffer

    def children_in_memory(self, node: Node) -> Iterator[int]:
        for link in node.iter_links():
            if self.is_buffer_id(link):
                yield link  # type: ignore we know for sure this is an int if it's a buffer id

    def has_children_in_memory(self, node: Node) -> bool:
        for _ in self.children_in_memory(node):
            return True
        return False

    def needs_flushing(self) -> bool:
        return getsizeof(self.buffer) > self.hamt.inmemory_tree_limit

    # TODO, reimplement without treating the root node specially, but do assume that the root node is always in the in memory buffer since it will always be called first in any operations doing tree traversal
    async def flush_buffer(self, flush_everything: bool = False):
        if (not self.needs_flushing()) and (not flush_everything):
            return

        # Start with the root node's children, which should always be in the in memory buffer if there's anything there
        # there is an implicit assumption that the entire continuous line of branches up to an ancestor in the memory buffer, which will happen because of the DFS style traversal in all HAMT operations
        # node stack is a list of tuples that look like (parent_id, self_id, node)
        node_stack: list[tuple[int, int, Node]] = []
        root_node: Node = self.buffer[self.hamt.root_node_id]  # type: ignore
        for child_buffer_id in self.children_in_memory(root_node):
            child_node = self.buffer[child_buffer_id]
            node_stack.append((self.hamt.root_node_id, child_buffer_id, child_node))  # type: ignore

        shuffle(node_stack)  # avoid always recursing down the rightmost branch

        while len(node_stack) > 0 and (self.needs_flushing() or flush_everything):
            parent_buffer_id, top_buffer_id, top_node = node_stack[-1]
            new_nodes_on_stack = []
            if self.has_children_in_memory(top_node):
                for child_buffer_id in self.children_in_memory(top_node):
                    child_node = self.buffer[child_buffer_id]
                    new_nodes_on_stack.append(
                        (top_buffer_id, child_buffer_id, child_node)
                    )

                shuffle(
                    new_nodes_on_stack
                )  # avoid always recursing down the rightmost branch
                node_stack.extend(new_nodes_on_stack)
            else:
                # We can flush this one completely since it has no children in memory, we'll just have to edit its parent after serializing it
                old_id = top_buffer_id
                new_id = await self.hamt.store.save(
                    top_node.serialize(), codec="dag-cbor"
                )
                parent_node = self.buffer[parent_buffer_id]
                parent_node.replace_link(old_id, new_id)

                del self.buffer[top_buffer_id]
                node_stack.pop()

        # Only flush the root node if our size limits are still too big, or we've been instructed to remove everything. Always keeping the root node is important for performance
        if self.needs_flushing() or flush_everything:
            del self.buffer[self.root_node_id]  # type: ignore
            self.root_node_id = await self.hamt.store.save(
                root_node.serialize(), codec="dag-cbor"
            )

        if flush_everything:
            assert len(self.buffer) == 0
        if self.needs_flushing():
            raise Exception(
                "Could not flush in memory tree buffer enough to reach limit, consider raising limit to accomodate for python empty object overheads"
            )

    async def add_to_buffer(self, node: Node) -> IPLDKind:
        # This is IPLDKind type, but not dag_cbor serializable due to int limits so this will throw errors early if the tree is written with the buffer_ids still in there
        buffer_id = uuid.uuid4().int
        # Returns the virtual store ID that was generated for this node
        self.buffer[buffer_id] = node

        return buffer_id

    async def save(self, original_id: IPLDKind, node: Node) -> IPLDKind:
        if self.is_buffer_id(original_id):
            return original_id

        # This node was not in the buffer, don't save it to the backing store but rather add to it to the in memory buffer
        buffer_id = await self.add_to_buffer(node)
        return buffer_id

    async def load(self, id: IPLDKind) -> Node:
        # Hit for something already in the in memory tree
        if self.is_buffer_id(id):
            node: Node = self.buffer[id]  # type: ignore
            return node

        # Miss, something that isn't in the in memory tree, add it
        node = Node.deserialize(await self.hamt.store.load(id))
        await self.add_to_buffer(node)
        return node


class HAMT:
    """
    This HAMT presents a key value interface, like a python dictionary. The only limits are that keys can only be strings, and values can only be types amenable with [IPLDKind](https://dag-cbor.readthedocs.io/en/stable/api/dag_cbor.ipld.html#dag_cbor.ipld.IPLDKind). IPLDKind is a fairly flexible data model, but do note that integers are must be within the bounds of a signed 64-bit integer.

    py-hamt uses blake3 with a 32 byte wide hash by default, but to bring your own, read more in the documentation of `blake3_hashfn`.

    # Some notes about thread safety
    TODO write about it not being thread safe but async concurrency model safe, so dont use it in multiple threads but do use it within an async event loop

    Since modifying a HAMT changes all parts of the tree, due to reserializing and saving to the backing store, modificiations are not thread safe. Thus, we offer a read-only mode which allows for parallel accesses, or a thread safe write enabled mode. You can see what type it is from the `read_only` variable. HAMTs default to write enabled mode on creation. Calling mutating operations in read only mode will raise Exceptions.

    In write enabled mode, all operations block and wait, so multiple threads can write to the HAMT but the operations will in reality happen one after the other.

    # dunder method documentation
    These are not generated by pdoc automatically so we are including their documentation here.

    ## `__len__`
    Total number of keys. Note that this will have to scan the entire tree in order to count, which can take a while depending on the speed of the store retrieval.

    ## `__iter__`
    Generator of all string keys. When initially called, this will freeze the root node and then start iteration, so subsequent sets and deletes that mutate will not be reflected in the keys returned in this iteration.

    ## `__deepcopy__`
    For use with the python copy module. This creates deep copies of the current HAMT. The only thing it does not copy over is the cache.
    ```python
    from copy import deepcopy
    hamt = # ...
    copy_hamt = deepcopy(hamt)
    ```
    """

    store: Store
    """@private"""

    # Every call of this will look like self.hash_fn() and so the first argument will always take a self argument
    hash_fn: Callable[[bytes], bytes]
    """@private"""

    # Only important for writing, when reading this will use buckets even if they are overly big
    max_bucket_size: int
    """@private"""

    root_node_id: IPLDKind
    """DO NOT modify this directly.

    TODO write about how you do need to read it at times, and it should only be read when the buffer has been flushed to the backing store

    This is the ID that the store returns for the root node of the HAMT. This is exposed since it is sometimes useful to reconstruct other HAMTs later if using a persistent store.

    This is really type IPLDKind, but the documentation generates this strange type instead since IPLDKind is a type union.
    """

    lock: asyncio.Lock
    """
    @private
    For use in multithreading
    """

    def __init__(
        self,
        store: Store,
        hash_fn: Callable[[bytes], bytes] = blake3_hashfn,
        root_node_id: IPLDKind = None,
        read_only: bool = False,
        max_bucket_size: int = 67,
        inmemory_tree_limit: int = 10_000_000,  # 10 MB default
        read_cache_limit: int = 10_000_000,  # 10 MB default
    ):
        self.store = store
        self.hash_fn = hash_fn
        self.lock = asyncio.Lock()
        self.read_only = read_only

        if max_bucket_size < 1:
            raise ValueError("Bucket size maximum must be a positive integer")
        self.max_bucket_size = max_bucket_size

        # TODO add documentation about buffer limit and cache limit minimums being 10 KB, to avoid strange boundary cases in the algorithms due to python object overhead meaning that an in memory buffer should just never be used
        # Add documentation about how its values should not be positive, otherwise the HAMT will not work
        # Specify a minimum size of a couple hundred bytes, since just an empty dict will consume some memory

        if read_cache_limit < 10_000:
            raise ValueError("Cache limit too small, less than 10 KB")
        if inmemory_tree_limit < 10_000:
            raise ValueError("In memory tree limit too small, less than 10 KB")

        self.inmemory_tree_limit = inmemory_tree_limit
        self.read_cache_limit = read_cache_limit

        self.root_node_id = root_node_id

        self.node_store: NodeStore
        if read_only:
            self.node_store = ReadCacheStore(self)
        else:
            self.node_store = InMemoryTreeStore(self)

    @classmethod
    async def build(cls, *args, **kwargs) -> "HAMT":
        hamt = cls(*args, **kwargs)
        if hamt.root_node_id is None:
            hamt.root_node_id = await hamt.node_store.save(None, Node())
        return hamt

    # This is typically a massive blocking operation, you dont want to be running this concurrently with a bunch of other operations, so it's ok to have it not be async
    async def make_read_only(self):
        async with self.lock:
            self.read_only = True

            inmemory_tree: InMemoryTreeStore = self.node_store  # type: ignore
            await inmemory_tree.flush_buffer(flush_everything=True)

            self.node_store = ReadCacheStore(self)

    async def enable_write(self):
        async with self.lock:
            self.read_only = False
            self.node_store = InMemoryTreeStore(self)

    async def _reserialize_and_link(self, node_stack: list[tuple[Link, Node]]):
        """
        This function starts from the node at the end of the list and reserializes so that each node holds valid new IDs after insertion into the store
        Takes a stack of nodes, we represent a stack with a list where the first element is the root element and the last element is the top of the stack
        Each element in the list is a tuple where the first element is the ID from the store and the second element is the Node in python
        If a node ends up being empty, then it is deleted entirely, unless it is the root node
        Modifies in place
        """
        # iterate in the reverse direction, this range goes from n-1 to 0, from the bottommost tree node to the root
        for stack_index in range(len(node_stack) - 1, -1, -1):
            old_id, node = node_stack[stack_index]

            # If this node is empty, and it's not the root node, then we can delete it entirely from the list
            is_empty = True
            for i in range(len(node.data)):
                item = node.data[i]
                # If we have a nonempty bucket, or a link at all, then this is a nonempty node
                if (isinstance(item, dict) and len(item) > 0) or isinstance(item, list):
                    is_empty = False
                    break

            is_root = stack_index == 0

            if is_empty and not is_root:
                # Unlink from the rest of the tree
                _, prev_node = node_stack[stack_index - 1]
                # When removing links, don't worry about two nodes having the same link since all nodes are guaranteed to be different by the removal of empty nodes after every single operation
                for link_index in prev_node.iter_link_indices():
                    link = prev_node.get_link(link_index)
                    if link == old_id:
                        # Delete the link by making it an empty bucket
                        prev_node.data[link_index] = dict()
                        break

                # Remove from our stack, continue reserializing up the tree
                node_stack.pop(stack_index)
                continue

            # If not an empty node, just reserialize like normal and replace this one
            new_store_id = await self.node_store.save(old_id, node)
            node_stack[stack_index] = (new_store_id, node)

            # If this is not the last i.e. root node, we need to change the linking of the node prior in the list since we just reserialized
            if not is_root:
                _, prev_node = node_stack[stack_index - 1]
                prev_node.replace_link(old_id, new_store_id)

    async def set(self, key: str, val: IPLDKind):
        val_ptr = await self.store.save(dag_cbor.encode(val), codec="raw")
        await self._set_pointer(key, val_ptr)

    async def _set_pointer(self, key: str, val_ptr: IPLDKind):
        if self.read_only:
            raise Exception("Cannot call set on a read only HAMT")

        async with self.lock:
            node_stack: list[tuple[IPLDKind, Node]] = []
            root_node = await self.node_store.load(self.root_node_id)
            node_stack.append((self.root_node_id, root_node))

            # FIFO queue to keep track of all the KVs we need to insert
            # This is needed if any buckets overflow and so we need to reinsert all those KVs
            kvs_queue: list[tuple[str, IPLDKind]] = []
            kvs_queue.append((key, val_ptr))

            while len(kvs_queue) > 0:
                _, top_node = node_stack[-1]
                curr_key, curr_val_ptr = kvs_queue[0]

                raw_hash = self.hash_fn(curr_key.encode())
                map_key = extract_bits(raw_hash, len(node_stack) - 1, 8)

                item = top_node.data[map_key]
                if isinstance(item, list):
                    next_node_id = item[0]
                    next_node = await self.node_store.load(next_node_id)
                    node_stack.append((next_node_id, next_node))
                elif isinstance(item, dict):
                    bucket: dict[str, IPLDKind] = item

                    # If this bucket already has this same key, or has space, just rewrite the value and then go work on the others in the queue
                    if curr_key in bucket or len(bucket) < self.max_bucket_size:
                        bucket[curr_key] = curr_val_ptr
                        kvs_queue.pop(0)
                        continue

                    # The current key is not in the bucket and the bucket is too full, so empty KVs from the bucket and restart insertion
                    for k, v_ptr in bucket:
                        kvs_queue.append((k, v_ptr))

                    # Create a new link to a new node so that we can reflow these KVs into a new subtree
                    new_node = Node()
                    new_node_id = await self.node_store.save(None, new_node)
                    link = [new_node_id]
                    top_node.data[map_key] = link

            # Finally, reserialize and fix all links, deleting empty nodes as needed
            await self._reserialize_and_link(node_stack)
            self.root_node_id = node_stack[0][0]
            # This needs to be called after the root node is set from the node stack, since this may change the root node if the memory buffer is too small to hold the root node
            if isinstance(self.node_store, InMemoryTreeStore):
                await self.node_store.flush_buffer()

    async def delete(self, key: str):
        # Also deletes the pointer at the same time so this doesn't have a _delete_pointer duo
        if self.read_only:
            raise Exception("Cannot call delete on a read only HAMT")

        async with self.lock:
            raw_hash = self.hash_fn(key.encode())

            node_stack: list[tuple[IPLDKind, Node]] = []
            root_node = await self.node_store.load(self.root_node_id)
            node_stack.append((self.root_node_id, root_node))

            created_change = False
            while True:
                _, top_node = node_stack[-1]
                map_key = extract_bits(raw_hash, len(node_stack) - 1, 8)

                item = top_node.data[map_key]
                if isinstance(item, dict):
                    bucket = item
                    if key in bucket:
                        del bucket[key]
                        created_change = True
                    # Break out since whether or not the key is in the bucket, it should have been here so either now reserialize or raise a KeyError
                    break
                elif isinstance(item, list):
                    link = item[0]
                    next_node = await self.node_store.load(link)
                    node_stack.append((link, next_node))

            # Finally, reserialize and fix all links, deleting empty nodes as needed
            if created_change:
                await self._reserialize_and_link(node_stack)
                self.root_node_id = node_stack[0][0]
                # This needs to be called after the root node is set from the node stack, since this may change the root node if the memory buffer is too small to hold the root node
                if isinstance(self.node_store, InMemoryTreeStore):
                    await self.node_store.flush_buffer()
            else:
                # If we didn't make a change, then this key must not exist within the HAMT
                raise KeyError

    async def get(self, key: str) -> IPLDKind:
        pointer: IPLDKind = await self._get_pointer(key)
        return dag_cbor.decode(await self.store.load(pointer))

    async def _get_pointer(self, key: str) -> IPLDKind:
        async with self.lock:
            raw_hash = self.hash_fn(key.encode())

            current_id = self.root_node_id
            current_depth = 0

            # Don't check if result is none but use a boolean to indicate finding something, this is because None is a possible value of IPLDKind
            result_ptr: IPLDKind = None
            found_a_result: bool = False
            while True:
                top_id = current_id
                top_node = await self.node_store.load(top_id)
                map_key = extract_bits(raw_hash, current_depth, 8)

                # Check if this key is in one of the buckets
                item = top_node.data[map_key]
                if isinstance(item, dict):
                    bucket = item
                    if key in bucket:
                        result_ptr = bucket[key]
                        found_a_result = True
                        break

                if isinstance(item, list):
                    link = item[0]
                    current_id = link
                    current_depth += 1
                    continue

                # Nowhere left to go, stop walking down the tree
                break

            if isinstance(self.node_store, InMemoryTreeStore):
                await self.node_store.flush_buffer()

            if not found_a_result:
                raise KeyError

            return result_ptr

    async def _iter_nodes(self) -> AsyncIterator[tuple[IPLDKind, Node]]:
        async with self.lock:
            node_id_stack = [self.root_node_id]
            while len(node_id_stack) > 0:
                top_id = node_id_stack.pop()
                node = await self.node_store.load(top_id)
                yield (top_id, node)
                node_id_stack.extend(list(node.iter_links()))

    # TODO documentation warning that iterating does not lock the HAMT so be careful, but this choice was made to allow for getting keys in the middle of iteration, otherwise you'd have to wait for the whole thing to finish
    async def keys(self) -> AsyncIterator[str]:
        async for _, node in self._iter_nodes():
            for bucket_index in node.iter_bucket_indices():
                bucket: dict[str, IPLDKind] = node.data[bucket_index]  # type: ignore
                for key in bucket:
                    yield key

    # TODO similar documentation warning that iterating does not lock the HAMT so don't modify the HAMT in the middle of this thing running or you will get incorrect results
    async def len(self) -> int:
        """Return the number of key value mappings in this HAMT."""
        count = 0
        async for _ in self.keys():
            count += 1

        return count
