import asyncio
import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    cast,
)

import dag_cbor
from dag_cbor.ipld import IPLDKind
from multiformats import multihash

from .store_httpx import ContentAddressedStore


def extract_bits(hash_bytes: bytes, depth: int, nbits: int) -> int:
    """
    Extract `nbits` bits from `hash_bytes`, beginning at position `depth * nbits`, and convert them into an unsigned integer value.

    hash_bytes: binary hash to extract bit sequence from
    depth: depth of the node containing the hash
    nbits: bit width of hash

    returns an unsigned integer version of the bit sequence
    """
    # This is needed since int.bit_length on a integer representation of a bytes object ignores leading 0s
    hash_bit_length: int = len(hash_bytes) * 8

    start_bit_index: int = depth * nbits

    if hash_bit_length - start_bit_index < nbits:
        raise IndexError("Arguments extract more bits than remain in the hash bits")

    mask: int = (0b1 << (hash_bit_length - start_bit_index)) - 1
    n_chop_off_at_end: int = int.bit_length(mask) - nbits

    hash_as_int: int = int.from_bytes(hash_bytes)
    result: int = (hash_as_int & mask) >> n_chop_off_at_end

    return result


b3 = multihash.get("blake3")


def blake3_hashfn(input_bytes: bytes) -> bytes:
    """
    This is the default blake3 hash function used for the `HAMT`, with a 32 byte hash size.

    """
    # 32 bytes is the recommended byte size for blake3 and the default, but multihash forces us to explicitly specify
    digest: bytes = b3.digest(input_bytes, size=32)
    raw_bytes: bytes = b3.unwrap(digest)
    return raw_bytes


class Node:
    def __init__(self) -> None:
        self.data: list[IPLDKind] = [{} for _ in range(0, 256)]
        # empty dicts represent empty buckets
        # lists with one element are links, where list[0] is the real link

    def iter_buckets(self) -> Iterator[Dict[str, IPLDKind]]:
        for item in self.data:
            if isinstance(item, dict):
                yield cast(Dict[str, IPLDKind], item)

    def iter_link_indices(self) -> Iterator[int]:
        for i in range(len(self.data)):
            if isinstance(self.data[i], list):
                yield i

    def iter_links(self) -> Iterator[IPLDKind]:
        for item in self.data:
            if isinstance(item, list):
                yield cast(list[IPLDKind], item)[0]

    def get_link(self, index: int) -> IPLDKind:
        link_wrapper: list[IPLDKind] = cast(list[IPLDKind], self.data[index])
        return link_wrapper[0]

    def set_link(self, index: int, link: IPLDKind):
        wrapped: list[IPLDKind] = [link]
        self.data[index] = wrapped

    # This assumes that there is only one unique link matching the old link, which is valid since we prune empty Nodes when reserializing and relinking
    def replace_link(self, old_link: IPLDKind, new_link: IPLDKind) -> None:
        for link_index in self.iter_link_indices():
            link = self.get_link(link_index)
            if link == old_link:
                self.set_link(link_index, new_link)
                return

    def __deepcopy__(self, memo: dict[int, Any]) -> "Node":
        new_node = Node()
        new_node.data = deepcopy(self.data)
        return new_node

    def is_empty(self) -> bool:
        for i in range(len(self.data)):
            item: IPLDKind = self.data[i]
            if (isinstance(item, dict) and len(item) > 0) or isinstance(item, list):
                return False
        return True

    def serialize(self) -> bytes:
        return dag_cbor.encode(self.data)

    @classmethod
    def deserialize(cls, data: bytes) -> "Node":
        try:
            # dag_cbor.decode() returns a union of many IPLD kinds.
            # We know a serialised Node is always a *list* of IPLDKinds,
            # so narrow the type for mypy.
            decoded_data = cast(list[IPLDKind], dag_cbor.decode(data))
        except:  # noqa: E722
            raise Exception(
                "Invalid dag-cbor encoded data from the store was attempted to be decoded"
            )

        node = cls()
        node.data = decoded_data  # type: ignore
        return node


class NodeStore(ABC):
    @abstractmethod
    async def save(self, original_id: IPLDKind, node: Node) -> IPLDKind:
        """"""

    @abstractmethod
    async def load(self, id: IPLDKind) -> Node:
        """"""

    @abstractmethod
    def size(self) -> int:
        """Calculate the size of all internal Nodes in memory. Should not include the overhead in python of the dict or all the keys, which will be negligible compared to the Node sizes."""

    # Async only because the in memory tree has to make async calls when flushing out
    @abstractmethod
    async def vacate(self):
        """Remove everything from the cache."""


class ReadCacheStore(NodeStore):
    def __init__(self, hamt: "HAMT") -> None:
        self.hamt: HAMT = hamt
        self.cache: dict[IPLDKind, Node] = {}

    async def save(self, original_id: IPLDKind, node: Node) -> IPLDKind:
        raise Exception("Node was attempted to be written to the read cache")

    async def load(self, id: IPLDKind) -> Node:
        # Cache Hit
        if id in self.cache:
            node: Node = self.cache[id]
            return node

        # Cache Miss
        node = Node.deserialize(await self.hamt.cas.load(id))
        self.cache[id] = node
        return node

    def size(self) -> int:
        total: int = 0
        for k in self.cache:
            node: Node = self.cache[k]
            total += len(node.serialize())
        return total

    async def vacate(self):
        self.cache = {}


class InMemoryTreeStore(NodeStore):
    def __init__(self, hamt: "HAMT") -> None:
        self.hamt: HAMT = hamt
        # The integer key is a uuidv4 128-bit integer, for (almost perfectly) guaranteeing uniqueness
        self.buffer: dict[int, Node] = {}

    def is_buffer_id(self, id: IPLDKind) -> bool:
        return id in self.buffer

    def children_in_memory(self, node: Node) -> Iterator[int]:
        for link in node.iter_links():
            if self.is_buffer_id(link):
                yield cast(
                    int, link
                )  # we know for sure this is an int if it's a buffer id

    _replaced_id_marker: bytes = bytes(64)

    # Callers must acquire a lock/stop operations to ensure an accurate calculation!
    def size(self) -> int:
        # This is more difficult than normal links are intentionally unencodable in dag-cbor as the UUIDv4 128-bit integer IDs do not fit
        # So before serializing, make a copy of the node, replace all of those integer links with zeroed out 64 byte marker. 64 bytes is a conservative estimate for most IDs returned by CASes
        total: int = 0
        for k in self.buffer:
            node: Node = self.buffer[k]
            copied: Node = deepcopy(node)
            for link_index in copied.iter_link_indices():
                if self.is_buffer_id(copied.get_link(link_index)):
                    copied.set_link(link_index, InMemoryTreeStore._replaced_id_marker)
            total += len(copied.serialize())

        return total

    # The HAMT must properly acquire a lock for this to run successfully! This is not async or thread safe
    # This algorithm has an implicit assumption that the entire continuous line of branches up to an ancestor is in the memory buffer, which will happen because of the DFS style traversal in all HAMT operations
    async def vacate(self) -> None:
        # node stack is a list of tuples that look like (parent_id, self_id, node)
        node_stack: list[tuple[int | None, int, Node]] = []
        # The root node may not be in the buffer, e.g. this is HAMT initialized with a specific root node id
        if self.is_buffer_id(self.hamt.root_node_id):
            root_node: Node = self.buffer[cast(int, self.hamt.root_node_id)]
            node_stack.append((None, cast(int, self.hamt.root_node_id), root_node))

        while len(node_stack) > 0:
            parent_buffer_id, top_buffer_id, top_node = node_stack[-1]
            new_nodes_on_stack: list[tuple[int, int, Node]] = []
            for child_buffer_id in self.children_in_memory(top_node):
                child_node: Node = self.buffer[child_buffer_id]
                new_nodes_on_stack.append((top_buffer_id, child_buffer_id, child_node))

            no_children_in_memory: bool = len(new_nodes_on_stack) == 0
            # Flush this node out and relink the rest of the tree
            if no_children_in_memory:
                is_root: bool = parent_buffer_id is None
                old_id: int = top_buffer_id
                new_id: IPLDKind = await self.hamt.cas.save(
                    top_node.serialize(), codec="dag-cbor"
                )
                del self.buffer[old_id]
                node_stack.pop()

                # If it's the root, we need to set the hamt's root node id once this is done sending to the backing store
                if is_root:
                    self.hamt.root_node_id = new_id
                # Edit and properly relink the parent if this is not the root
                else:
                    # parent_buffer_id is never None in this branch
                    assert parent_buffer_id is not None
                    parent_node: Node = self.buffer[parent_buffer_id]
                    parent_node.replace_link(old_id, new_id)
            # Continue recursing down the tree
            else:
                node_stack.extend(new_nodes_on_stack)

        # There are only two types of nodes left in the buffer:
        # 1. A bunch of nodes that seem "unlinked" to anything else, since Links used within the Nodes will reference the real underlying CAS
        # 2. A bunch of empty nodes leftover if the key values they contain are deleted, these would normally be consolidated by a content addressed system but will be leftover in the buffer
        # So we can just clear out everything since these nodes are not used by the rest of the tree
        self.buffer = {}

    async def add_to_buffer(self, node: Node) -> IPLDKind:
        # This buffer_id is IPLDKind type since technically it's an int, but it's not dag_cbor serializable since that library can only do up to 64-bit ints. Thus this will throw errors early if a node is written with buffer_ids still in there
        buffer_id: int = uuid.uuid4().int
        self.buffer[buffer_id] = node

        return buffer_id

    async def save(self, original_id: IPLDKind, node: Node) -> IPLDKind:
        if self.is_buffer_id(original_id):
            return original_id

        # This node was not in the buffer, don't save it to the backing store but rather add to it to the in memory buffer
        buffer_id: IPLDKind = await self.add_to_buffer(node)
        return buffer_id

    async def load(self, id: IPLDKind) -> Node:
        # Hit for something already in the in memory tree
        if self.is_buffer_id(id):
            node: Node = self.buffer[cast(int, id)]  # we know all buffer ids are ints
            return node

        # Something that isn't in the in memory tree, add it
        node = Node.deserialize(await self.hamt.cas.load(id))
        await self.add_to_buffer(node)
        return node


class HAMT:
    """
    An implementation of a Hash Array Mapped Trie for an arbitrary Content Addressed Storage (CAS) system, e.g. IPFS. This uses the IPLD data model.

    Use this to store arbitrarily large key-value mappings in your CAS of choice.

    For writing, this HAMT is async safe but NOT thread safe. Only write in an async event loop within the same thread.

    When in read-only mode, the HAMT is both async and thread safe.

    #### A note about memory management, read+write and read-only modes
    The HAMT can be in either read+write mode or read-only mode. For either of these modes, the HAMT has some internal performance optimizations.

    Note that in read+write, the real root node id IS NOT VALID. You should call `make_read_only()` to convert to read only mode and then read `root_node_id`.

    These optimizations also trade off performance for memory use. Use `cache_size` to monitor the approximate memory usage. Be warned that for large key-value mapping sets this may take a bit to run. Use `cache_vacate` if you are over your memory limits.

    #### IPFS HAMT Sample Code
    ```python
    kubo_cas = KuboCAS() # connects to a local kubo node with the default endpoints
    hamt = await HAMT.build(cas=kubo_cas)
    await hamt.set("foo", "bar")
    assert (await hamt.get("foo")) == "bar"
    await hamt.make_read_only()
    cid = hamt.root_node_id # our root node CID
    print(cid)
    ```
    """

    def __init__(
        self,
        cas: ContentAddressedStore,
        hash_fn: Callable[[bytes], bytes] = blake3_hashfn,
        root_node_id: IPLDKind | None = None,
        read_only: bool = False,
        max_bucket_size: int = 4,
        values_are_bytes: bool = False,
    ):
        """
        Use `build` if you need to create a completely empty HAMT, as this requires some async operations with the CAS. For what each of the constructor input variables refer to, check the documentation with the matching names below.
        """

        self.cas: ContentAddressedStore = cas
        """The backing storage system. py-hamt provides an implementation `KuboCAS` for IPFS."""

        self.hash_fn: Callable[[bytes], bytes] = hash_fn
        """
        This is the hash function used to place a key-value within the HAMT.

        To provide your own hash function, create a function that takes in arbitrarily long bytes and returns the hash bytes.

        It's important to note that the resulting hash must must always be a multiple of 8 bits since python bytes object can only represent in segments of bytes, and thus 8 bits.

        Theoretically your hash size must only be a minimum of 1 byte, and there can be less than or the same number of hash collisions as the bucket size. Any more and the HAMT will most likely throw errors.
        """

        self.lock: asyncio.Lock = asyncio.Lock()
        """@private"""

        self.values_are_bytes: bool = values_are_bytes
        """Set this to true if you are only going to be storing python bytes objects into the hamt. This will improve performance by skipping a serialization step from IPLDKind.

        This is theoretically safe to change in between operations, but this has not been verified in testing, so only do this at your own risk.
        """

        if max_bucket_size < 1:
            raise ValueError("Bucket size maximum must be a positive integer")
        self.max_bucket_size: int = max_bucket_size
        """
        This is only important for tuning performance when writing! For reading a HAMT that was written with a different max bucket size, this does not need to match and can be left unprovided.

        This is an internal detail that has been exposed for performance tuning. The HAMT handles large key-value mapping sets even on a content addressed system by essentially sharding all the mappings across many smaller Nodes. The memory footprint of each of these Nodes footprint is a linear function of the maximum bucket size. Larger bucket sizes will result in larger Nodes, but more time taken to retrieve and decode these nodes from your backing CAS.

        This must be a positive integer with a minimum of 1.
        """

        self.root_node_id: IPLDKind = root_node_id
        """
        This is type IPLDKind but the documentation generator pdoc mangles it a bit.

        Read from this only when in read mode to get something valid!
        """

        self.read_only: bool = read_only
        """Clients should NOT modify this.

        This is here for checking whether the HAMT is in read only or read/write mode.

        The distinction is made for performance and correctness reasons. In read only mode, the HAMT has an internal read cache that can speed up operations. In read/write mode, for reads the HAMT maintains strong consistency for reads by using async locks, and for writes the HAMT writes to an in memory buffer rather than performing (possibly) network calls to the underlying CAS.
        """
        self.node_store: NodeStore
        """@private"""
        if read_only:
            self.node_store = ReadCacheStore(self)
        else:
            self.node_store = InMemoryTreeStore(self)

    @classmethod
    async def build(cls, *args: Any, **kwargs: Any) -> "HAMT":
        """
        Use this if you are initializing a completely empty HAMT! That means passing in None for the root_node_id. Method arguments are the exact same as `__init__`. If the root_node_id is not None, this will have no difference than creating a HAMT instance with __init__.

        This separate async method is required since initializing an empty HAMT means sending some internal objects to the underlying CAS, which requires async operations. python does not allow for an async __init__, so this method is separately provided.
        """
        hamt = cls(*args, **kwargs)
        if hamt.root_node_id is None:
            hamt.root_node_id = await hamt.node_store.save(None, Node())
        return hamt

    # This is typically a massive blocking operation, you dont want to be running this concurrently with a bunch of other operations, so it's ok to have it not be async
    async def make_read_only(self) -> None:
        """
        Makes the HAMT read only, which allows for more parallel read operations. The HAMT also needs to be in read only mode to get the real root node ID.

        In read+write mode, the HAMT normally has to block separate get calls to enable strong consistency in case a set/delete operation falls in between.
        """
        async with self.lock:
            inmemory_tree: InMemoryTreeStore = cast(InMemoryTreeStore, self.node_store)
            await inmemory_tree.vacate()

            self.read_only = True
            self.node_store = ReadCacheStore(self)

    async def enable_write(self) -> None:
        """
        Enable both reads and writes. This creates an internal structure for performance optimizations which will result in the root node ID no longer being valid, in order to read that at the end of your operations you must first use `make_read_only`.
        """
        async with self.lock:
            # The read cache has no writes that need to be sent upstream so we can remove it without vacating
            self.read_only = False
            self.node_store = InMemoryTreeStore(self)

    async def cache_size(self) -> int:
        """
        Returns the memory used by some internal performance optimization tools in bytes.

        This is async concurrency safe, so call it whenever. This does mean it will block and wait for other writes to finish however.

        Be warned that this may take a while to run for large HAMTs.

        For more on memory management, see the `HAMT` class documentation.
        """
        if self.read_only:
            return self.node_store.size()
        async with self.lock:
            return self.node_store.size()

    async def cache_vacate(self) -> None:
        """
        Vacate and completely empty out the internal read/write cache.

        Be warned that this may take a while if there have been a lot of write operations.

        For more on memory management, see the `HAMT` class documentation.
        """
        if self.read_only:
            await self.node_store.vacate()
        else:
            async with self.lock:
                await self.node_store.vacate()

    async def _reserialize_and_link(
        self, node_stack: list[tuple[IPLDKind, Node]]
    ) -> None:
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
            is_root: bool = stack_index == 0
            if node.is_empty() and not is_root:
                # Unlink from the rest of the tree
                _, prev_node = node_stack[stack_index - 1]
                # When removing links, don't worry about two nodes having the same link since all nodes are guaranteed to be different by the removal of empty nodes after every single operation
                for link_index in prev_node.iter_link_indices():
                    link = prev_node.get_link(link_index)
                    if link == old_id:
                        # Delete the link by making it an empty bucket
                        prev_node.data[link_index] = {}
                        break

                # Remove from our stack, continue reserializing up the tree
                node_stack.pop(stack_index)
                continue

            # If not an empty node, just reserialize like normal and replace this one
            new_store_id: IPLDKind = await self.node_store.save(old_id, node)
            node_stack[stack_index] = (new_store_id, node)

            # If this is not the last i.e. root node, we need to change the linking of the node prior in the list since we just reserialized
            if not is_root:
                _, prev_node = node_stack[stack_index - 1]
                prev_node.replace_link(old_id, new_store_id)

    # automatically skip encoding if the value provided is of the bytes variety
    async def set(self, key: str, val: IPLDKind) -> None:
        """Write a key-value mapping."""
        if self.read_only:
            raise Exception("Cannot call set on a read only HAMT")

        data: bytes
        if self.values_are_bytes:
            data = cast(
                bytes, val
            )  # let users get an exception if they pass in a non bytes when they want to skip encoding
        else:
            data = dag_cbor.encode(val)

        pointer: IPLDKind = await self.cas.save(data, codec="raw")
        await self._set_pointer(key, pointer)

    async def _set_pointer(self, key: str, val_ptr: IPLDKind) -> None:
        async with self.lock:
            node_stack: list[tuple[IPLDKind, Node]] = []
            root_node: Node = await self.node_store.load(self.root_node_id)
            node_stack.append((self.root_node_id, root_node))

            # FIFO queue to keep track of all the KVs we need to insert
            # This is needed if any buckets overflow and so we need to reinsert all those KVs
            kvs_queue: list[tuple[str, IPLDKind]] = []
            kvs_queue.append((key, val_ptr))

            while len(kvs_queue) > 0:
                _, top_node = node_stack[-1]
                curr_key, curr_val_ptr = kvs_queue[0]

                raw_hash: bytes = self.hash_fn(curr_key.encode())
                map_key: int = extract_bits(raw_hash, len(node_stack) - 1, 8)

                item = top_node.data[map_key]
                if isinstance(item, list):
                    next_node_id: IPLDKind = item[0]
                    next_node: Node = await self.node_store.load(next_node_id)
                    node_stack.append((next_node_id, next_node))
                elif isinstance(item, dict):
                    bucket: dict[str, IPLDKind] = item

                    # If this bucket already has this same key, or has space, just rewrite the value and then go work on the others in the queue
                    if curr_key in bucket or len(bucket) < self.max_bucket_size:
                        bucket[curr_key] = curr_val_ptr
                        kvs_queue.pop(0)
                        continue

                    # The current key is not in the bucket and the bucket is too full, so empty KVs from the bucket and restart insertion
                    for k in bucket:
                        v_ptr = bucket[k]
                        kvs_queue.append((k, v_ptr))

                    # Create a new link to a new node so that we can reflow these KVs into a new subtree
                    new_node = Node()
                    new_node_id: IPLDKind = await self.node_store.save(None, new_node)
                    link: list[IPLDKind] = [new_node_id]
                    top_node.data[map_key] = link

            # Finally, reserialize and fix all links, deleting empty nodes as needed
            await self._reserialize_and_link(node_stack)
            self.root_node_id = node_stack[0][0]

    async def delete(self, key: str) -> None:
        """Delete a key-value mapping."""

        # Also deletes the pointer at the same time so this doesn't have a _delete_pointer duo
        if self.read_only:
            raise Exception("Cannot call delete on a read only HAMT")

        async with self.lock:
            raw_hash: bytes = self.hash_fn(key.encode())

            node_stack: list[tuple[IPLDKind, Node]] = []
            root_node: Node = await self.node_store.load(self.root_node_id)
            node_stack.append((self.root_node_id, root_node))

            created_change: bool = False
            while True:
                _, top_node = node_stack[-1]
                map_key: int = extract_bits(raw_hash, len(node_stack) - 1, 8)

                item = top_node.data[map_key]
                if isinstance(item, dict):
                    bucket = item
                    if key in bucket:
                        del bucket[key]
                        created_change = True
                    # Break out since whether or not the key is in the bucket, it should have been here so either now reserialize or raise a KeyError
                    break
                elif isinstance(item, list):
                    link: IPLDKind = item[0]
                    next_node: Node = await self.node_store.load(link)
                    node_stack.append((link, next_node))

            # Finally, reserialize and fix all links, deleting empty nodes as needed
            if created_change:
                await self._reserialize_and_link(node_stack)
                self.root_node_id = node_stack[0][0]
            else:
                # If we didn't make a change, then this key must not exist within the HAMT
                raise KeyError

    async def get(self, key: str) -> IPLDKind:
        """Get a value."""
        pointer: IPLDKind = await self.get_pointer(key)
        data: bytes = await self.cas.load(pointer)
        if self.values_are_bytes:
            return data
        else:
            return dag_cbor.decode(data)

    async def get_pointer(self, key: str) -> IPLDKind:
        """
        Get a store ID that points to the value for this key.

        This is useful for some applications that want to implement a read cache. Due to the restrictions of `ContentAddressedStore` on IDs, pointers are regarded as immutable by python so they can be easily used as IDs for read caches. This is utilized in `ZarrHAMTStore` for example.
        """
        # If read only, no need to acquire a lock
        pointer: IPLDKind
        if self.read_only:
            pointer = await self._get_pointer(key)
        else:
            async with self.lock:
                pointer = await self._get_pointer(key)

        return pointer

    # Callers MUST handle acquiring a lock
    async def _get_pointer(self, key: str) -> IPLDKind:
        raw_hash: bytes = self.hash_fn(key.encode())

        current_id: IPLDKind = self.root_node_id
        current_depth: int = 0

        # Don't check if result is none but use a boolean to indicate finding something, this is because None is a possible value of IPLDKind
        result_ptr: IPLDKind = None
        found_a_result: bool = False
        while True:
            top_id: IPLDKind = current_id
            top_node: Node = await self.node_store.load(top_id)
            map_key: int = extract_bits(raw_hash, current_depth, 8)

            # Check if this key is in one of the buckets
            item = top_node.data[map_key]
            if isinstance(item, dict):
                bucket = item
                if key in bucket:
                    result_ptr = bucket[key]
                    found_a_result = True
                    break

            if isinstance(item, list):
                link: IPLDKind = item[0]
                current_id = link
                current_depth += 1
                continue

            # Nowhere left to go, stop walking down the tree
            break

        if not found_a_result:
            raise KeyError

        return result_ptr

    # Callers MUST handle locking or not on their own
    async def _iter_nodes(self) -> AsyncIterator[tuple[IPLDKind, Node]]:
        node_id_stack: list[IPLDKind] = [self.root_node_id]
        while len(node_id_stack) > 0:
            top_id: IPLDKind = node_id_stack.pop()
            node: Node = await self.node_store.load(top_id)
            yield (top_id, node)
            node_id_stack.extend(list(node.iter_links()))

    async def keys(self) -> AsyncIterator[str]:
        """
        AsyncIterator returning all keys in the HAMT.

        If the HAMT is write enabled, to maintain strong consistency this will obtain an async lock and not allow any other operations to proceed.

        When the HAMT is in read only mode however, this can be run concurrently with get operations.
        """
        if self.read_only:
            async for k in self._keys_no_locking():
                yield k
        else:
            async with self.lock:
                async for k in self._keys_no_locking():
                    yield k

    async def _keys_no_locking(self) -> AsyncIterator[str]:
        async for _, node in self._iter_nodes():
            for bucket in node.iter_buckets():
                for key in bucket:
                    yield key

    async def len(self) -> int:
        """
        Return the number of key value mappings in this HAMT.

        When the HAMT is write enabled, to maintain strong consistency it will acquire a lock and thus not allow any other operations to proceed until the length is fully done being calculated. If read only, then this can be run concurrently with other operations.
        """
        count: int = 0
        async for _ in self.keys():
            count += 1

        return count
