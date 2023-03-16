import typing
from py_hamt.bit_utils import extract_bits, set_bit, bitmap_has, rank
import math
from functools import cmp_to_key

# set defaults
DEFAULT_BIT_WIDTH = 8  # 2^8 = 256 buckets or children per node
DEFAULT_BUCKET_SIZE = 5  # array size for a bucket of values

# Using camelCase here to allign with the spec
DEFAULT_OPTIONS = {
    "bitWidth": DEFAULT_BIT_WIDTH,
    "bucketSize": DEFAULT_BUCKET_SIZE,
    "hashAlg": 0x23,
}


class KV:
    """Object representing a single key/value pair"""

    def __init__(self, key: bytes, value):
        self.key = key
        self.value = value

    def to_serializable(self):
        return [self.key, self.value]

    @staticmethod
    def from_serializable(obj):
        assert isinstance(obj, list)
        assert len(obj) == 2
        return KV(obj[0], obj[1])


class Element:
    """an element in the data array that each node holds, each element could be either a container of
    an array (bucket) of KVs or a link to a child node
    """

    def __init__(self, bucket: typing.Optional[typing.List[KV]] = None, link=None):
        assert bucket or link
        assert not (bucket and link)
        self.bucket = bucket  # should be array of KV's
        self.link = link

    def to_serializable(self):
        if self.bucket is not None:
            return [c.to_serializable() for c in self.bucket]
        else:
            return self.link

    @staticmethod
    def from_serializable(is_link, obj):
        if is_link(obj):
            return Element(None, obj)
        elif isinstance(obj, list):
            return Element([KV.from_serializable(ele) for ele in obj])


class Hamt:
    hasher_registry: typing.Dict[int, typing.Dict] = {}

    @classmethod
    def register_hasher(
        cls, hash_alg: int, hash_bytes: int, hasher: typing.Callable[[bytes], bytes]
    ):
        """Register a hashing algorithm with the class

        Args:
            hash_alg (int): A multicodec
                (see https://github.com/multiformats/multicodec/blob/master/table.csv)
                hash function identifier  e.g. `0x23` for `murmur3-32`.
            hash_bytes (int): The length of the `bytes` object returned by `hasher`
            hasher (typing.Callable): Hash function that converts a string to bytes
        """
        if not isinstance(hash_alg, int):
            raise TypeError("`hash_alg` should be of type `int`")
        if not isinstance(hash_bytes, int):
            raise TypeError("`hash_bytes` should be of type `int`")
        if not callable(hasher):
            raise TypeError("`hasher` should be a function")
        cls.hasher_registry[hash_alg] = {"hash_bytes": hash_bytes, "hasher": hasher}

    @classmethod
    def create(
        cls,
        store,
        options: dict = DEFAULT_OPTIONS,
        map: typing.Optional[bytes] = None,
        depth: int = 0,
        data: typing.Optional[typing.List[Element]] = None,
    ) -> "Hamt":
        """Creates a Hamt and uses `save` to generate an id for it

        Args:
            store: backing store for new node
            options (dict): Configuration for new node. Defaults to DEFAULT_OPTIONS.
            map (typing.Optional[bytes]): bitmap used to quickly find keys. Defaults to None.
            depth (int, optional): how deeply in the HAMT this node sits. Defaults to 0.
            data (typing.Optional[typing.List[Element]], optional):
                Elements representing all data stored in this node. Defaults to None.

        Returns:
            Hamt: Created and saved HAMT
        """
        new_node = cls(store, options, map, depth, data)
        return save(store, new_node)

    def __init__(
        self,
        store,
        options: dict = DEFAULT_OPTIONS,
        map: typing.Optional[bytes] = None,
        depth: int = 0,
        data: typing.Optional[typing.List[Element]] = None,
    ):
        self.store = store
        self.data = data if data is not None else []
        for e in self.data:
            if not isinstance(e, Element):
                raise Exception("`data` array must contain only `Element` types")

        self.id = None

        if map and not isinstance(map, bytes):
            raise TypeError("map must be bytes")
        map_length = math.ceil(2 ** options["bitWidth"] / 8)

        if map and map_length != len(map):
            raise Exception(f"`map` must be a bytes of length {map_length}")

        self.map = map if map is not None else bytes([0 for _ in range(map_length)])

        self.depth = depth

        self.config = options
        hash_bytes = self.hasher_registry[self.config["hashAlg"]]["hash_bytes"]
        if self.depth > math.floor((hash_bytes * 8) / self.config["bitWidth"]):
            raise Exception("Overflow: maximum tree depth reached")

    def hasher(self) -> typing.Callable[[bytes], bytes]:
        """Gets the hashing function corresponding to the hash_alg in config
        Returns:
            typing.Callable: hasher stored in `hasher_registry`
        """
        return self.hasher_registry[self.config["hashAlg"]]["hasher"]

    def set(
        self, key: typing.Union[str, bytes], value, _cached_hash: typing.Optional[bytes] = None
    ) -> "Hamt":
        """Create a new `Hamt` instance identical to this one but with `key` set to `value`.

        Args:
            key (typing.Union[str, bytes]): key used to locate value within Hamt
            value: value to be placed at key
            _cached_hash (typing.Optional[bytes], optional):
                If key has already been hashed, cache it in this variable for use in recursion.
                Defaults to None.

        Returns:
            Hamt: Instance of Hamt identical to `self` but with `key` set to `value`
        """
        if not isinstance(key, bytes):
            key = key.encode("utf-8")
        hashed_key = (
            _cached_hash if _cached_hash is not None else self.hasher()(key)
        )
        bitpos = extract_bits(hashed_key, self.depth, self.config["bitWidth"])
        if bitmap_has(self.map, bitpos):
            find_elem = self.find_element(bitpos, key)
            if "data" in find_elem:
                data = find_elem["data"]
                if data["found"]:
                    if data["bucket_index"] is None or data["bucket_entry"] is None:
                        raise Exception("Unexpected error")
                    if data["bucket_entry"].value == value:
                        return self
                    return self.update_bucket(
                        data["element_at"],
                        data["bucket_index"],
                        key,
                        value,
                    )
                else:
                    if len(data["element"].bucket) >= self.config["bucketSize"]:
                        new_map = self.replace_bucket_with_node(data["element_at"])
                        return new_map.set(key, value, hashed_key)
                    return self.update_bucket(data["element_at"], -1, key, value)
            elif "link" in find_elem:
                link = find_elem["link"]
                child = load(
                    self.store, link["element"].link, self.depth + 1, self.config
                )
                assert child
                new_child = child.set(key, value, hashed_key)
                return self.update_node(link["element_at"], new_child)
            else:
                raise Exception("Neither link nor data found")
        else:
            return self.add_new_element(bitpos, key, value)

    def get(self, key: typing.Union[str, bytes], _cached_hash: typing.Optional[bytes] = None):
        """Find and return a value for the given `key` if it exists within this `Hamt`.
        Raise KeyError otherwise

        Args:
            key (typing.Union[str, bytes]): where to find value
            _cached_hash (typing.Optional[bytes], optional):
                If key has already been hashed, cache it in this variable for use in recursion.
                Defaults to None.

        Raises:
            KeyError: Raised when `key` cannot be found in `self`

        Returns:
            value located at `key`
        """
        if not isinstance(key, bytes):
            key = key.encode("utf-8")

        hashed_key = (
            _cached_hash if isinstance(_cached_hash, bytes) else self.hasher()(key)
        )
        bitpos = extract_bits(hashed_key, self.depth, self.config["bitWidth"])
        if bitmap_has(self.map, bitpos):
            find_elem = self.find_element(bitpos, key)
            if "data" in find_elem:
                data = find_elem["data"]
                if data["found"]:
                    return data["bucket_entry"].value
                else:
                    raise KeyError("not in hamt")
            elif "link" in find_elem:
                link = find_elem["link"]
                child = load(
                    self.store, link["element"].link, self.depth + 1, self.config
                )
                assert child
                return child.get(key, hashed_key)
            else:
                raise Exception("Neither link nor data found")
        else:
            raise KeyError("not in hamt")

    def delete(self, key):
        raise NotImplementedError

    def has(self, key: typing.Union[str, bytes]) -> bool:
        """Determines whether hamt has `key`

        Args:
            key (typing.Union[str, bytes])

        Returns:
            bool: whether hamt has `key`
        """
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def size(self) -> int:
        """Gets the total number of keys in the hamt

        Returns:
            int
        """
        c = 0
        for e in self.data:
            if e.bucket is not None:
                c += len(e.bucket)
            else:
                child = load(self.store, e.link, self.depth + 1, self.config)
                c += child.size()
        return c

    def keys(self) -> typing.Iterator[str]:
        """Get iterator with all keys in hamt

        Yields:
            bytes: key
        """
        for e in self.data:
            if e.bucket is not None:
                for kv in e.bucket:
                    yield kv.key
            else:
                child = load(self.store, e.link, self.depth + 1, self.config)
                yield from child.keys()

    def ids(self):
        """Get iterator with all ids in hamt

        Yields:
            id
        """
        yield self.id
        for e in self.data:
            if e.link:
                child = load(self.store, e.link, self.depth + 1, self.config)
                yield from child.ids()

    def update_bucket(self, element_at: int, bucket_at: int, key, value) -> "Hamt":
        """Modify bucket with new key/value

        Args:
            element_at (int): index of element containing bucket to update
            bucket_at (int): index of kv within bucket to update. When -1,
                append to bucket then sort. Otherwise, update bucket at this index,
                meaning that `key` already exists within bucket
            key: key to set
            value: value corresponding to key

        Returns:
            Hamt: Node with key set to value through bucket update
        """

        old_element = self.data[element_at]

        if old_element.bucket is None:
            raise Exception("Expected element with bucket")

        new_element = Element(list(old_element.bucket))
        new_kv = KV(key, value)

        if bucket_at == -1:
            new_element.bucket.append(new_kv)
            new_element.bucket.sort(key=cmp_to_key(byte_compare))
        else:
            new_element.bucket[bucket_at] = new_kv

        new_data = list(self.data)
        new_data[element_at] = new_element
        return self.create(self.store, self.config, self.map, self.depth, new_data)

    def find_element(self, bitpos: int, key) -> dict:
        """Find a key within bucket or link in element located at bitpos

        Args:
            bitpos (int): bitpos within node used to find element
            key: key to locate

        Returns:
            dict: dict representing whether the key was found, whether it was found in a link
                or bucket and where to locate the key within the bucket
        """
        element_at = rank(self.map, bitpos)
        element = self.data[element_at]
        if element.bucket:
            for bucket_index in range(len(element.bucket)):
                bucket_entry = element.bucket[bucket_index]
                if byte_compare(bucket_entry.key, key) == 0:
                    return {
                        "data": {
                            "found": True,
                            "element_at": element_at,
                            "element": element,
                            "bucket_index": bucket_index,
                            "bucket_entry": bucket_entry,
                        }
                    }
            return {
                "data": {
                    "found": False,
                    "element_at": element_at,
                    "element": element,
                    "bucket_index": None,
                    "bucket_entry": None,
                }
            }
        assert element.link
        return {"link": {"element_at": element_at, "element": element}}

    def update_node(self, element_at: int, new_child: "Hamt") -> "Hamt":
        """Update a child node

        Args:
            element_at (int): index of element at which to insert child
            new_child (Hamt): child to update with

        Returns:
            Hamt: New Hamt with child node updated at the given index
        """
        assert new_child.id
        new_element = Element(None, new_child.id)
        new_data = list(self.data)
        new_data[element_at] = new_element
        return self.create(self.store, self.config, self.map, self.depth, new_data)

    def replace_bucket_with_node(self, element_at: int) -> "Hamt":
        """Bucket has overflowed and needs to be replaced with a node

        Args:
            node (Hamt): parent of bucket that has overflowed
            element_at (int): index of element containing overflowing bucket

        Returns:
            Hamt: Hamt with bucket replaced with child node
        """
        new_node = Hamt(self.store, self.config, None, self.depth + 1)
        element = self.data[element_at]
        assert element
        if element.bucket is None:
            raise Exception("Expected element with bucket")

        for c in element.bucket:
            new_node = new_node.set(c.key, c.value)
        new_node = save(self.store, new_node)
        new_data = list(self.data)
        new_data[element_at] = Element(None, new_node.id)
        return self.create(self.store, self.config, self.map, self.depth, new_data)

    def add_new_element(self, bitpos: int, key, value) -> "Hamt":
        """Insert a new element containing a bucket with a single kv into node

        Args:
            bitpos (int): location of element within bitmap
            key: key to add
            value: value

        Returns:
            Hamt: Node with element inserted
        """
        insert_at = rank(self.map, bitpos)
        new_data = list(self.data)
        new_data.insert(insert_at, Element([KV(key, value)]))
        new_map = set_bit(self.map, bitpos, True)
        return self.create(self.store, self.config, new_map, self.depth, new_data)

    @staticmethod
    def is_hamt(node) -> bool:
        return isinstance(node, Hamt)

    @staticmethod
    def from_serializable(
        store, id, serializable: typing.Union[list, dict], options: typing.Optional[dict], depth: int = 0
    ) -> "Hamt":
        """Generates a `Hamt` object from a serialized dict or list, which typically comes from
        `Hamt.to_serializable`

        Args:
            store: backing store
            id: id representing the entire node in the store
            serializable (typing.Union[list, dict]): object generated from `to_serializable` to be turned
                into a `Hamt` object
            options (dict): Config for new hamt. Will be ignored if `serializable` is of depth 0
                (and thus has its own options)
            depth (int, optional): How deep in the hamt this node will sit. Defaults to 0.

        Returns:
            Hamt: node generated from `serializable`
        """
        if depth == 0:
            if not is_root_serializable(serializable):
                raise Exception(
                    "Loaded  object does not appear to be an Hamt root (depth==0)"
                )
            # don't use passed-in options, since the object itself specifies options
            options = serializable_to_options(serializable)
            hamt = serializable["hamt"]
        else:
            if not is_serializable(serializable):
                raise Exception(
                    "Loaded object does not appear to be an Hamt node (depth>0)"
                )
            hamt = serializable
        data = [Element.from_serializable(store.is_link, ele) for ele in hamt[1]]

        node = Hamt(store, options, hamt[0], depth, data)
        if id is not None:
            node.id = id
        return node

    def to_serializable(self) -> typing.Union[list, dict]:
        """Turns hamt into serialized list or dict

        Returns:
            typing.Union[list, dict]: serialized version of the `self`
        """
        data = [ele.to_serializable() for ele in self.data]
        hamt = [self.map, data]
        if self.depth != 0:
            return hamt

        return {
            "hashAlg": self.config["hashAlg"],
            "bucketSize": self.config["bucketSize"],
            "hamt": hamt,
        }

    def from_child_serializable(
        self, id, serializable: typing.Union[list, dict], depth: int
    ) -> "Hamt":
        """A convenience shortcut to `hamt.from_serializable` that uses this IAMap node
        instance's backing `store` and configuration `options`. Intended to be used to instantiate
        child IAMap nodes from a root IAMap node.

        Args:
            id: id for child
            serializable (typing.Union[list, dict]):
            depth (int): _description_

        Returns:
            Hamt: new HAMT generated from serializable object
        """
        return self.from_serializable(self.store, id, serializable, self.config, depth)

    def direct_entry_count(self) -> int:
        """Count the number of direct children of the node (not links)

        Returns:
            int: number of direct children of the node
        """
        count = 0
        for ele in self.data:
            if ele.bucket:
                count += len(ele.bucket)
        return count

    def direct_node_count(self) -> int:
        """Count the number of children of the node that are also Hamt

        Returns:
            int: number of child links for this node
        """
        count = 0
        for ele in self.data:
            if ele.link:
                count += 1
        return count

    def is_invariant(self) -> bool:
        """Perform a check on this node and its children that it is in
        canonical format for the current data. As this uses `size()` to calculate the total
        number of entries in this node and its children, it performs a full
        scan of nodes and therefore incurs a load and deserialisation cost for each child node.
        A `false` result from this method suggests a flaw in the implemetation.

        Returns:
            bool: whether the tree is in its canonical form
        """
        size = self.size()
        entry_arity = self.direct_entry_count()
        node_arity = self.direct_node_count()
        arity = entry_arity + node_arity
        size_predicate = 2
        if node_arity == 0:
            size_predicate = min(2, entry_arity)

        inv1 = size - entry_arity >= 2 * (arity - entry_arity)
        inv2 = size_predicate == 0 if arity == 0 else True
        inv3 = size_predicate == 1 if (arity == 1 and entry_arity == 1) else True
        inv4 = size_predicate == 2 if arity >= 2 else True
        inv5 = (
            node_arity >= 0 and entry_arity >= 0 and (entry_arity + node_arity == arity)
        )

        return inv1 and inv2 and inv3 and inv4 and inv5


def serializable_to_options(serializable: typing.Union[dict, list]) -> dict:
    """Turn serialized node into configs that can be passed into Hamt as options

    Args:
        serializable (dict): Serialized node

    Returns:
        dict: Options generated for serialized node
    """
    return {
        "hashAlg": serializable["hashAlg"],
        "bitWidth": int(
            math.log2(len(serializable["hamt"][0]) * 8)
        ),  # inverse of (2**bitWidth) / 8
        "bucketSize": serializable["bucketSize"],
    }


def is_serializable(serializable: typing.Union[dict, list]) -> bool:
    """Check if `serializable` is a valid serialized Hamt

    Args:
        serializable (typing.Union[dict, list]): Serialized object to test

    Returns:
        bool: whether object is valid hamt
    """
    if isinstance(serializable, list):
        return (
            len(serializable) == 2
            and isinstance(serializable[0], bytes)
            and isinstance(serializable[1], list)
        )
    return is_root_serializable(serializable)


def is_root_serializable(serializable: typing.Union[dict, list]) -> bool:
    """Whether `serializable` is the serialized root of a Hamt

    Args:
        serializable (dict): object to check

    Returns:
        bool: whether object is serialized root
    """
    return (
        isinstance(serializable, dict)
        and isinstance(serializable.get("hashAlg"), int)
        and isinstance(serializable["hamt"], list)
        and is_serializable(serializable["hamt"])
    )


def save(store, new_node: Hamt) -> Hamt:
    """Save a Hamt `new_node` to the store, giving it an id

    Args:
        store: backing store
        new_node (Hamt): node to save to store

    Returns:
        Hamt: node with store-generated id attached
    """
    id = store.save(new_node.to_serializable())
    new_node.id = id
    return new_node


def load(store, id, depth: int = 0, options: typing.Optional[dict] = None) -> Hamt:
    """Use id to load a node from the store

    Args:
        store: backing store
        id: id to load from store
        depth (int, optional): Depth at which node sits in tree. Defaults to 0.
        options (typing.Optional[dict], optional): config to be passed into Hamt constructor.
            Defaults to None.

        Hamt: node loaded from store
    """
    if depth != 0 and not options:
        raise Exception("Cannot load() without options at depth > 0")
    serialized = store.load(id)
    return Hamt.from_serializable(store, id, serialized, options, depth)


def byte_compare(b1: typing.Union[bytes, str, KV], b2: typing.Union[bytes, str, KV]) -> int:
    """Compare bytes/keys for use in sorting function

    Args:
        b1 (typing.Union[bytes, KV]): first byte/key to compare
        b2 (typing.Union[bytes, KV]): second byte/key to compare

    Returns:
        int: 1 if b1 > b2, -1 if b2 > b1, 0 if b1 == b2
    """
    if hasattr(b1, "key"):
        b1 = b1.key
    if hasattr(b2, "key"):
        b2 = b2.key
    x = len(b1)
    y = len(b2)
    for i in range(min(x, y)):
        if b1[i] != b2[i]:
            if b1[i] < b2[i]:
                return -1
            if b1[i] > b2[i]:
                return 1
    if x < y:
        return -1
    if x > y:
        return 1
    return 0
