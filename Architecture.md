# py-hamt Library Architecture Analysis

## Overview

The py-hamt library is a Python implementation of a Hash Array Mapped Trie (HAMT) designed for content-addressed storage systems like IPFS. It provides efficient storage and retrieval of large key-value mappings using the IPLD data model.

## Core Architecture

### 1. HAMT Structure (`hamt.py`)

The core HAMT implementation follows a hierarchical trie structure where:
- **Nodes** contain 256 buckets (indexed 0-255)
- Each bucket can either be:
  - A dictionary containing key-value mappings (when bucket size ≤ max_bucket_size)
  - A link to a child Node (when bucket overflows)
- Hash values are consumed 8 bits at a time to determine bucket indices

#### Key Components:

**Node Class** (`hamt.py:62`)
- Fixed array of 256 elements representing buckets
- Uses `list[IPLDKind]` where empty dicts `{}` represent empty buckets
- Links are stored as single-element lists `[link_id]`
- Serializes to DAG-CBOR format for content addressing

**HAMT Class** (`hamt.py:287`)
- Main interface for trie operations
- Supports both read-only and read-write modes
- Uses asyncio locks for thread safety in write mode
- Implements two node storage strategies via NodeStore abstraction

### 2. Storage Abstraction (`store.py`)

**ContentAddressedStore** (`store.py:11`)
- Abstract base class for content-addressed storage backends
- Returns immutable IDs (IPLDKind) for stored content
- Two codec types: "raw" for data, "dag-cbor" for structured content

**KuboCAS** (`store.py:74`)
- IPFS implementation using Kubo daemon
- Uses RPC API for writes (`/api/v0/add`)
- Uses HTTP Gateway for reads (`/ipfs/{cid}`)
- Supports authentication and custom headers
- Implements connection pooling and concurrency limiting

**InMemoryCAS** (`store.py:37`)
- Testing implementation using in-memory dictionary
- Content-addressed via Blake3 hashing

### 3. Node Storage Strategies

**ReadCacheStore** (`hamt.py:150`)
- Used in read-only mode
- Implements LRU-style caching of loaded nodes
- Cannot perform writes (throws exception)
- Optimized for concurrent read operations

**InMemoryTreeStore** (`hamt.py:180`)
- Used in read-write mode
- Maintains modified nodes in memory buffer
- Uses UUID4 integers as temporary node IDs
- Implements sophisticated flush algorithm during `vacate()`

### 4. Zarr Integration (`zarr_hamt_store.py`)

**ZarrHAMTStore** (`zarr_hamt_store.py:11`)
- Implements Zarr v3 Store interface
- Provides metadata caching for `zarr.json` files
- Supports directory listing operations with efficient prefix matching
- Key insight: Zarr keys map directly to HAMT keys without transformation

**SimpleEncryptedZarrHAMTStore** (`encryption_hamt_store.py:12`)
- Extends ZarrHAMTStore with ChaCha20-Poly1305 encryption
- Encrypts all data including metadata
- Uses 24-byte nonces and 16-byte authentication tags

## Value Setting Mechanism & Data Flow

### Setting Values (`_set_pointer` in `hamt.py:505`)

1. **Hash and Navigate**: Key is hashed, bits extracted to determine path
2. **Queue-based Insertion**: Uses FIFO queue to handle bucket overflows
3. **Bucket Overflow Handling**: When bucket exceeds max_bucket_size:
   - All existing items moved to queue for reinsertion
   - New child node created and linked
   - Continues insertion process at deeper level
4. **Tree Rebalancing**: After insertion, `_reserialize_and_link` updates all affected nodes

### Getting Values (`_get_pointer` in `hamt.py:618`)

1. **Hash Traversal**: Follow hash-determined path through trie
2. **Bucket Search**: Check final bucket for key
3. **Pointer Resolution**: Return content-addressed pointer
4. **Value Retrieval**: Load actual value from CAS using pointer

### Memory Management

**Read-Write Mode**:
- Uses `InMemoryTreeStore` with UUID-based temporary IDs
- Modified nodes stay in memory until `make_read_only()` or `cache_vacate()`
- Flush algorithm uses DFS to preserve parent-child relationships

**Read-Only Mode**:
- Switches to `ReadCacheStore` for better read performance
- Allows concurrent operations without locks
- Cache size can be monitored and manually vacated

## Architecture Gotchas & Edge Cases

### 1. Mode Switching Complexity
- **Problem**: HAMT can switch between read-only and read-write modes
- **Gotcha**: In read-write mode, `root_node_id` is invalid until `make_read_only()`
- **Why**: InMemoryTreeStore uses temporary UUIDs that aren't content-addressed
- **Solution**: Always call `make_read_only()` before reading `root_node_id`

### 2. Thread Safety Limitations
- **Problem**: Only async-safe, not thread-safe in write mode
- **Gotcha**: Multiple threads writing simultaneously can corrupt state
- **Why**: Uses asyncio.Lock, not threading.Lock
- **Solution**: Use single event loop for all write operations

### 3. Hash Function Constraints
- **Problem**: Hash must be multiple of 8 bits (bytes)
- **Gotcha**: Custom hash functions with odd bit lengths will fail
- **Why**: `extract_bits` assumes byte-aligned hash values
- **Solution**: Ensure hash functions return byte-aligned results

### 4. Bucket Size Tuning
- **Problem**: max_bucket_size affects performance vs memory trade-offs
- **Gotcha**: Very small bucket sizes (1) force deep trees, large sizes create big nodes
- **Why**: Small buckets = more CAS operations, large buckets = bigger serialized nodes
- **Solution**: Test with your specific workload (default 4 is reasonable)

### 5. Empty Node Pruning
- **Problem**: Deletions can leave empty nodes in tree
- **Gotcha**: Empty nodes are automatically pruned except root
- **Why**: Content addressing means identical empty nodes have same hash
- **Solution**: Pruning is automatic, but be aware root can become empty

### 6. Serialization Edge Cases
- **Problem**: InMemoryTreeStore nodes contain non-serializable UUID links
- **Gotcha**: Attempting to serialize buffer nodes directly will fail
- **Why**: UUID4.int values exceed DAG-CBOR integer limits
- **Solution**: Always use `vacate()` or `make_read_only()` before accessing serialized form

### 7. CAS ID Immutability Requirement
- **Problem**: CAS implementations must return immutable IDs
- **Gotcha**: Using mutable objects (lists, dicts) as IDs breaks assumptions
- **Why**: HAMT uses ID equality checks and as dictionary keys
- **Solution**: Ensure CAS returns immutable types (bytes, CID, int, str, etc.)

### 8. Zarr Metadata Caching
- **Problem**: ZarrHAMTStore caches `zarr.json` files
- **Gotcha**: Cache isn't invalidated on writes, can become stale
- **Why**: Zarr frequently re-reads metadata, caching improves performance
- **Solution**: Cache is updated on writes, but be aware of this optimization

### 9. Concurrent Operations in Read Mode
- **Problem**: Read-only mode allows concurrent access
- **Gotcha**: Switching to write mode while reads are happening is unsafe
- **Why**: Mode switch changes internal data structures
- **Solution**: Ensure all operations complete before mode switches

### 10. Key Encoding Assumptions
- **Problem**: Keys are encoded as UTF-8 bytes for hashing
- **Gotcha**: Non-UTF-8 string keys or binary keys need special handling
- **Why**: `key.encode()` assumes UTF-8 encoding
- **Solution**: Ensure keys are valid UTF-8 strings or modify key handling

## Performance Characteristics

### Time Complexity
- **Get/Set/Delete**: O(log₂₅₆ n) average case, O(depth) worst case
- **Iteration**: O(n) for all keys/values
- **Tree depth**: Typically 1-4 levels for most datasets

### Space Complexity
- **Node overhead**: ~2KB per node (256 × 8-byte pointers)
- **Memory efficiency**: Improves with higher bucket sizes
- **CAS efficiency**: Content addressing deduplicates identical subtrees

### Concurrency
- **Read-only mode**: Full concurrency support
- **Write mode**: Single-writer, async-safe
- **Mode switching**: Blocking operation requiring exclusive access

## Integration Patterns

### 1. IPFS Integration
```python
kubo_cas = KuboCAS()
hamt = await HAMT.build(cas=kubo_cas)
await hamt.set("key", "value")
await hamt.make_read_only()
cid = hamt.root_node_id  # IPFS CID
```

### 2. Zarr Storage
```python
hamt = await HAMT.build(cas=kubo_cas, values_are_bytes=True)
zarr_store = ZarrHAMTStore(hamt, read_only=False)
dataset.to_zarr(store=zarr_store, zarr_format=3)
```

### 3. Encrypted Storage
```python
encryption_key = get_random_bytes(32)
encrypted_store = SimpleEncryptedZarrHAMTStore(
    hamt, read_only=False, encryption_key=encryption_key, header=b"app-name"
)
```

The py-hamt library provides a robust, efficient implementation of HAMTs for content-addressed storage with careful attention to memory management, concurrency, and integration patterns. Understanding these architectural details and gotchas is crucial for successful implementation in production systems.
