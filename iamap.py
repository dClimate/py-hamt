from bit_utils import mask, set_bit, bitmap_has, index
import asyncio
import math
'''
WORK IN PROGRESS, NOT FINISHED

This is based on https://github.com/rvagg/iamap/blob/master/iamap.js

Currently, the bitswapping functionality (found in bit_utils.py) which is based on this (https://github.com/rvagg/iamap/blob/master/bit-utils.js) is done.

The next step is creating all the functionality present in iamap.js. Mainly, finishing the IAMap primary class methods (get/set/delete). 

The IAmap class methods, and related serializable functions in this file are unfinished. Some of the helper functions which are self-sufficient are finished,
and these are commented with [#*] to indicate they are finished.
'''

#defining python equivalent for TextEncoder/TextDecoder 
class TextEncoder():
    def __init__(self):
        pass
    
    def encode(self, text):
        """
        exp:
            >>> textencoder = TextEncoder()
            >>> textencoder.encode('$')
            >>> [36]
        """
        if isinstance(text, str):
            encoded_text = text.encode('utf-8')
            byte_array = bytearray(encoded_text)
            return list(byte_array)
        else:
            raise TypeError(f'Expecting a str but got {type(text)}')

class TextDecoder():
    def __init__(self):
        pass
    
    def decode(self, array):
        """
        exp:
            >>> textdecoder = TextDecoder()
            >>> textdecoder.decode([36])
            >>> $
        """
        if isinstance(array, list):
            return bytearray(array).decode('utf-8')
        elif isinstance(array, bytearray):
            return array.decode('utf-8')
        else:
            raise TypeError(f'expecting a list or bytearray got: {type(array)}')

#set defaults
default_bit_width = 8 # 2^8 = 256 buckets or children per node 
default_bucket_size = 5 # array size for a bucket of values

hasher_registry = {}
text_encoder = TextEncoder()

#*
def register_hasher(hash_alg, hash_bytes, hasher):
    hasher_registry[hash_alg] = {'hash_bytes': hash_bytes, 'hasher': hasher}

class KV(): 
    def __init__(self, key, value):
        self.key = key 
        self.value = value

    def to_serializable(self):
        return [self.key, self.value]

    def from_serializable(self, obj):
        #to do, add checks for obj
        #assert(Array.isArray(obj))
        #assert(obj.length === 2)
        return KV(obj[0], obj[1])

# a element in the data array that each node holds, each element could be either a container of
#  an array (bucket) of KVs or a link to a child node
class Element():
    def __init__(self, bucket=None, link=None):
        assert ((bucket == None) == (link != None)) # assert that either the bucket or link is None
        self.bucket = bucket #should be array of KV's
        self.link = link

    def to_serializable():
        pass

class IAMap(object):
    #Create a new IAMap instance with a backing store
    ## define async create func https://stackoverflow.com/questions/33128325/how-to-set-class-attribute-with-await-in-init
    @classmethod
    async def create(self, store, data=None, options={'bit_width': default_bit_width, 'bucket_size': default_bucket_size}, map = [], depth=0):
        self = IAMap()

        self.store = store
        # in javascript, self.data is made immutable
        self.data = data
        self.map = map
        self.depth = depth

        self.config = build_config(options)
        self.id = None

        return self

    # Asynchronously create a new `IAMap` instance identical to this one but with `key` set to `value`.
    async def set(self, key, value, _cached_hash):
        hash = _cached_hash #this needs to check if the submitted hash is a byte array, and if not await call the hasher function for this key... for now i will assume hash input
        bitpos = mask(hash, self.depth, self.options['bit_width'])
        if bitmap_has(self.map, bitpos):
            find_elem = find_element(self, bitpos, key)
            if 'data' in find_elem.keys():
                if find_elem['data']['found'] == True:
                    if find_elem['data'].bucket_index == None or find_elem['data'].bucket_entry == None:
                        raise Exception('Unexpected error')
                    if find_elem['data'].bucket_entry['value'] == value:
                        return self  

        pass 

    # Asynchronously find and return a value for the given `key` if it exists within this `IAMap`.
    async def get(self, key, _cached_hash):
        if not isinstance(key, bytes):
            key = text_encoder.encode(key)
        # hash = something (todo)
        bitpos = mask(hash, self.depth, self.config['bit_width'])
        if bitmap_has(self.map, bitpos):
            pass

    async def has(self, key):
        await self.get(key) != None
        pass

    async def delete(self, key):
        if not isinstance(key, bytes):
            key = text_encoder.encode(key)
        pass

    def is_serializable(self, serializable):
        if isinstance(serializable, list):
            return len(serializable) == 2 and isinstance(serializable[0], bytes) and isinstance(serializable[1], list)
        else:
            return self.is_root_serializable(self, serializable)

    def is_root_serializable(self, serializable):
        return self.is_serializable(self, serializable['hamt'])
        #return type(serializable) == dict and isinstance(serializable['hash_alg'], int) and isinstance(serializable['bucket_size'], int) and isinstance(serializable['hamt'], list)

    def from_serializable(self, store, id, serializable, options, depth=0):
        if depth == 0: 
            if not self.is_root_serializable(self, serializable):
                raise Exception('Loaded  object does not appear to be an IAMap root (depth==0)')
            # don't use passed-in options 
            options = serializable_to_options(serializable)
            hamt = serializable['hamt']
        else:
            if not self.is_serializable(self, serializable):
                raise Exception('Loaded object does not appear to be an IAMap node (depth>0)')
            hamt = serializable['hamt']
        #data = hamt[1]
        #need to write anonymous function for line 1130 of code from iamap.js (.map on the hamt[1])
    
    def from_child_serializable(self, id, serializable, depth):
        return self.from_serializable(self.store, id, serializable, self.config, depth)
    
    async def is_invariant(self):
        size = await self.size()
        entry_arity = self.direct_entry_count()
        node_arity = self.direct_node_count()
        arity = entry_arity + node_arity 
        size_predicate = 2 
        if node_arity == 0:
            size_predicate = min(2, entry_arity)
        
        inv1 = size - entry_arity >= 2 * (arity - entry_arity)
        #to do invars 2-4
        inv5 = node_arity >=0 and entry_arity >= 0 and (entry_arity + node_arity == arity)

        return inv1 and inv5
    
    async def size(self):
        c = 0 
        for e in self.data:
            if e.bucket is not None: 
                c += len(e.bucket) 
            else: 
                child = await self.load(self.store, e.link, self.depth + 1, self.config)
                c += await child.size()
        return c 

async def update_bucket(node, element_at, bucket_at, key, value):
    old_element = node['data'][element_at]

def build_config(options):
    config = {} 
    if not isinstance(options['hash_alg'], int):
        raise TypeError('Invalid hash_alg option')

    if options['hash_alg'] not in hasher_registry.keys():
        raise Exception(f'Unkown "hash_alg": {options["hash_alg"]}')

    config['hash_alg'] = options['hash_alg']
    #to do: check bitwidth/buckketsize, for now just defaulting 
    config['bit_with'] = default_bit_width
    config['bucket_size'] = default_bucket_size

#*
def serializable_to_options(serializable):
    return {
        'hash_alg': serializable['hash_alg'], 
        'bit_width': math.log2(serializable['hamt'][0].length*8), # inverse of (2**bit_width) / 8
        'bucket_size': serializable['bucket_size'],
            }

#*
def is_iamap(node):
    return type(node) is IAMap

async def save(store, new_node):
    id = await store.save(new_node.to_serializable())
    new_node.id = id 
    return new_node

def find_element(node, bitpos, key):
    element_at = index(node.map, bitpos)
    element = node.data[element_at]
    if element.bucket: 
        for bucket_index in range(len(element.bucket)):
            bucket_entry = element.bucket[bucket_index]
            if byte_compare(bucket_entry.key, key) == 0: 
                return {'data': { 'found': True, 'element_at': element_at, 'element': element, 'bucket_index': bucket_index, 'bucket_entry': bucket_entry}}
        return {'data': {'found': False, 'element_at': element_at, 'element': element}}

    return {'link': {'element_at': element_at, 'element': element}}

#*
def hasher(map):
    return hasher_registry[map.config['hash_alg']]['hasher']

#*
def byte_compare(b1, b2): 
    x = len(b1)
    y = len(b2)

    for i in range(min(x, y)): 
        if b1[i] != b2[i]: 
            x = b1[i]
            y = b2[i]
            break
    
    if x < y: 
        return -1 
    
    if x > y: 
        return 1 
    
    return 0

