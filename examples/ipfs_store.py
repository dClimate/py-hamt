import asyncio
import requests
import dag_cbor
from multiformats import CID
import sys
import cbor2
sys.path.append("..")
from iamap import create, TextEncoder, load, save

def default_encoder(encoder, value):
    encoder.encode(cbor2.CBORTag(42,  b'\x00' + bytes(value)))

class IPFSStore:
    async def save(self, obj):
        obj = cbor2.dumps(obj, default=default_encoder)
        res = requests.post(
            "http://localhost:5001/api/v0/dag/put",
            params={"store-codec": "dag-cbor", "input-codec": "dag-cbor", "pin": False},
            files={"dummy": obj},
        )
        res.raise_for_status()
        return CID.decode(res.json()["Cid"]["/"])

    async def load(self, id: CID):
        res = requests.post(
            "http://localhost:5001/api/v0/block/get",
            params ={"arg": str(id)}
        )
        res.raise_for_status()
        return cbor2.loads(res.content)

    def is_equal(self, id1: CID, id2: CID):
        return id1 == id2

    def is_link(self, obj):
        return isinstance(obj, int)
        
store = IPFSStore()

# async def add_all_keys(cur_map, d):
#     for k in d:
#         if isinstance(d[k], dict):
#             print('adding dict')
#             cur_map = await add_all_keys(cur_map, d[k])
#         else:
#             print("adding key")
#             print(k)
#             cur_map = await cur_map.set(bytes(k, encoding="utf-8"), d[k])
#     return cur_map
    

async def add_all_keys(map, dct):
    stack = [dct]
    while stack:  # non-empty
        d = stack.pop()
        for k, v in d.items():
            if isinstance(v, dict):
                stack.append(v)
            else:
                map = await map.set(bytes(k, encoding="utf-8"), v)
    return map
    
async def main(store): 
    res = requests.post(
        "http://localhost:5001/api/v0/block/get",
        params ={"arg": "bafyreif3v2k5ibkrtlu5db4qu2avvltlks7ocr6twddkp6saxqc3it5l3i"}
    )
    d = cbor2.loads(res.content)
    iamap = await create(store)
    new_iamap = await add_all_keys(iamap, d)
    # new_map = await iamap.set(b"a", "random value")
    # val  = await new_map.get(b"a")
    import ipdb; ipdb.set_trace()

asyncio.run(main(store))