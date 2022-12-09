import requests
import dag_cbor
from multiformats import CID
import sys
import cbor2

sys.path.append("..")
from iamap import create, TextEncoder, load, save


def default_encoder(encoder, value):
    encoder.encode(cbor2.CBORTag(42, b"\x00" + bytes(value)))


def tag_hook(decoder, tag):
    if tag.tag != 42:
        return tag
    return CID.decode(tag.value[1:])


class IPFSStore:
    def save(self, obj):
        obj = dag_cbor.encode(obj)
        res = requests.post(
            "http://localhost:5001/api/v0/dag/put",
            params={"store-codec": "dag-cbor", "input-codec": "dag-cbor", "pin": False},
            files={"dummy": obj},
        )
        res.raise_for_status()
        return CID.decode(res.json()["Cid"]["/"])

    def load(self, id: CID):
        res = requests.post(
            "http://localhost:5001/api/v0/block/get", params={"arg": str(id)}
        )
        res.raise_for_status()
        return dag_cbor.decode(res.content)

    def is_equal(self, id1: CID, id2: CID):
        return str(id1) == str(id2)

    def is_link(self, obj: CID):
        return isinstance(obj, CID) and obj.codec.name == "dag-cbor"


store = IPFSStore()

def add_all_keys(iamap):
    kvs = {str(i): i for i in range(100, 200)}
    for k in kvs:
        iamap = iamap.set(k, kvs[k])
    return iamap


def main(store):
    res = requests.post(
        "http://localhost:5001/api/v0/block/get",
        params={"arg": "bafyreif3v2k5ibkrtlu5db4qu2avvltlks7ocr6twddkp6saxqc3it5l3i"},
    )   
    d = cbor2.loads(res.content, tag_hook=tag_hook)
    
    iamap = create(store)
    iamap = add_all_keys(iamap)

    import ipdb; ipdb.set_trace()
    # s = new_iamap.size()
    # import ipdb; ipdb.set_trace()

main(store)