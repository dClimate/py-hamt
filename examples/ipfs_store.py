import requests
import dag_cbor
import cbor2
from multiformats import CID

class HamtIPFSStore:
    def save(self, obj):
        obj = dag_cbor.encode(obj)
        res = requests.post(
            "http://localhost:5001/api/v0/dag/put",
            params={"store-codec": "dag-cbor", "input-codec": "dag-cbor", "pin": False},
            files={"dummy": obj},
        )
        res.raise_for_status()
        return CID.decode(res.json()["Cid"]["/"])

    def load(self, id):
        if isinstance(id, cbor2.CBORTag):
            id = CID.decode(id.value[1:])
        res = requests.post(
            "http://localhost:5001/api/v0/block/get", params={"arg": str(id)}
        )
        res.raise_for_status()
        return dag_cbor.decode(res.content)

    def is_equal(self, id1: CID, id2: CID):
        return str(id1) == str(id2)

    def is_link(self, obj: CID):
        return isinstance(obj, CID) and obj.codec.name == "dag-cbor"
