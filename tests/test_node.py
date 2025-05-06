from py_hamt.hamt import Node
from hypothesis import given

from testing_utils import key_value_list


@given(buckets_data=key_value_list, links_data=key_value_list)
def test_node(buckets_data, links_data):
    node = Node()
    buckets = dict()
    for k, v in buckets_data:
        buckets[k] = v
    links = dict()
    for k, v in links_data:
        links[k] = v
    node.data = [buckets, links]

    new_node = Node.deserialize(node.serialize())
    assert buckets == new_node.get_buckets()
    assert links == new_node.get_links()
