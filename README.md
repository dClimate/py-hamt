# python-iamap

## python implementation of a HAMT, adapted from rvagg's JavaScript version

See https://github.com/rvagg/iamap#readme for details on a JS implementation of this code. This python version is adapted from rvagg's code; there are one-to-one mappings between functions/classes in this repo and those in the JS one. As a result, the JS code can serve as a canonical guide to implementation and functionality.

See https://ipld.io/specs/advanced-data-layouts/hamt/spec/ for information on the concept of a HAMT and how it fits into the IPLD/IPFS ecosystem. 


## Motivation

dClimate uses HAMTs as key/value stores that can be distributed across multiple nodes and used without the whole data structure being loaded into memory. This is extremely useful in the context of [zarrs](https://zarr.readthedocs.io/en/stable/), where metadata mapping coordinates to chunks containing the actual data can stretch into the 10s or even 100s of MBs. Because IPFS imposes a limit on the sizes of blocks that can be transferred from peer to peer, it is not feasible to store all this metadata in a single IPFS object. Instead, a HAMT can be used to provide efficient lookups in a data structure distributed across many IPFS objects, with only the parts of the HAMT needed for the lookup ever being accessed. 

See [ipldstore](https://github.com/dClimate/ipldstore) for an example of this HAMT implementation in action.
