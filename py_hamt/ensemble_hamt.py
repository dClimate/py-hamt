import multiprocessing
from collections.abc import MutableMapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy

from dag_cbor.ipld import IPLDKind

from .hamt import HAMT


class EnsembleHAMT(MutableMapping):
    """
    An `EnsembleHAMT` provides for much faster write speeds through parallelization, with the same MutableMapping interface for mostly drop-in replacement.

    While a normal HAMT's reads can be parallelized, due to the content-immutable assumption of the backing store, it is very difficult to perform multi-threaded writes. We overcome this with an ensemble of HAMTs. The basic idea is to split up work across an arbitrary number N of single-threaded HAMTs, and then merge them together in pairs once we are done with our writes back into a single HAMT.

    Here's an example of its use.
    ```python
    from py_hamt import EnsembleHAMT, IPFSStore

    N = 10 # the parallelization number, approximate speedup
    # Measure first to see what the speedup factor is compared to N, it will probably be nonlinear
    hamts = []
    for _ in range(0, N):
        hamts.append(HAMT(store=DictStore()))
    ensemble = EnsembleHAMT(ensemble=hamts)
    # large number of set operations
    for i in range(0, 2000000):
      ensemble[str(i)] = i
    final_hamt = HAMT(store=DictStore())
    ensemble.consolidate(final_hamt)
    # now final hamt has all of the KVs
    ```
    """

    ensemble: list[HAMT]
    """@private"""

    def __init__(self, ensemble: list[HAMT]):
        """The ensemble should be a nonempty length list of HAMTs. They will be deepcopied, so the original objects will not be mutated."""
        self.ensemble = []
        for hamt in ensemble:
            self.ensemble.append(deepcopy(hamt))

        for hamt in self.ensemble:
            hamt.enable_write()

    def __setitem__(self, key: str, val: IPLDKind):
        index = hash(key) % len(self.ensemble)
        chosen_hamt = self.ensemble[index]
        chosen_hamt[key] = val

    def __delitem__(self, key: str):
        index = hash(key) % len(self.ensemble)
        chosen_hamt = self.ensemble[index]
        del chosen_hamt[key]

    def __getitem__(self, key: str) -> IPLDKind:
        index = hash(key) % len(self.ensemble)
        chosen_hamt = self.ensemble[index]
        return chosen_hamt[key]

    def __len__(self):
        with ThreadPoolExecutor() as executor:
            return sum(executor.map(len, self.ensemble))

    def __iter__(self):
        for hamt in self.ensemble:
            yield from hamt

    def consolidate(self, target: HAMT):
        """Merges all the KVs inside all the HAMTs into the target.

        Since an EnsembleHAMT allows for varied HAMT configurations, it is important to note here the algorithm used, so that clients understand the usage patterns on the backing stores.

        This adds the input HAMT into the list of HAMTs, and then merges everything in pairs until only the target is left holding all KVs.
        """
        to_merge: list[HAMT] = [target]
        for hamt in self.ensemble:
            to_merge.append(hamt)

        while len(to_merge) > 1:
            merged: list[HAMT] = []
            for i in range(0, len(to_merge), 2):
                h1 = to_merge[i]

                # Handle odd numbered list of HAMTs
                if i != len(to_merge) - 1:
                    h2 = to_merge[i + 1]
                    # By always adding in the second hamt, we make sure that the target hamt, which is always first, remains until the end, and thus receives all of the KVs
                    h1.merge(h2)

                merged.append(h1)
            to_merge = merged
