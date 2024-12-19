from py_hamt import HAMT, IPFSStore
import time


# Run this with varying cache sizes to see the impact on performance of the cache when using IPFSStore()
def test_and_print_perf():
    # usual cache size
    hamt = HAMT(store=IPFSStore())
    start_time = time.perf_counter()
    num_ops = 50
    for key_int in range(num_ops):
        hamt[str(key_int)] = key_int
    end_time = time.perf_counter()
    cache_time = end_time - start_time

    # no cache
    hamt = HAMT(store=IPFSStore(), max_cache_size_bytes=0)
    start_time = time.perf_counter()
    for key_int in range(num_ops):
        hamt[str(key_int)] = key_int
    end_time = time.perf_counter()
    no_cache_time = end_time - start_time

    print(f"Improvement of {(1 - cache_time / no_cache_time) * 100:.2f}%")
