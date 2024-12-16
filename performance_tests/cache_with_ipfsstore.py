from py_hamt import HAMT, IPFSStore


# Run this with varying cache sizes to see the impact on performance of the cache when using IPFSStore()
def test_and_print_perf():
    import time

    num_ops = 50

    # usual cache size
    hamt = HAMT(store=IPFSStore())
    start_time = time.time()
    for key_int in range(num_ops):
        hamt[str(key_int)] = key_int
    end_time = time.time()
    op_avg_cache = (end_time - start_time) / 100

    # no cache
    hamt = HAMT(store=IPFSStore(), max_cache_size_bytes=0)
    start_time = time.time()
    for key_int in range(num_ops):
        hamt[str(key_int)] = key_int
    end_time = time.time()
    op_avg_no_cache = (end_time - start_time) / 100

    print(f"Improvement of {(1 - op_avg_cache / op_avg_no_cache) * 100:.2f}%")
