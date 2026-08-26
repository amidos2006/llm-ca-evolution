from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import os
import random
import threading

DEFAULT_LIMIT = 8
DEFAULT_PROCESSES = 4
BASE_BACKOFF = 1.0
MAX_BACKOFF = 30.0

_limit = DEFAULT_LIMIT
_semaphore = threading.BoundedSemaphore(DEFAULT_LIMIT)
_processes = DEFAULT_PROCESSES
_pool = None
_pool_size = None


def configure(limit, processes=None):
    # limit caps the API calls in flight, processes caps the workers used for CPU work.
    global _limit, _semaphore, _processes
    _limit = max(1, int(limit))
    _semaphore = threading.BoundedSemaphore(_limit)
    if processes is not None:
        processes = int(processes)
        # 0 or less means take whatever the machine has.
        _processes = processes if processes > 0 else (os.cpu_count() or 1)


def api_slot():
    # Use as a context manager around a single API call.
    return _semaphore


def backoff_delay(attempt):
    # Exponential backoff with jitter so retries from different threads do not sync up.
    delay = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** max(0, attempt - 1)))
    return delay * (0.5 + 0.5 * random.random())


def run_parallel(function, items):
    # Map over items on a private pool of threads and keep the input order. For work that
    # waits on the network. Nesting these is safe because api_slot, not the pool size, is
    # what throttles the API.
    items = list(items)
    if len(items) <= 1:
        return [function(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(len(items), _limit)) as pool:
        return list(pool.map(function, items))


def process_pool():
    # One pool for the whole process, rebuilt only when the worker count changes. Keeping it
    # alive means workers pay their import and setup cost once instead of once per batch.
    global _pool, _pool_size
    if _pool is not None and _pool_size != _processes:
        reset_processes()
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=_processes)
        _pool_size = _processes
    return _pool


def reset_processes():
    global _pool, _pool_size
    if _pool is not None:
        _pool.shutdown(wait=False)
    _pool, _pool_size = None, None


def run_processes(function, items):
    # Same contract as run_parallel but for work that burns CPU, so it sidesteps the GIL.
    # The function and every item have to be picklable.
    items = list(items)
    if len(items) <= 1 or _processes == 1:
        return [function(item) for item in items]
    try:
        return list(process_pool().map(function, items))
    except BrokenProcessPool:
        # Generated code runs in the workers, so a hard crash there drops back to this process.
        print("Worker process died, evaluating this batch sequentially")
        reset_processes()
        return [function(item) for item in items]
