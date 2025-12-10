from functools import wraps
import logging
from memory_profiler import memory_usage

# Create a custom logger for memory usage
logger = logging.getLogger('memory_logger')
logger.setLevel(logging.INFO)
# Writes to memory_use.log
fh = logging.FileHandler('memory_use.log')
fh.setLevel(logging.INFO)
# Message formatting
formatter = logging.Formatter('%(asctime)s — %(message)s')
fh.setFormatter(formatter)
logger.addHandler(fh)

def log_peak_memory(node_number=0):
    """
    A decorator to track maximum memory usage in a function.
    For this project's purposes, this should be added to the
    process_batch function of each node.
    
    Usage
    @log_peak_memory(node_number)
    def process_batch(...):
    """
    def fn_decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            # The memory_usage API runs 'fn' and returns a tuple of (max memory usage, fn return value)
            peak_mem, retval = memory_usage((fn, args, kwargs), max_usage=True, retval=True)
            msg = f"Node {node_number}, {fn.__name__} peak memory: {peak_mem} MiB"
            logger.info(msg)
            return retval
        return wrapper
    return fn_decorator
