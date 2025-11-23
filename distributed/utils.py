import asyncio
import time
from typing import Iterable, List, Sequence, TypeVar

import torch

T = TypeVar("T")


async def opportunistic_batch(
    queue: asyncio.Queue,
    max_batch_size: int,
    batch_timeout_s: float,
) -> List[T]:
    """
    Collect as many items as possible up to max_batch_size or until timeout.
    This keeps the worker busy without starving stragglers.
    """
    first = await queue.get()
    batch: List[T] = [first]
    start = time.perf_counter()
    while len(batch) < max_batch_size:
        remaining = batch_timeout_s - (time.perf_counter() - start)
        if remaining <= 0:
            break
        try:
            next_item = await asyncio.wait_for(queue.get(), timeout=remaining)
            batch.append(next_item)
        except asyncio.TimeoutError:
            break
    return batch


def resolve_device(prefer_gpu: bool, only_cpu: bool = False) -> torch.device:
    if only_cpu:
        return torch.device("cpu")
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def chunked(seq: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for idx in range(0, len(seq), size):
        yield seq[idx : idx + size]
