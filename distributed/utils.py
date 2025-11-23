import asyncio
import csv
import os
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


# Timing logic helpers (new)
METRIC_FIELDNAMES = [
    "request_id",
    "start_time",
    "retrieval_finished_at",
    "generation_finished_at",
    "retrieval_duration",
    "generation_duration",
    "total_processing_time",
    "sentiment",
    "is_toxic",
    "node_number",
    "stage_embeddings",
    "stage_faiss_search",
    "stage_fetch_documents",
    "stage_rerank",
    "stage_generate",
    "stage_sentiment",
    "stage_safety_filter",
    "node_latency",
    "node_throughput_rps",
]


def write_metrics_row(path: str, row: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=METRIC_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
