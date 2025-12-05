# Use asyncio to have non-blocking thread, event-loop to achieve high concurrency
import asyncio
import time
from typing import Dict

import httpx
from fastapi import FastAPI, HTTPException

from .config import Settings
from .models import (
    PipelineResult,
    QueryRequest,
    RetrievalBatch,
    RetrievalItem,
    ResultBatch,
)
from .utils import opportunistic_batch, write_metrics_row


class Node0State:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.queue: asyncio.Queue[RetrievalItem] = asyncio.Queue()
        self.pending: Dict[str, asyncio.Future] = {}
        self.pending_lock = asyncio.Lock()
        self.client = httpx.AsyncClient(timeout=settings.http_timeout)


async def _send_to_node1(state: Node0State, batch: list[RetrievalItem]) -> None:
    payload = RetrievalBatch(batch=batch).model_dump()
    try:
        await state.client.post(f"http://{state.settings.node_1_ip}/enqueue_retrieval_batch", json=payload)
    except Exception as exc:  # network failure -> fail futures
        async with state.pending_lock:
            for item in batch:
                fut = state.pending.pop(item.request_id, None)
                if fut and not fut.done():
                    fut.set_exception(RuntimeError(f"node1 dispatch failed: {exc}"))


async def dispatcher_loop(state: Node0State) -> None:
    while True:
        batch = await opportunistic_batch(
            state.queue, state.settings.max_batch_size_0, state.settings.batch_timeout_0
        )
        await _send_to_node1(state, batch)


def build_app(settings: Settings) -> FastAPI:
    state = Node0State(settings)
    app = FastAPI(title="Node0 Gateway", version="1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.tasks = [asyncio.create_task(dispatcher_loop(state))]

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for task in getattr(app.state, "tasks", []):
            task.cancel()
        await state.client.aclose()

    @app.post("/query")
    async def query(request: QueryRequest) -> PipelineResult:
        now = time.time()
        item = RetrievalItem(
            request_id=request.request_id, query=request.query, start_time=now
        )
        async with state.pending_lock:
            if request.request_id in state.pending:
                fut = state.pending[request.request_id]
            else:
                fut = asyncio.get_running_loop().create_future()
                state.pending[request.request_id] = fut
                await state.queue.put(item)

        try:
            result = await asyncio.wait_for(fut, timeout=settings.request_timeout)
        except asyncio.TimeoutError:
            async with state.pending_lock:
                state.pending.pop(request.request_id, None)
            raise HTTPException(status_code=504, detail="request timed out")
        except Exception as exc:
            async with state.pending_lock:
                state.pending.pop(request.request_id, None)
            raise HTTPException(status_code=500, detail=str(exc))

        if isinstance(result, PipelineResult):
            return result
        return PipelineResult(**result)

    @app.post("/result")
    async def result(batch: ResultBatch) -> dict:
        delivered = 0
        async with state.pending_lock:
            for res in batch.results:
                fut = state.pending.pop(res.request_id, None)
                if fut and not fut.done():
                    fut.set_result(res)
                    delivered += 1
                _log_metrics(state, res)
        return {"delivered": delivered}

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "healthy",
            "node": settings.node_number,
            "queued": state.queue.qsize(),
        }

    return app


def _log_metrics(state: Node0State, res: PipelineResult) -> None:
    if not state.settings.metrics_enabled:
        return
    path = state.settings.metrics_csv_path
    stage_sum = lambda names: sum(  # noqa: E731
        v for v in (getattr(res, name, None) for name in names) if v is not None
    )

    retrieval_duration = res.retrieval_duration
    if retrieval_duration is None:
        retrieval_duration = stage_sum(
            ["stage_embeddings", "stage_faiss_search", "stage_fetch_documents", "stage_rerank"]
        ) or None

    generation_duration = res.generation_duration
    if generation_duration is None and res.generation_finished_at and res.retrieval_finished_at:
        generation_duration = res.generation_finished_at - res.retrieval_finished_at
    if generation_duration is None:
        generation_duration = stage_sum(
            ["stage_generate", "stage_sentiment", "stage_safety_filter"]
        ) or None

    total_processing_time = res.processing_time
    if total_processing_time is None and res.start_time and res.generation_finished_at:
        total_processing_time = res.generation_finished_at - res.start_time

    retrieval_finished_at = res.retrieval_finished_at
    if retrieval_finished_at is None and res.start_time and retrieval_duration:
        retrieval_finished_at = res.start_time + retrieval_duration

    generation_finished_at = res.generation_finished_at
    if generation_finished_at is None and res.start_time and total_processing_time:
        generation_finished_at = res.start_time + total_processing_time

    node_latency = total_processing_time
    node_throughput = (1.0 / node_latency) if node_latency else None

    row = {
        "request_id": res.request_id,
        "start_time": res.start_time,
        "retrieval_finished_at": retrieval_finished_at,
        "generation_finished_at": generation_finished_at,
        "retrieval_duration": retrieval_duration,
        "generation_duration": generation_duration,
        "total_processing_time": total_processing_time,
        "sentiment": res.sentiment,
        "is_toxic": res.is_toxic,
        "node_number": state.settings.node_number,
        "stage_embeddings": getattr(res, "stage_embeddings", None),
        "stage_faiss_search": getattr(res, "stage_faiss_search", None),
        "stage_fetch_documents": getattr(res, "stage_fetch_documents", None),
        "stage_rerank": getattr(res, "stage_rerank", None),
        "stage_generate": getattr(res, "stage_generate", None),
        "stage_sentiment": getattr(res, "stage_sentiment", None),
        "stage_safety_filter": getattr(res, "stage_safety_filter", None),
        "node_latency": node_latency,
        "node_throughput_rps": node_throughput,
    }

    try:
        write_metrics_row(path, row)
    except Exception:
        return
