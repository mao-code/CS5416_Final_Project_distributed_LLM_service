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
from .utils import opportunistic_batch


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
        await state.client.post(
            state.settings.node_url(1, "/enqueue_retrieval_batch"), json=payload
        )
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
        return {"delivered": delivered}

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "healthy",
            "node": settings.node_number,
            "queued": state.queue.qsize(),
        }

    return app
