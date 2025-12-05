import asyncio
import sqlite3
import time
from typing import List

import faiss
import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import Settings
from .models import GenerationBatch, GenerationItem, RetrievalBatch, RetrievalItem
from .utils import opportunistic_batch, resolve_device, write_metrics_row


class RetrievalProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.prefer_gpu, settings.only_cpu)
        self.model = SentenceTransformer("BAAI/bge-base-en-v1.5", device=self.device)
        self.index = faiss.read_index(self.settings.faiss_index_path)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base").to(self.device)
        self.reranker_model.eval()
        self.conn = sqlite3.connect(self.settings.documents_db_path, check_same_thread=False)

    def process_batch(self, items: List[RetrievalItem]) -> List[GenerationItem]:
        queries = [item.query for item in items]
        qids = [item.request_id for item in items]
        starts = [item.start_time for item in items]

        embeddings_start = time.time()
        query_embeddings = self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype("float32")
        embeddings_end = time.time()

        faiss_start = embeddings_end
        _, indices = self.index.search(query_embeddings, self.settings.retrieval_k)
        faiss_end = time.time()

        fetch_start = faiss_end
        documents_batch = self._fetch_documents(indices)
        fetch_end = time.time()

        rerank_start = fetch_end
        reranked_docs_batch = self._rerank_documents_batch(queries, documents_batch)
        retrieval_finished_at = time.time()

        stage_timings = {
            "stage_embeddings": embeddings_end - embeddings_start,
            "stage_faiss_search": faiss_end - faiss_start,
            "stage_fetch_documents": fetch_end - fetch_start,
            "stage_rerank": retrieval_finished_at - rerank_start,
        }

        output: List[GenerationItem] = []
        for idx, docs in enumerate(reranked_docs_batch):
            doc_ids = [doc["doc_id"] for doc in docs[:3]]
            output.append(
                GenerationItem(
                    request_id=qids[idx],
                    query=queries[idx],
                    reranked_doc_ids=doc_ids,
                    start_time=starts[idx],
                    retrieval_finished_at=retrieval_finished_at,
                    **stage_timings,
                )
            )
        return output

    def _fetch_documents(self, doc_id_batches: np.ndarray) -> List[List[dict]]:
        cursor = self.conn.cursor()
        documents_batch: List[List[dict]] = []
        for doc_ids in doc_id_batches:
            documents: List[dict] = []
            for doc_id in doc_ids:
                cursor.execute(
                    'SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?',
                    (doc_id,)
                )
                result = cursor.fetchone()
                if result:
                    documents.append({
                        "doc_id": result[0],
                        "title": result[1],
                        "content": result[2],
                        "category": result[3],
                    })
            documents_batch.append(documents)
        return documents_batch

    def _rerank_documents_batch(self, queries: List[str], documents_batch: List[List[dict]]) -> List[List[dict]]:
        reranked_batches: List[List[dict]] = []
        for query, documents in zip(queries, documents_batch):
            if not documents:
                reranked_batches.append([])
                continue
            pairs = [[query, doc["content"]] for doc in documents]
            with torch.inference_mode():
                inputs = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.settings.truncate_length,
                ).to(self.device)
                scores = (self.reranker_model(**inputs, return_dict=True)
                    .logits.view(-1)
                    .float()
                    .cpu()
                    .numpy()
                )
            doc_scores = list(zip(documents, scores))
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            reranked_batches.append([doc for doc, _ in doc_scores])
        return reranked_batches


class Node1State:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.queue: asyncio.Queue[RetrievalItem] = asyncio.Queue()
        self.processor = RetrievalProcessor(settings)
        self.client = httpx.AsyncClient(timeout=settings.http_timeout)


async def _send_generation_batch(state: Node1State, batch: List[GenerationItem]) -> None:
    payload = GenerationBatch(batch=batch).model_dump()
    for attempt in range(3):
        try:
            await state.client.post(f"http://{state.settings.node_2_ip}/enqueue_generation_batch", json=payload)
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))


async def worker_loop(state: Node1State) -> None:
    loop = asyncio.get_running_loop()
    while True:
        batch = await opportunistic_batch(
            state.queue, state.settings.max_batch_size_1, state.settings.batch_timeout_1
        )
        try:
            generation_batch = await loop.run_in_executor(
                None, state.processor.process_batch, batch
            )
        except Exception as exc:
            raise RuntimeError(f"retrieval batch failed: {exc}") from exc # Fail fast so Node 0 can retry
        try:
            await _send_generation_batch(state, generation_batch)
            _log_metrics(state, generation_batch)
        except Exception:
            # Requeue on send failure
            for item in batch:
                await state.queue.put(item)


def build_app(settings: Settings) -> FastAPI:
    state = Node1State(settings)
    app = FastAPI(title="Node1 Retrieval", version="1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.tasks = [asyncio.create_task(worker_loop(state))]

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for task in getattr(app.state, "tasks", []):
            task.cancel()
        await state.client.aclose()

    @app.post("/enqueue_retrieval_batch")
    async def enqueue_retrieval_batch(batch: RetrievalBatch) -> dict:
        for item in batch.batch:
            await state.queue.put(item)
        return {"queued": len(batch.batch)}

    @app.get("/health")
    async def health() -> dict:
        return {
            "status": "healthy",
            "node": settings.node_number,
            "queued": state.queue.qsize(),
            "device": str(state.processor.device),
        }

    return app


def _log_metrics(state: Node1State, generation_batch: List[GenerationItem]) -> None:
    if not state.settings.metrics_enabled or not generation_batch:
        return
    path = state.settings.metrics_csv_path
    for item in generation_batch:
        retrieval_duration = None
        if item.retrieval_finished_at:
            retrieval_duration = item.retrieval_finished_at - item.start_time
        elif all(
            stage is not None
            for stage in [
                item.stage_embeddings,
                item.stage_faiss_search,
                item.stage_fetch_documents,
                item.stage_rerank,
            ]
        ):
            retrieval_duration = (
                item.stage_embeddings
                + item.stage_faiss_search
                + item.stage_fetch_documents
                + item.stage_rerank
            )

        node_latency = retrieval_duration
        node_throughput = (1.0 / node_latency) if node_latency else None

        row = {
            "request_id": item.request_id,
            "start_time": item.start_time,
            "retrieval_finished_at": item.retrieval_finished_at,
            "generation_finished_at": None,
            "retrieval_duration": retrieval_duration,
            "generation_duration": None,
            "total_processing_time": retrieval_duration,
            "sentiment": "",
            "is_toxic": "",
            "node_number": state.settings.node_number,
            "stage_embeddings": item.stage_embeddings,
            "stage_faiss_search": item.stage_faiss_search,
            "stage_fetch_documents": item.stage_fetch_documents,
            "stage_rerank": item.stage_rerank,
            "stage_generate": None,
            "stage_sentiment": None,
            "stage_safety_filter": None,
            "node_latency": node_latency,
            "node_throughput_rps": node_throughput,
        }
        try:
            write_metrics_row(path, row)
        except Exception:
            continue
