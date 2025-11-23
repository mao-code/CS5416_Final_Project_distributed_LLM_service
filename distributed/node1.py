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
from .utils import opportunistic_batch, resolve_device


class RetrievalProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.prefer_gpu, settings.only_cpu)
        self.embedder = SentenceTransformer(
            "BAAI/bge-base-en-v1.5", device=str(self.device)
        )
        self.index = faiss.read_index(self.settings.faiss_index_path)
        self.reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
        self.reranker_model = AutoModelForSequenceClassification.from_pretrained(
            "BAAI/bge-reranker-base"
        ).to(self.device)
        self.reranker_model.eval()
        self.conn = sqlite3.connect(
            self.settings.documents_db_path, check_same_thread=False
        )

    def process_batch(self, items: List[RetrievalItem]) -> List[GenerationItem]:
        queries = [item.query for item in items]
        qids = [item.request_id for item in items]
        starts = [item.start_time for item in items]

        embeddings = self.embedder.encode(
            queries, normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")

        _, doc_indices = self.index.search(embeddings, self.settings.retrieval_k)
        documents_batch = self._fetch_documents(doc_indices)
        reranked_docs = self._rerank(queries, documents_batch)
        retrieval_finished_at = time.time()

        output: List[GenerationItem] = []
        for idx, docs in enumerate(reranked_docs):
            doc_ids = [doc["doc_id"] for doc in docs[: self.settings.rerank_top_k]]
            output.append(
                GenerationItem(
                    request_id=qids[idx],
                    query=queries[idx],
                    reranked_doc_ids=doc_ids,
                    start_time=starts[idx],
                    retrieval_finished_at=retrieval_finished_at,
                )
            )
        return output

    def _fetch_documents(self, doc_idx_batch: np.ndarray) -> List[List[dict]]:
        cursor = self.conn.cursor()
        documents: List[List[dict]] = []
        for row in doc_idx_batch:
            docs: List[dict] = []
            for doc_id in row:
                cursor.execute(
                    "SELECT doc_id, title, content, category FROM documents WHERE doc_id = ?",
                    (int(doc_id),),
                )
                res = cursor.fetchone()
                if res:
                    docs.append(
                        {
                            "doc_id": res[0],
                            "title": res[1],
                            "content": res[2],
                            "category": res[3],
                        }
                    )
            documents.append(docs)
        return documents

    def _rerank(self, queries: List[str], documents_batch: List[List[dict]]) -> List[List[dict]]:
        reranked: List[List[dict]] = []
        for query, docs in zip(queries, documents_batch):
            if not docs:
                reranked.append([])
                continue
            pairs = [[query, doc["content"]] for doc in docs]
            with torch.inference_mode():
                encoded = self.reranker_tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=self.settings.truncate_length,
                ).to(self.device)
                scores = (
                    self.reranker_model(**encoded, return_dict=True)
                    .logits.view(-1)
                    .float()
                    .cpu()
                    .numpy()
                )
            merged = list(zip(docs, scores))
            merged.sort(key=lambda x: x[1], reverse=True)
            reranked.append([doc for doc, _ in merged])
        return reranked


class Node1State:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.queue: asyncio.Queue[RetrievalItem] = asyncio.Queue()
        self.processor = RetrievalProcessor(settings)
        self.client = httpx.AsyncClient(timeout=settings.http_timeout)


async def _send_generation_batch(state: Node1State, batch: List[GenerationItem]) -> None:
    payload = GenerationBatch(batch=batch).model_dump()
    url = state.settings.node_url(2, "/enqueue_generation_batch")
    for attempt in range(3):
        try:
            await state.client.post(url, json=payload)
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
            # Fail fast so Node 0 can retry
            raise RuntimeError(f"retrieval batch failed: {exc}") from exc
        try:
            await _send_generation_batch(state, generation_batch)
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
