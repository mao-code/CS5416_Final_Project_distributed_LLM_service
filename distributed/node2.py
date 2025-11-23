import asyncio
import csv
import os
import sqlite3
import time
from typing import Dict, List

import httpx
import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline as hf_pipeline

from .config import Settings
from .models import GenerationBatch, GenerationItem, PipelineResult, ResultBatch
from .utils import chunked, opportunistic_batch, resolve_device


class GenerationProcessor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = resolve_device(settings.prefer_gpu, settings.only_cpu)
        self.device_index = 0 if self.device.type == "cuda" else -1
        self.conn = sqlite3.connect(
            self.settings.documents_db_path, check_same_thread=False
        )

        # Heavy models loaded once
        self.llm_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct"
        ).to(self.device)
        self.sentiment = hf_pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            device=self.device_index,
        )
        self.toxicity = hf_pipeline(
            "text-classification",
            model="unitary/toxic-bert",
            device=self.device_index,
        )

    def process_batch(self, items: List[GenerationItem]) -> List[PipelineResult]:
        docs = self._fetch_documents(items)
        prompts = [self._build_prompt(item.query, docs.get(item.request_id, [])) for item in items]

        generate_start = time.time()
        completions = self._generate(prompts)
        generate_end = time.time()

        sentiment_start = generate_end
        sentiments = self._sentiment(completions)
        sentiment_end = time.time()

        safety_start = sentiment_end
        toxicity = self._toxicity(completions)
        safety_end = time.time()

        stage_timings = {
            "stage_generate": generate_end - generate_start,
            "stage_sentiment": sentiment_end - sentiment_start,
            "stage_safety_filter": safety_end - safety_start,
        }

        results: List[PipelineResult] = []
        metrics_rows: List[dict] = []
        generation_finished_at_ts = safety_end

        for idx, item in enumerate(items):
            generation_finished_at = generation_finished_at_ts
            retrieval_finished_at = item.retrieval_finished_at or generation_finished_at
            processing_time = generation_finished_at - item.start_time
            retrieval_duration = (
                retrieval_finished_at - item.start_time if item.start_time else None
            )
            generation_duration = (
                generation_finished_at - retrieval_finished_at
                if retrieval_finished_at
                else None
            )
            results.append(
                PipelineResult(
                    request_id=item.request_id,
                    generated_response=completions[idx],
                    sentiment=sentiments[idx],
                    is_toxic="true" if toxicity[idx] else "false",
                    start_time=item.start_time,
                    processing_time=processing_time,
                    retrieval_finished_at=retrieval_finished_at,
                    generation_finished_at=generation_finished_at,
                    retrieval_duration=retrieval_duration,
                    generation_duration=generation_duration,
                    stage_embeddings=item.stage_embeddings,
                    stage_faiss_search=item.stage_faiss_search,
                    stage_fetch_documents=item.stage_fetch_documents,
                    stage_rerank=item.stage_rerank,
                    stage_generate=stage_timings["stage_generate"],
                    stage_sentiment=stage_timings["stage_sentiment"],
                    stage_safety_filter=stage_timings["stage_safety_filter"],
                )
            )
            metrics_rows.append(
                {
                    "request_id": item.request_id,
                    "start_time": item.start_time,
                    "retrieval_finished_at": retrieval_finished_at,
                    "generation_finished_at": generation_finished_at,
                    "retrieval_duration": retrieval_duration,
                    "generation_duration": generation_duration,
                    "total_processing_time": processing_time,
                    "sentiment": sentiments[idx],
                    "is_toxic": "true" if toxicity[idx] else "false",
                    "stage_embeddings": item.stage_embeddings,
                    "stage_faiss_search": item.stage_faiss_search,
                    "stage_fetch_documents": item.stage_fetch_documents,
                    "stage_rerank": item.stage_rerank,
                    "stage_generate": stage_timings["stage_generate"],
                    "stage_sentiment": stage_timings["stage_sentiment"],
                    "stage_safety_filter": stage_timings["stage_safety_filter"],
                    "node_number": self.settings.node_number,
                    "node_latency": generation_duration,
                    "node_throughput_rps": (1.0 / generation_duration) if generation_duration else None,
                }
            )
        self._log_metrics(metrics_rows)
        return results

    def _fetch_documents(self, items: List[GenerationItem]) -> Dict[str, List[dict]]:
        cursor = self.conn.cursor()
        resolved: Dict[str, List[dict]] = {}
        for item in items:
            docs: List[dict] = []
            for doc_id in item.reranked_doc_ids:
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
            resolved[item.request_id] = docs
        return resolved

    def _build_prompt(self, query: str, docs: List[dict]) -> str:
        context = "\n".join(
            f"- {doc['title']}: {doc['content'][:200]}"
            for doc in docs[: self.settings.rerank_top_k]
        )
        return f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    def _generate(self, prompts: List[str]) -> List[str]:
        outputs: List[str] = []
        for chunk in chunked(prompts, self.settings.llm_max_batch):
            model_inputs = self.llm_tokenizer(
                list(chunk),
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.settings.truncate_length,
            ).to(self.device)
            with torch.inference_mode():
                generated = self.llm_model.generate(
                    **model_inputs,
                    max_new_tokens=self.settings.max_tokens,
                    temperature=0.01,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                )
            decoded = self.llm_tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            outputs.extend(decoded)
        return outputs

    def _sentiment(self, texts: List[str]) -> List[str]:
        truncated = [text[: self.settings.truncate_length] for text in texts]
        raw = self.sentiment(truncated)
        mapping = {
            "1 star": "very negative",
            "2 stars": "negative",
            "3 stars": "neutral",
            "4 stars": "positive",
            "5 stars": "very positive",
        }
        return [mapping.get(item["label"], "neutral") for item in raw]

    def _toxicity(self, texts: List[str]) -> List[bool]:
        truncated = [text[: self.settings.truncate_length] for text in texts]
        raw = self.toxicity(truncated)
        return [entry.get("score", 0.0) > 0.5 for entry in raw]

    def _log_metrics(self, rows: List[dict]) -> None:
        if not self.settings.metrics_enabled or not rows:
            return
        path = self.settings.metrics_csv_path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        write_header = not os.path.exists(path)
        fieldnames = [
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
        try:
            with open(path, "a", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerows(rows)
        except Exception:
            # Logging should never break the main pipeline
            return


class Node2State:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.queue: asyncio.Queue[GenerationItem] = asyncio.Queue()
        self.processor = GenerationProcessor(settings)
        self.client = httpx.AsyncClient(timeout=settings.http_timeout)


async def _send_results(state: Node2State, results: List[PipelineResult]) -> None:
    payload = ResultBatch(results=results).model_dump()
    url = state.settings.node_url(0, "/result")
    for attempt in range(3):
        try:
            await state.client.post(url, json=payload)
            return
        except Exception:
            if attempt == 2:
                raise
            await asyncio.sleep(0.2 * (attempt + 1))


async def worker_loop(state: Node2State) -> None:
    loop = asyncio.get_running_loop()
    while True:
        batch = await opportunistic_batch(
            state.queue, state.settings.max_batch_size_2, state.settings.batch_timeout_2
        )
        try:
            results = await loop.run_in_executor(
                None, state.processor.process_batch, batch
            )
            await _send_results(state, results)
        except Exception:
            for item in batch:
                await state.queue.put(item)
            await asyncio.sleep(0.5)


def build_app(settings: Settings) -> FastAPI:
    state = Node2State(settings)
    app = FastAPI(title="Node2 Generation", version="1.0")

    @app.on_event("startup")
    async def _startup() -> None:
        app.state.tasks = [asyncio.create_task(worker_loop(state))]

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        for task in getattr(app.state, "tasks", []):
            task.cancel()
        await state.client.aclose()

    @app.post("/enqueue_generation_batch")
    async def enqueue_generation_batch(batch: GenerationBatch) -> dict:
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
